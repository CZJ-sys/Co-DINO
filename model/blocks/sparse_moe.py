"""Co-DINO top-k sparse MoE with communication among selected experts.

Routing happens before expert execution.  Only selected token/expert pairs run
their expert MLPs; two shared experts run for every token and are averaged.
Selected hidden states communicate before their expert-specific decoders.
"""

import torch
from torch import nn

from .experts import PointwiseExpert


class SelectedExpertCommunication(nn.Module):
    """Residual communication on the selected expert subgraph."""

    def __init__(self, num_experts):
        super().__init__()
        # Zero initialization makes the communication model start exactly as
        # the ordinary sparse MoE, then learn cross-expert residual messages.
        self.mix = nn.Parameter(torch.zeros(num_experts, num_experts))

    def forward(self, selected_hidden, selected_indices):
        # selected_hidden [T,k,H], indices [T,k]. Top-k has no duplicates, so
        # the diagonal in slot space is precisely the self-message.
        receiver = selected_indices.unsqueeze(-1)
        sender = selected_indices.unsqueeze(-2)
        coefficients = torch.tanh(self.mix[receiver, sender])
        top_k = selected_hidden.shape[1]
        off_diagonal = ~torch.eye(top_k, dtype=torch.bool,
                                  device=selected_hidden.device)[None]
        coefficients = coefficients * off_diagonal.to(coefficients.dtype)
        delta = torch.einsum("tij,tjh->tih", coefficients, selected_hidden)
        return selected_hidden + delta / max(1, top_k - 1)


class SparseCommunicationMoE(nn.Module):
    """Per-token top-k expert dispatch with two always-on shared experts."""

    def __init__(self, config, channels):
        super().__init__()
        if channels not in config.stage_dims:
            raise ValueError("channels must match one of the configured stage widths")
        hidden_dim = config.expert_hidden_dim(channels)
        self.num_routed_experts = config.num_routed_experts
        self.num_active_experts = config.num_active_experts
        self.shared_experts = nn.ModuleList([
            PointwiseExpert(channels, hidden_dim)
            for _ in range(config.num_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            PointwiseExpert(channels, hidden_dim)
            for _ in range(self.num_routed_experts)
        ])
        self.router = nn.Linear(channels, self.num_routed_experts)
        self.communication = SelectedExpertCommunication(self.num_routed_experts)
        self.last_routing_stats = None

    def _dispatch_hidden(self, flat_x, indices):
        tokens, top_k = indices.shape
        hidden_dim = self.routed_experts[0].output_projection.in_features
        selected = flat_x.new_zeros(tokens, top_k, hidden_dim)
        for expert_index, expert in enumerate(self.routed_experts):
            token_index, slot_index = torch.where(indices == expert_index)
            if token_index.numel() == 0:
                continue
            hidden = expert.encode(flat_x.index_select(0, token_index)).to(selected.dtype)
            selected = selected.index_put((token_index, slot_index), hidden)
        return selected

    def _decode_selected(self, hidden, indices, weights, channels):
        output = hidden.new_zeros(hidden.shape[0], channels)
        selected_outputs = hidden.new_zeros(hidden.shape[0], hidden.shape[1], channels)
        for expert_index, expert in enumerate(self.routed_experts):
            token_index, slot_index = torch.where(indices == expert_index)
            if token_index.numel() == 0:
                continue
            decoded = expert.decode(hidden[token_index, slot_index])
            decoded_for_accumulation = decoded.to(output.dtype)
            selected_outputs = selected_outputs.index_put(
                (token_index, slot_index), decoded_for_accumulation)
            # Linear runs in BF16/FP16 under autocast, while the scatter
            # buffers remain FP32.  index_put/index_add require exact dtype
            # matches, so all sparse accumulation is explicit FP32 here.
            weighted = decoded_for_accumulation * weights[
                token_index, slot_index, None].to(output.dtype)
            output.index_add_(0, token_index, weighted)
        return output, selected_outputs

    def _forward_impl(self, x):
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        logits = self.router(flat_x)
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_values, indices = torch.topk(
            probabilities, self.num_active_experts, dim=-1, sorted=True)
        weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        selected_hidden = self._dispatch_hidden(flat_x, indices)
        communicated_hidden = self.communication(selected_hidden, indices)
        routed, selected_outputs = self._decode_selected(
            communicated_hidden, indices, weights, original_shape[-1])

        shared_outputs = torch.stack(
            [expert(flat_x) for expert in self.shared_experts], dim=1)
        shared = shared_outputs.mean(dim=1)
        output = shared + routed
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_routed_experts).float()
        mean_probability = probabilities.float().mean(dim=0)
        selected_weight_sum = probabilities.new_zeros(self.num_routed_experts)
        selected_weight_sum.scatter_add_(0, indices.reshape(-1), weights.reshape(-1))
        state = {
            "router_probabilities": probabilities,
            "topk_indices": indices,
            "topk_weights": weights,
            "selected_hidden": selected_hidden,
            "communicated_hidden": communicated_hidden,
            "selected_outputs": selected_outputs,
            "shared_outputs": shared_outputs,
            "expert_counts": counts,
            "mean_router_probability": mean_probability,
            "selected_weight_sum": selected_weight_sum,
            "output": output,
        }
        return output.reshape(original_shape), state

    def forward(self, x, return_router_state=False):
        result, state = self._forward_impl(x)
        self.last_routing_stats = {
            "expert_counts": state["expert_counts"].detach(),
            "probability_sum": (
                state["mean_router_probability"] * state["router_probabilities"].shape[0]
            ).detach(),
            "selected_weight_sum": state["selected_weight_sum"].detach(),
            "tokens": state["router_probabilities"].shape[0],
        }
        return (result, state) if return_router_state else result

    def diagnostic_state(self, x):
        """Return routing, expert and communication state for visualization."""
        output, state = self._forward_impl(x)
        state["output"] = output
        return state


def communication_moe_modules(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, SparseCommunicationMoE)}
