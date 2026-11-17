"""Streaming visual diagnostics for Co-DINO's routed experts."""

import math

import torch

from .blocks.sparse_moe import communication_moe_modules


def communicating_moe_modules(model):
    return communication_moe_modules(model)


def representative_moe_layers(module_names):
    """Map the four requested roles to modules in actual forward-pass order."""
    parsed = {"encoder": [], "decoder": []}
    for name in module_names:
        parts = name.split(".")
        positions = [parts.index(branch) for branch in parsed if branch in parts]
        if len(positions) != 1:
            continue
        position = positions[0]
        branch = parts[position]
        suffix = parts[position + 1:]
        if len(suffix) != 4 or suffix[0] != "stages" or suffix[3] != "moe":
            continue
        try:
            stage, block = int(suffix[1]), int(suffix[2])
        except ValueError:
            continue
        parsed[branch].append((stage, block, name))
    if not parsed["encoder"] or not parsed["decoder"]:
        raise ValueError(
            "Could not identify encoder/decoder MoE layers from module names: "
            f"{sorted(module_names)}")
    # Encoder executes stages from shallow to deep. Decoder executes stages
    # from deep to shallow, while blocks within one stage retain ascending order.
    encoder = sorted(parsed["encoder"], key=lambda row: (row[0], row[1]))
    decoder = sorted(parsed["decoder"], key=lambda row: (-row[0], row[1]))
    return {
        "encoder_first": encoder[0][2],
        "encoder_last": encoder[-1][2],
        "decoder_first": decoder[0][2],
        "decoder_last": decoder[-1][2],
    }


class LayerExpertStatistics:
    def __init__(self, module):
        self.routed = module.num_routed_experts
        self.shared = len(module.shared_experts)
        self.tokens = 0
        self.counts = torch.zeros(self.routed, dtype=torch.float64)
        self.probability_sum = torch.zeros(self.routed, dtype=torch.float64)
        self.selected_weight_sum = torch.zeros(self.routed, dtype=torch.float64)
        self.hidden_norm_sum = torch.zeros(self.routed, dtype=torch.float64)
        self.output_norm_sum = torch.zeros(self.routed, dtype=torch.float64)
        self.signed_contribution_sum = torch.zeros(self.routed, dtype=torch.float64)
        self.absolute_contribution_sum = torch.zeros(self.routed, dtype=torch.float64)
        self.valid_signed_contribution_tokens = 0
        self.shared_norm_sum = torch.zeros(self.shared, dtype=torch.float64)
        self.entropy_sum = 0.0
        self.hidden_delta_square = 0.0
        self.hidden_square = 0.0

    def update(self, state):
        probabilities = state["router_probabilities"].detach().float()
        indices = state["topk_indices"].detach()
        weights = state["topk_weights"].detach().float()
        hidden = state["selected_hidden"].detach().float()
        communicated = state["communicated_hidden"].detach().float()
        outputs = state["selected_outputs"].detach().float()
        token_count = probabilities.shape[0]
        self.tokens += token_count
        self.probability_sum += probabilities.double().cpu().sum(0)
        self.entropy_sum += (-(probabilities * probabilities.clamp_min(1e-12).log())
                             .sum(-1).double().sum().item())
        weighted_outputs = outputs * weights[..., None]
        routed_output = weighted_outputs.sum(dim=1)
        denominator = routed_output.square().sum(dim=-1)
        component_scale = weighted_outputs.square().sum(dim=(-1, -2))
        valid = denominator > torch.finfo(outputs.dtype).eps * component_scale.clamp_min(
            torch.finfo(outputs.dtype).tiny)
        self.valid_signed_contribution_tokens += int(valid.sum().item())
        # These additive projection shares sum to one for every non-zero routed
        # output, but an individual expert may be negative when it opposes the
        # aggregate update. The softmax/top-k gate itself remains non-negative.
        numerator = (weighted_outputs * routed_output[:, None, :]).sum(dim=-1)
        signed_contributions = torch.where(
            valid[:, None], numerator / denominator.clamp_min(
                torch.finfo(outputs.dtype).tiny)[:, None], torch.zeros_like(numerator))
        for expert in range(self.routed):
            token_index, slot_index = torch.where(indices == expert)
            count = token_index.numel()
            if count == 0:
                continue
            self.counts[expert] += count
            self.selected_weight_sum[expert] += weights[token_index, slot_index].double().sum().cpu()
            self.hidden_norm_sum[expert] += hidden[token_index, slot_index].norm(
                dim=-1).double().sum().cpu()
            self.output_norm_sum[expert] += outputs[token_index, slot_index].norm(
                dim=-1).double().sum().cpu()
            contribution = signed_contributions[token_index, slot_index].double().cpu()
            self.signed_contribution_sum[expert] += contribution.sum()
            self.absolute_contribution_sum[expert] += contribution.abs().sum()
        shared = state["shared_outputs"].detach().float()
        self.shared_norm_sum += shared.norm(dim=-1).double().cpu().sum(0)
        self.hidden_delta_square += (communicated - hidden).square().double().sum().item()
        self.hidden_square += hidden.square().double().sum().item()

    def result(self, module):
        if self.tokens == 0:
            raise ValueError("No expert tokens were accumulated")
        total_selections = self.counts.sum().clamp_min(1.0)
        total_selected_weight = self.selected_weight_sum.sum().clamp_min(1e-24)
        experts = []
        for index in range(self.routed):
            count = self.counts[index].clamp_min(1.0)
            experts.append({
                "kind": "routed", "expert": index,
                "activation_count": int(self.counts[index].item()),
                "activation_fraction": (self.counts[index] / total_selections).item(),
                "mean_router_weight": (self.probability_sum[index] / self.tokens).item(),
                "selected_weight_fraction": (
                    self.selected_weight_sum[index] / total_selected_weight).item(),
                "mean_selected_weight": (self.selected_weight_sum[index] / count).item(),
                "mean_signed_contribution_per_token": (
                    self.signed_contribution_sum[index] / self.tokens).item(),
                "mean_signed_contribution_when_selected": (
                    self.signed_contribution_sum[index] / count).item(),
                "mean_absolute_contribution_when_selected": (
                    self.absolute_contribution_sum[index] / count).item(),
                "mean_hidden_l2": (self.hidden_norm_sum[index] / count).item(),
                "mean_output_l2": (self.output_norm_sum[index] / count).item(),
            })
        for index in range(self.shared):
            experts.append({"kind": "shared", "expert": index,
                            "mean_output_l2": (
                                self.shared_norm_sum[index] / self.tokens).item()})
        matrix = torch.tanh(module.communication.mix.detach()).double().cpu()
        matrix.fill_diagonal_(0.0)
        absolute = matrix.abs()
        row_normalized = absolute / absolute.sum(-1, keepdim=True).clamp_min(1e-24)
        off_diagonal = ~torch.eye(self.routed, dtype=torch.bool)
        total_absolute = absolute.sum().clamp_min(1e-24)
        return {
            "tokens": self.tokens,
            "top_k": int(self.counts.sum().item() / self.tokens),
            "experts": experts,
            "routing": {
                "mean_normalized_entropy": self.entropy_sum /
                    (self.tokens * math.log(self.routed)),
                "valid_signed_contribution_token_fraction":
                    self.valid_signed_contribution_tokens / self.tokens,
            },
            "communication_effect": {
                "relative_hidden_l2_change": math.sqrt(
                    self.hidden_delta_square / max(self.hidden_square, 1e-24)),
            },
            "communication_matrix": {
                "present": True,
                "signed": matrix.tolist(),
                "absolute_row_normalized": row_normalized.tolist(),
                "data_conditioned_influence_row_normalized": row_normalized.tolist(),
                "diagonal_absolute_fraction": (absolute.diag().sum() / total_absolute).item(),
                "off_diagonal_absolute_fraction": (
                    absolute[off_diagonal].sum() / total_absolute).item(),
            },
        }


class ExpertDiagnosticsCollector:
    def __init__(self, model):
        self.modules = communication_moe_modules(model)
        if not self.modules:
            raise ValueError("Checkpoint has no sparse MoE modules")
        self.current_dataset = None
        self.enabled = True
        self.statistics = {}
        self.handles = [module.register_forward_hook(self._hook(name))
                        for name, module in self.modules.items()]

    def _hook(self, name):
        def collect(module, inputs, _output):
            if not self.enabled or self.current_dataset is None:
                return
            key = (self.current_dataset, name)
            stats = self.statistics.setdefault(key, LayerExpertStatistics(module))
            with torch.no_grad():
                stats.update(module.diagnostic_state(inputs[0]))
        return collect

    def select_dataset(self, dataset):
        self.current_dataset = dataset

    def result(self):
        result = {}
        for (dataset, name), stats in self.statistics.items():
            result.setdefault(dataset, {})[name] = stats.result(self.modules[name])
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
