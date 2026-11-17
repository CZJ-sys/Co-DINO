"""The four fixed Co-DINO architecture variants."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CoDINOConfig:
    """Architecture contract shared by training, evaluation and checkpoints."""

    model_size: str
    embed_dim: int
    depths: tuple[int, ...]
    input_channels: int = 4
    output_channels: int = 4
    patch_size: int = 2
    num_shared_experts: int = 2
    num_routed_experts: int = 8
    num_active_experts: int = 4
    expert_hidden_ratio: float = 1.0
    mlp_ratio: int = 4
    fft_norm: str = "forward"
    spectral_radial_bands: int = 4
    spectral_num_angles: int = 4
    spectral_angular_width_deg: float = 22.5
    spectral_mixer_init_scale: float = 0.02

    def __post_init__(self):
        if self.model_size not in ("T", "S", "B", "L"):
            raise ValueError("model_size must be T, S, B or L")
        if len(self.depths) != 4 or any(depth <= 0 for depth in self.depths):
            raise ValueError("Co-DINO requires four non-empty stages")

    @property
    def stage_dims(self):
        return tuple(self.embed_dim * 2**stage for stage in range(4))

    @property
    def spectral_angle_centers_deg(self):
        return (0.0, 45.0, 90.0, 135.0)

    def expert_hidden_dim(self, channels):
        width = channels * self.expert_hidden_ratio
        if width <= 0 or int(width) != width:
            raise ValueError("Expert hidden width must be a positive integer")
        return int(width)

    def to_dict(self):
        result = asdict(self)
        result["depths"] = list(self.depths)
        result["stage_dims"] = list(self.stage_dims)
        result["spectral_angle_centers_deg"] = list(self.spectral_angle_centers_deg)
        result["num_experts_total"] = self.num_shared_experts + self.num_routed_experts
        result["expert_hidden_dims"] = [self.expert_hidden_dim(c) for c in self.stage_dims]
        return result


def _registry():
    path = Path(__file__).resolve().parents[1] / "configs" / "model_sizes.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_model_config(model_size="L"):
    size = str(model_size).upper()
    presets = _registry()
    if size not in presets:
        raise ValueError(f"Unknown Co-DINO size {model_size!r}; choose T, S, B or L")
    preset = presets[size]
    return CoDINOConfig(size, preset["embed_dim"], tuple(preset["depths"]))


def architecture_from_dict(raw):
    """Restore clean checkpoints and the exact selected legacy architecture."""
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint is missing the Co-DINO architecture")
    fields = set(CoDINOConfig.__dataclass_fields__)
    supplied = {key: value for key, value in raw.items() if key in fields}
    if "depths" in supplied:
        supplied["depths"] = tuple(supplied["depths"])
    normalized = dict(raw)
    # The 48M architecture was named B before the 30M B preset was added.
    if (supplied.get("model_size") == "B" and supplied.get("embed_dim") == 48
            and supplied.get("depths") == (2, 2, 2, 2)):
        supplied["model_size"] = "L"
        normalized["model_size"] = "L"
    config = CoDINOConfig(**supplied)
    preset = get_model_config(config.model_size)
    if config != preset:
        raise ValueError("Checkpoint does not use the registered Co-DINO size")
    expected = preset.to_dict()
    legacy_selected = {
        "aggregation": "shared_mean_plus_sparse_routed",
        "routing_mode": "top4_softmax",
        "communication": "selected_expert_hidden_residual",
        "spectral_mode": "radial_angular",
        "spectral_angle_offset_deg": 0.0,
        "spectral_directional_mixers": True,
        "spectral_filter_cutoff": 0.25,
        "spatial_mixer": "sparse_communication",
        "spatial_reference_experts": 8,
        "talklora_rank": 16,
        "local_branch_mode": "zero",
    }
    allowed = set(expected) | set(legacy_selected)
    if set(normalized) - allowed:
        raise ValueError("Checkpoint contains unsupported architecture fields")
    if any(normalized.get(key, value) != value for key, value in legacy_selected.items()):
        raise ValueError("Legacy checkpoint is not the selected Co-DINO architecture")
    if any(key in expected and normalized[key] != expected[key] for key in normalized):
        raise ValueError("Checkpoint architecture does not match clean Co-DINO")
    return preset


def estimate_parameters(config):
    block_total = 0
    for channels, depth in zip(config.stage_dims, config.depths):
        hidden = config.expert_hidden_dim(channels)
        ffn = channels * config.mlp_ratio
        sectors = config.spectral_radial_bands * config.spectral_num_angles
        spectral = (2 * sectors * channels * channels
                    + 2 * config.spectral_radial_bands - 1
                    + config.spectral_num_angles)
        local = 9 * channels
        expert_count = config.num_shared_experts + config.num_routed_experts
        experts = expert_count * (2 * channels * hidden + hidden + channels)
        router = channels * config.num_routed_experts + config.num_routed_experts
        communication = config.num_routed_experts**2
        norms = 6 * channels
        pointwise_ffn = 2 * channels * ffn + ffn + channels
        block_total += depth * (
            spectral + local + experts + router + communication + norms + pointwise_ffn)
    dims = config.stage_dims
    c0 = dims[0]
    embedding = config.input_channels * config.patch_size**2 * c0 + 3 * c0
    merging = sum(8 * c * c + 8 * c for c in dims[:-1])
    unmerging = sum(8 * c * c + 8 * c for c in dims[:-1])
    skip_projection = sum(2 * c * c + 3 * c for c in dims)
    recovery = (c0 + 1) * config.output_channels * config.patch_size**2
    encoder = embedding + merging + block_total
    decoder = block_total + unmerging + skip_projection + recovery + sum(2*c for c in dims)
    return {"encoder_estimate": encoder, "decoder_estimate": decoder,
            "forecasting_estimate": encoder + decoder}


def count_parameters(model):
    total = trainable = 0
    seen = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        scalar_count = parameter.numel() * (2 if parameter.is_complex() else 1)
        total += scalar_count
        if parameter.requires_grad:
            trainable += scalar_count
    return {"total_real_scalars": total, "trainable_real_scalars": trainable}
