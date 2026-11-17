"""Single-GPU Co-DINO pretraining and joint single-PDE finetuning."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
import yaml

from model.architecture_config import architecture_from_dict, count_parameters, get_model_config
from model.blocks.sparse_moe import communication_moe_modules
from model.checkpoint import atomic_save, load_checkpoint, load_forecasting_pretrain
from model.data import CHANNEL_CONTRACT, DATASETS, PoseidonDataset
from model.model import CoDINOForForecasting
from model.objectives import forecasting_relative_l1


def parser(stage):
    if stage not in ("forecast_pretrain", "finetune"):
        raise ValueError("Only full-model pretraining and joint finetuning are supported")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", "--data-path", required=True)
    p.add_argument("--output_dir", "--output-dir", required=True)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--amp", choices=("auto", "none", "fp16", "bf16"), default="auto")
    p.add_argument("--resume", help="Resume this exact run from last.pt")
    p.add_argument("--model_size", "--model-size", choices=("T", "S", "B", "L"))
    p.add_argument("--lr", type=float)
    for name in ("num_epochs", "seed", "num_trajectories", "batch_size", "num_workers",
                 "gradient_accumulation_steps", "validation_trajectories", "test_trajectories"):
        p.add_argument("--" + name, type=int)
    if stage == "forecast_pretrain":
        p.add_argument("--pretrain_datasets", "--pretrain-datasets", nargs="+")
    else:
        p.add_argument("--full_checkpoint", "--full-checkpoint",
                       help="Co-DINO full-model pretraining best.pt")
        p.add_argument("--dataset", choices=DATASETS, required=True)
    return p


def read_config(args, stage):
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    allowed = {
        "model_size", "dataset", "num_trajectories", "num_epochs", "seed",
        "num_input_frames", "just_velocities", "batch_size",
        "gradient_accumulation_steps", "num_workers", "lr", "weight_decay",
        "lr_scheduler", "warmup_ratio", "max_grad_norm", "train_time_step_size",
        "train_small_time_transition", "loss", "loss_reduction",
        "validation_trajectories", "test_trajectories", "max_num_time_steps",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown clean Co-DINO config fields: {sorted(unknown)}")
    if args.seed is None:
        raise ValueError("--seed must be passed explicitly")
    for name in allowed:
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = value
    if stage == "finetune":
        cfg["dataset"] = [args.dataset]
    elif args.pretrain_datasets is not None:
        cfg["dataset"] = [name.strip() for item in args.pretrain_datasets
                          for name in item.replace("、", ",").replace("，", ",").split(",")]
    if cfg.get("loss") != "relative_l1" or cfg.get("loss_reduction") != "sample_mean":
        raise ValueError("Co-DINO training uses per-sample joint-channel Relative-L1")
    if cfg.get("train_time_step_size") != 1 or not cfg.get("train_small_time_transition"):
        raise ValueError("Co-DINO training uses adjacent dt=1 pairs")
    if cfg.get("num_input_frames") != 1 or cfg.get("just_velocities"):
        raise ValueError("Expected one four-channel input state")
    if cfg.get("lr_scheduler") != "cosine":
        raise ValueError("Only cosine scheduling is supported")
    for key, default in {"validation_trajectories": 120, "test_trajectories": 240,
                         "max_num_time_steps": 20,
                         "gradient_accumulation_steps": 1}.items():
        cfg.setdefault(key, default)
    required_positive = ("batch_size", "num_epochs", "num_trajectories",
                         "gradient_accumulation_steps", "validation_trajectories",
                         "test_trajectories", "max_num_time_steps")
    for key in required_positive:
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.get("num_workers")) is not int or cfg["num_workers"] < 0 or cfg["seed"] < 0:
        raise ValueError("Workers and seed must be non-negative integers")
    if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
        raise ValueError("Learning rate must be positive")
    if cfg["weight_decay"] < 0 or not 0 <= cfg["warmup_ratio"] < 1:
        raise ValueError("Invalid weight decay or warmup ratio")
    if (not isinstance(cfg.get("dataset"), list) or not cfg["dataset"]
            or not set(cfg["dataset"]) <= set(DATASETS)):
        raise ValueError("dataset must contain supported Poseidon datasets")
    if len(set(cfg["dataset"])) != len(cfg["dataset"]):
        raise ValueError("Duplicate datasets are not allowed")
    if stage == "finetune" and not args.full_checkpoint and not args.resume:
        raise ValueError("Joint finetuning requires --full-checkpoint or --resume")
    return cfg


def prepare_device(args):
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected exactly one visible CUDA GPU")
    amp = args.amp
    if amp == "auto":
        amp = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device == "cuda" else "none"
    if device == "cpu" and amp != "none":
        raise ValueError("CPU runs require --amp none")
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The allocated GPU does not support bfloat16")
    return torch.device(device), amp


def make_loaders(cfg, args, device):
    parts, generators, loaders = {}, {}, {}
    for split in ("train", "val"):
        parts[split] = [PoseidonDataset(
            args.data_path, name, split=split,
            num_trajectories=cfg["num_trajectories"],
            validation_trajectories=cfg["validation_trajectories"],
            test_trajectories=cfg["test_trajectories"],
            max_num_time_steps=cfg["max_num_time_steps"]) for name in cfg["dataset"]]
        generators[split] = torch.Generator().manual_seed(cfg["seed"] + (split == "val"))
        loaders[split] = DataLoader(
            ConcatDataset(parts[split]), batch_size=cfg["batch_size"],
            shuffle=split == "train", num_workers=cfg["num_workers"],
            pin_memory=device.type == "cuda", persistent_workers=cfg["num_workers"] > 0,
            generator=generators[split])
        if not len(loaders[split]):
            raise ValueError(f"No {split} batches")
    manifests = {split: [part.manifest for part in items] for split, items in parts.items()}
    return loaders, generators, manifests


def identity_differences(saved, current):
    differences = {}
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            differences[key] = {"checkpoint": saved.get(key), "current": current.get(key)}
    return differences


def accumulate_routing_statistics(model, accumulators):
    for name, module in communication_moe_modules(model).items():
        state = module.last_routing_stats
        if state is None:
            continue
        current = accumulators.setdefault(name, {
            "expert_counts": torch.zeros(module.num_routed_experts, dtype=torch.float64),
            "probability_sum": torch.zeros(module.num_routed_experts, dtype=torch.float64),
            "selected_weight_sum": torch.zeros(module.num_routed_experts, dtype=torch.float64),
            "tokens": 0,
        })
        for key in ("expert_counts", "probability_sum", "selected_weight_sum"):
            current[key] += state[key].double().cpu()
        current["tokens"] += int(state["tokens"])


def routing_record(epoch, split, accumulators):
    layers = {}
    for name, values in accumulators.items():
        selections = values["expert_counts"].sum().clamp_min(1)
        weight_sum = values["selected_weight_sum"].sum().clamp_min(1e-12)
        layers[name] = {
            "tokens": values["tokens"],
            "expert_counts": values["expert_counts"].tolist(),
            "activation_fraction": (values["expert_counts"] / selections).tolist(),
            "mean_router_probability": (
                values["probability_sum"] / max(1, values["tokens"])).tolist(),
            "selected_weight_fraction": (values["selected_weight_sum"] / weight_sum).tolist(),
        }
    return {"epoch": epoch, "split": split, "layers": layers}


def main(stage):
    args = parser(stage).parse_args()
    saved = load_checkpoint(args.resume) if args.resume else None
    if saved is not None:
        saved_identity = saved.get("identity")
        if (not isinstance(saved_identity, dict) or saved_identity.get("stage") != stage
                or saved_identity.get("task") != "forecasting"):
            raise ValueError("--resume requires a matching Co-DINO checkpoint")
        if stage == "finetune":
            args.full_checkpoint = saved_identity["full_checkpoint"]
    requested_cfg = read_config(args, stage)
    if saved is not None:
        architecture = architecture_from_dict(saved_identity["architecture"])
        cfg = dict(saved_identity["config"])
        changes = identity_differences(cfg, requested_cfg)
        if changes:
            print(json.dumps({"event": "resume_uses_checkpoint_config",
                              "ignored_requested_differences": changes}), flush=True)
    else:
        cfg = requested_cfg
        architecture = get_model_config(cfg["model_size"])

    device, amp = prepare_device(args)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    loaders, generators, manifests = make_loaders(cfg, args, device)
    model = CoDINOForForecasting(architecture)
    source_metadata = None
    if stage == "finetune" and saved is None:
        source_metadata = load_forecasting_pretrain(
            model, architecture, args.full_checkpoint)["identity"]
    elif stage == "finetune":
        source_metadata = saved.get("source_metadata")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    accumulation = cfg["gradient_accumulation_steps"]
    total_steps = cfg["num_epochs"] * math.ceil(len(loaders["train"]) / accumulation)
    warmup = int(total_steps * cfg["warmup_ratio"])

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.cuda.amp.GradScaler(enabled=amp == "fp16")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not args.resume:
        raise ValueError("Output already contains last.pt; use --resume or a new directory")
    identity = {
        "stage": stage, "task": "forecasting",
        "policy": "joint" if stage == "finetune" else None,
        "data_path": str(Path(args.data_path).resolve()),
        "architecture": architecture.to_dict(), "config": cfg, "manifests": manifests,
    }
    if stage == "finetune":
        identity["initialization"] = "full"
        identity["full_checkpoint"] = str(Path(args.full_checkpoint).resolve())
    completed, best = 0, float("inf")
    if saved is not None:
        if saved_identity != identity:
            raise ValueError("Resume identity mismatch: " + json.dumps(
                identity_differences(saved_identity, identity), ensure_ascii=False))
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        completed, best = saved["completed_epochs"], saved["best_validation"]
        random.setstate(saved["rng"]["python"]); np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and saved["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        for split, generator in generators.items():
            generator.set_state(saved["rng"]["loaders"][split])
    metadata = {**identity, "channel_contract": CHANNEL_CONTRACT,
                "source_full_model_metadata": source_metadata,
                "torch_version": torch.__version__, "device": str(device), "amp": amp,
                "parameters": count_parameters(model)}
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"event": "start", "model": "Co-DINO", "stage": stage,
                      "device": str(device), "amp": amp, "datasets": cfg["dataset"],
                      "parameters": count_parameters(model), "splits": manifests}), flush=True)

    for epoch in range(completed, cfg["num_epochs"]):
        started = time.monotonic()
        metrics = {}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = samples = 0
            routing = {}
            with torch.set_grad_enabled(training):
                for batch_index, batch in enumerate(loader):
                    batch = {name: value.to(device, non_blocking=True)
                             for name, value in batch.items()}
                    with torch.cuda.amp.autocast(
                            enabled=amp != "none",
                            dtype=torch.bfloat16 if amp == "bf16" else torch.float16):
                        prediction = model(batch["pixel_values"], batch["time"])
                        loss = forecasting_relative_l1(
                            prediction.float(), batch["labels"], batch["pixel_mask"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {split} loss at epoch {epoch + 1}")
                    accumulate_routing_statistics(model, routing)
                    batch_samples = len(batch["labels"])
                    loss_sum += loss.item() * batch_samples
                    samples += batch_samples
                    if training:
                        window_start = (batch_index // accumulation) * accumulation
                        window_length = min(accumulation, len(loader) - window_start)
                        scaler.scale(loss / window_length).backward()
                        if ((batch_index + 1) % accumulation == 0
                                or batch_index + 1 == len(loader)):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                            scaler.step(optimizer); scaler.update(); scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
            metrics[split] = loss_sum / samples
            with (output / "expert_activation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(routing_record(epoch + 1, split, routing)) + "\n")
        improved = metrics["val"] < best
        best = min(best, metrics["val"])
        record = {"epoch": epoch + 1, **metrics, "lr": scheduler.get_last_lr(),
                  "seconds": time.monotonic() - started,
                  "parameters": count_parameters(model)}
        print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        payload = {
            "identity": identity, "channel_contract": CHANNEL_CONTRACT,
            "source_metadata": source_metadata, "completed_epochs": epoch + 1,
            "best_validation": best, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "loaders": {split: generator.get_state()
                                for split, generator in generators.items()}},
        }
        atomic_save(payload, output / "last.pt")
        if improved:
            atomic_save(payload, output / "best.pt")
    atomic_save({"architecture": architecture.to_dict(), "channel_contract": CHANNEL_CONTRACT,
                 "model": model.state_dict(), "metadata": metadata}, output / "model_final.pt")
    print(json.dumps({"event": "complete", "epochs": cfg["num_epochs"],
                      "output": str(output)}), flush=True)
