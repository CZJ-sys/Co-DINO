"""Atomic local checkpoints and strict full-model transfer."""

import os
from pathlib import Path
import shutil
import tempfile

import torch

from .data import CHANNEL_CONTRACT
from .architecture_config import architecture_from_dict


def atomic_save(payload, path):
    path = Path(path)
    temporary = None
    try:
        # Do not silently recreate a directory deleted during training.
        # A same-directory temporary file keeps replacement on one filesystem.
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name+".", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as error:
        cleanup_error = None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup:
                cleanup_error = str(cleanup)
        probe = path.parent.absolute()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free_bytes = shutil.disk_usage(probe).free
        except OSError:
            free_bytes = "unavailable"
        free_inodes = "unavailable"
        if hasattr(os, "statvfs"):
            try:
                free_inodes = os.statvfs(probe).f_favail
            except OSError:
                pass
        chain, current, seen = [], error, set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        raise RuntimeError(
            f"Checkpoint save failed: target={path.absolute()}; "
            f"parent_exists={path.parent.is_dir()}; filesystem_probe={probe}; "
            f"filesystem_free_bytes={free_bytes}; free_inodes={free_inodes}; "
            f"temporary_cleanup_error={cleanup_error}. Check disk space, user quota, "
            f"directory deletion, permissions and filesystem I/O. Quota is not measured "
            f"by filesystem_free_bytes. Original errors: {' | '.join(chain)}"
        ) from error


def load_checkpoint(path):
    # Only use trusted checkpoints from this training workflow.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_forecasting_pretrain(model, config, path):
    """Load both branches from a validated full-model pretraining checkpoint."""
    payload = load_checkpoint(path)
    identity = payload.get("identity")
    if (not isinstance(identity, dict) or identity.get("stage") != "forecast_pretrain"
            or identity.get("task") != "forecasting"):
        raise ValueError("Expected a full Encoder-Decoder pretraining checkpoint")
    if (architecture_from_dict(identity.get("architecture")) != config
            or payload.get("channel_contract") != CHANNEL_CONTRACT):
        raise ValueError("Full checkpoint architecture or channel/normalization contract does not match")
    model.load_state_dict(payload["model"], strict=True)
    return payload
