"""Self-contained Poseidon HDF5 readers for Co-DINO."""

import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS = ("NS-Gauss", "NS-Sines", "CE-RP", "CE-CRP", "CE-Gauss", "CE-KH")
MEAN = [0.80, 0.0, 0.0, 0.0]
STD = [0.31, 0.391, 0.356, 0.185]
PRESSURE_OFFSETS = {"CE-RP": 0.215, "CE-CRP": 0.553, "CE-Gauss": 2.513, "CE-KH": 1.0}
CHANNEL_CONTRACT = {"channels": ["rho", "u", "v", "p"], "mean": MEAN, "std": STD,
                    "ns_constant_channels": [0, 3], "ns_supervised_channels": [1, 2],
                    "ce_pressure_offsets": PRESSURE_OFFSETS, "time_scale": 20.0}


class PoseidonDataset(Dataset):
    def __init__(self, data_path, name, split="train", num_trajectories=200,
                 task="forecasting", validation_trajectories=120, test_trajectories=240,
                 max_num_time_steps=20):
        if name not in DATASETS or split not in ("train", "val", "test"):
            raise ValueError("Unsupported dataset or split")
        if task != "forecasting":
            raise ValueError("Only forecasting datasets are supported")
        self.path = str(Path(data_path) / (name + ".nc"))
        self.name, self.split = name, split
        self.key = "velocity" if name.startswith("NS-") else "data"
        with h5py.File(self.path, "r") as reader:
            shape = reader[self.key].shape
        if len(shape) not in (4, 5):
            raise ValueError(f"{self.path}: expected N,T,C,H,W or N,T,C,HW; got {shape}")
        required_channels = 2 if name.startswith("NS-") else 4
        if shape[2] < required_channels:
            raise ValueError(f"{name}: expected at least {required_channels} channels; got {shape[2]}")
        total, frames = shape[:2]
        if min(num_trajectories, validation_trajectories, test_trajectories) <= 0:
            raise ValueError("All trajectory counts must be positive")
        if num_trajectories + validation_trajectories + test_trajectories > total:
            raise ValueError(f"{name}: need {num_trajectories} train + {validation_trajectories} "
                             f"val + {test_trajectories} test trajectories; file has {total}")
        self.start, self.count = {"train": (0, num_trajectories),
            "val": (total-validation_trajectories-test_trajectories, validation_trajectories),
            "test": (total-test_trajectories, test_trajectories)}[split]
        self.steps = min(max_num_time_steps, frames-1)
        if self.steps < 1:
            raise ValueError("At least two time frames are required")
        self.samples_per_trajectory = self.steps
        self.mean = torch.tensor(MEAN)[:, None, None]
        self.std = torch.tensor(STD)[:, None, None]
        self._reader, self._pid = None, None
        self.manifest = {"dataset": name, "split": split, "start": self.start,
                         "count": self.count, "file_trajectories": total, "steps": self.steps}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reader"], state["_pid"] = None, None
        return state

    def __len__(self):
        return self.count * self.samples_per_trajectory

    def _state(self, trajectory, frame):
        if self._reader is None or self._pid != os.getpid():
            if self._reader is not None:
                self._reader.close()
            self._reader, self._pid = h5py.File(self.path, "r"), os.getpid()
        channels = 2 if self.name.startswith("NS-") else 4
        state = torch.from_numpy(np.asarray(self._reader[self.key][trajectory, frame, :channels])).float()
        if state.ndim == 2:
            side = math.isqrt(state.shape[-1])
            if side*side != state.shape[-1]:
                raise ValueError("Flattened fields must have a square spatial shape")
            state = state.reshape(channels, side, side)
        if channels == 2:
            state = torch.cat([torch.ones_like(state[:1]), state, torch.zeros_like(state[:1])])
        else:
            state[3] -= PRESSURE_OFFSETS[self.name]
        return (state-self.mean) / self.std

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        trajectory, frame = divmod(index, self.samples_per_trajectory)
        trajectory += self.start
        x = self._state(trajectory, frame)
        y = self._state(trajectory, frame+1)
        mask = torch.tensor([True, False, False, True] if channels_are_ns(self.name)
                            else [False]*4)
        return {"pixel_values": x, "labels": y, "pixel_mask": mask,
                "time": torch.tensor(1.0/20.0)}


def channels_are_ns(name):
    return name.startswith("NS-")
