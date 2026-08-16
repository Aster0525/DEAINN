"""Experiment configuration and data-generating processes."""

from dataclasses import dataclass
from pathlib import Path

import torch


def scenario_a(x):
    return torch.log(x) + 3


def scenario_b(x):
    return 3 + torch.sqrt(x) + torch.log(x)


def scenario_c(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return 0.1 * x1 + 0.1 * x2 + 0.3 * torch.sqrt(x1 * x2)


def scenario_d(x):
    x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.1 * (x1 + x2 + x3) + 0.3 * (x1 * x2 * x3) ** (1 / 3)


def scenario_e(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return 0.1 * x1 + 0.1 * x2 + 0.3 * (x1 * x2) ** (1 / 3)


def scenario_f(x):
    x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.1 * (x1 + x2 + x3) + 0.3 * (x1 * x2 * x3) ** (1 / 4)


SCENARIOS = {
    "a": (scenario_a, 1),
    "b": (scenario_b, 1),
    "c": (scenario_c, 2),
    "d": (scenario_d, 3),
    "e": (scenario_e, 2),
    "f": (scenario_f, 3),
}


@dataclass
class ExperimentConfig:
    scenarios: tuple[str, ...] = tuple(SCENARIOS)
    sample_sizes: tuple[int, ...] = (50, 100, 150)
    repetitions: int = 3
    test_size: int | None = None  # None means: use the training sample size.
    epochs: int = 1500
    learning_rate: float = 0.005
    patience: int = 500
    monotonicity_weight: float = 150.0
    concavity_weight: float = 150.0
    envelopment_weight: float = 30.0
    inefficiency_std: float = 0.4
    activation: str = "sigmoid"
    seed: int | None = None
    output_dir: Path = Path("results")
    device: str = "auto"  # auto, cpu, cuda, cuda:0, ...
