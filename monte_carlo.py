"""Monte Carlo experiment orchestration and CSV output."""

from __future__ import annotations

import csv
import random
import time

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig, SCENARIOS
from deainn import (ProductionFunctionModel, evaluate, generate_data,
                    resolve_device, train_model)


def run_monte_carlo(config: ExperimentConfig):
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    summary_rows = []

    for scenario_name in config.scenarios:
        func, input_dim = SCENARIOS[scenario_name]
        for sample_size in config.sample_sizes:
            run_rows, diagnostic_rows = [], []
            for repetition in range(config.repetitions):
                print(f"Scenario {scenario_name}, N={sample_size}, repetition {repetition + 1}/{config.repetitions}")
                x, y_true, y_observed = generate_data(
                    func, input_dim, sample_size, config.inefficiency_std
                )
                x, y_observed = x.to(device), y_observed.to(device)
                checkpoint = config.output_dir / "checkpoints" / f"deainn_{scenario_name}_{sample_size}_{repetition}.pth"
                model = ProductionFunctionModel(input_dim, config.activation).to(device)
                start = time.perf_counter()
                train_model(model, x, y_observed, config.epochs, config.learning_rate,
                            config.patience, checkpoint,
                            config.monotonicity_weight, config.concavity_weight,
                            config.envelopment_weight)
                train_time = time.perf_counter() - start
                train_metrics, train_records = evaluate(model, x, y_true.to(device), y_observed, train_time)
                test_n = config.test_size or sample_size
                test_x, test_true, test_observed = generate_data(
                    func, input_dim, test_n, config.inefficiency_std
                )
                test_metrics, test_records = evaluate(
                    model, test_x.to(device), test_true.to(device), test_observed.to(device)
                )
                run_rows.extend([
                    {"Run": repetition + 1, "Phase": "Train", **train_metrics},
                    {"Run": repetition + 1, "Phase": "Test", **test_metrics},
                ])
                for phase, records in (("Train", train_records), ("Test", test_records)):
                    for index in range(len(records["MON"])):
                        diagnostic_rows.append({
                            "scenario": scenario_name, "N": sample_size, "rep": repetition + 1,
                            "set": phase.lower(), "MON": float(records["MON"][index]),
                            "CCV": float(records["CCV"][index]), "ENV": float(records["MEP"][index]),
                        })

            pd.DataFrame(run_rows).to_csv(
                config.output_dir / f"runs_{scenario_name}_{sample_size}.csv", index=False
            )
            pd.DataFrame(diagnostic_rows).to_csv(
                config.output_dir / f"axioms_{scenario_name}_{sample_size}.csv", index=False
            )
            frame = pd.DataFrame(run_rows)
            row = {"Scenario": scenario_name, "Number of Obs.": sample_size}
            for phase in ("Train", "Test"):
                averages = frame[frame["Phase"] == phase].drop(columns=["Run", "Phase"]).mean()
                row.update({f"{key}_{phase.lower()}": value for key, value in averages.items()})
            summary_rows.append(row)

    summary_path = config.output_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return summary_path
