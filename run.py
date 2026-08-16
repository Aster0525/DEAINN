"""Command-line entry point for the DEAINN Monte Carlo experiment."""

import argparse
from pathlib import Path

from config import ExperimentConfig, SCENARIOS
from monte_carlo import run_monte_carlo


def parse_args():
    parser = argparse.ArgumentParser(description="Run DEAINN Monte Carlo experiments")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[50, 100, 150])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--monotonicity-weight", type=float, default=150.0)
    parser.add_argument("--concavity-weight", type=float, default=150.0)
    parser.add_argument("--envelopment-weight", type=float, default=30.0)
    parser.add_argument("--inefficiency-std", type=float, default=0.4)
    parser.add_argument("--activation", choices=("tanh", "relu", "leaky_relu", "sigmoid"), default="sigmoid")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed; omitted by default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        scenarios=tuple(args.scenarios), sample_sizes=tuple(args.sample_sizes),
        repetitions=args.repetitions, epochs=args.epochs,
        learning_rate=args.learning_rate, patience=args.patience,
        monotonicity_weight=args.monotonicity_weight,
        concavity_weight=args.concavity_weight,
        envelopment_weight=args.envelopment_weight,
        inefficiency_std=args.inefficiency_std,
        activation=args.activation, seed=args.seed,
        device=args.device, output_dir=args.output_dir,
    )
    summary = run_monte_carlo(config)
    print(f"Finished. Summary saved to {summary}")


if __name__ == "__main__":
    main()
