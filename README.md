
# DEAINN Monte Carlo Code

This repository contains the reproducible simulation code for the DEAINN method. It is a cleaned, self-contained version of the original research scripts.

## Files

- `deainn.py`: neural-network architecture, constrained loss, training, evaluation, and axiom-violation diagnostics.
- `monte_carlo.py`: Monte Carlo repetitions and result aggregation.
- `config.py`: six data-generating processes and default experiment settings.
- `run.py`: command-line entry point.
- `README.md`: usage and repository documentation.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch numpy pandas scikit-learn
```

## Run

The default command runs scenarios A--F, sample sizes 50/100/150, three repetitions, and 1,500 training epochs. Use `--repetitions 100` to reproduce the supplied Monte Carlo design.

```bash
python run.py
```

For a quick smoke test:

```bash
python run.py --scenarios a --sample-sizes 10 --repetitions 1 --epochs 2 --patience 2
```

Useful options include `--device cpu`, `--device cuda:0`, `--activation sigmoid`, and `--output-dir results`.

## Output

Results are written under `results/` by default:

- `summary.csv`: averages over repetitions.
- `runs_<scenario>_<N>.csv`: train/test metrics for every repetition.
- `axioms_<scenario>_<N>.csv`: observation-level monotonicity, concavity, and envelopment diagnostics.
- `checkpoints/`: fitted PyTorch model checkpoints.

No random seed is set by default, so Monte Carlo repetitions and separate executions draw fresh inputs, inefficiency terms, and model initializations. For an exactly repeatable run, pass a seed such as `--seed 42`; deterministic PyTorch settings are enabled only when a seed is supplied.

## Method notes

Training minimizes mean squared error plus penalties for monotonicity, concavity, and the minimal-extrapolation/envelopment condition. The reproduction defaults are penalty weights 150/150/30, hidden layers 64/32, and Sigmoid activation. Observed output contains one-sided technical inefficiency only, `y_observed = y_true - u`, where `u = |N(0, 0.4)|`; there is no symmetric noise term `v`.
