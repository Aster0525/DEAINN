
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

## Cite
@article{CHEN2026,
title = {A data envelopment analysis-informed neural network for production frontier estimation with soft regularization of production axioms},
journal = {Computers \& Operations Research},
pages = {107636},
year = {2026},
issn = {0305-0548},
doi = {10.1016/j.cor.2026.107636},
author = {Xingyue Chen and Min Yang and Zixuan Wang and Liang Liang}}


