"""DEAINN model, training, evaluation, and axiom diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def generate_data(
    func: Callable, input_dim: int, num_samples: int,
    inefficiency_std: float = 0.4,
):
    x = (1 + 9 * torch.rand(num_samples, input_dim)).requires_grad_()
    # Targets are fixed observations, not part of the model's autograd graph.
    # Detaching them prevents reuse of the data-generation graph across epochs.
    y_true = func(x).detach()
    # One-sided technical inefficiency only. There is no symmetric noise term v.
    inefficiency = torch.abs(torch.randn_like(y_true) * inefficiency_std)
    y_observed = y_true - inefficiency
    y_observed = y_observed.detach()
    return x, y_true, y_observed


class ProductionFunctionModel(nn.Module):
    def __init__(self, input_dim: int, activation: str = "sigmoid"):
        super().__init__()
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        activation_layer = activations[activation]
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), activation_layer(),
            nn.Linear(64, 32), activation_layer(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def custom_loss(
    output, target, inputs,
    monotonicity_weight=150.0,
    concavity_weight=150.0,
    envelopment_weight=30.0,
):
    mse = nn.functional.mse_loss(output, target)
    gradients = torch.autograd.grad(
        output, inputs, torch.ones_like(output), create_graph=True, retain_graph=True
    )[0]
    hessian_rows = []
    for i in range(inputs.shape[1]):
        row = []
        for j in range(inputs.shape[1]):
            second = torch.autograd.grad(
                gradients[:, i], inputs, torch.ones_like(gradients[:, i]),
                create_graph=True, retain_graph=True,
            )[0][:, j]
            row.append(second)
        hessian_rows.append(torch.stack(row, dim=1))
    hessian = torch.stack(hessian_rows, dim=1)
    # Match the article code exactly: penalize every positive Hessian
    # eigenvalue for every observation, then average over observations.
    # Using only the maximum eigenvalue materially weakens this constraint.
    eigenvalues = torch.linalg.eigvals(hessian).real
    concavity_penalty = torch.relu(eigenvalues).sum(dim=1).mean()
    monotonicity_penalty = torch.relu(-gradients).mean()
    envelopment_penalty = torch.relu(target - output).mean()
    return (
        mse
        + monotonicity_weight * monotonicity_penalty
        + concavity_weight * concavity_penalty
        + envelopment_weight * envelopment_penalty
    )


def train_model(
    model, x, y_observed, epochs, learning_rate, patience, checkpoint: Path,
    monotonicity_weight=150.0,
    concavity_weight=150.0,
    envelopment_weight=30.0,
):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss, stale_epochs = float("inf"), 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = custom_loss(
            model(x), y_observed, x,
            monotonicity_weight, concavity_weight, envelopment_weight,
        )
        loss.backward()
        optimizer.step()
        value = loss.item()
        if value < best_loss:
            best_loss, stale_epochs = value, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1}/{epochs}, loss={value:.6f}")
        if stale_epochs >= patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=x.device, weights_only=True))
    return model


def regression_metrics(y_true, y_pred, elapsed):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = 100 * np.mean(np.divide(2 * np.abs(y_true - y_pred), denominator,
                                    out=np.zeros_like(denominator), where=denominator != 0))
    return {
        "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred),
        "MAPE": 0.0, "SMAPE": smape, "LOG": 0.0,
        "MAE": mean_absolute_error(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true), "Time": elapsed,
    }


def axiom_violations(model, x, y_observed, tau=1e-5):
    x = x.detach().clone().requires_grad_(True)
    y_pred = model(x)
    gradients = torch.autograd.grad(y_pred, x, torch.ones_like(y_pred), create_graph=True)[0]
    max_eigs = []
    for row in range(x.shape[0]):
        hessian = []
        for column in range(x.shape[1]):
            derivative = torch.autograd.grad(
                gradients[row, column], x, retain_graph=True
            )[0][row]
            hessian.append(derivative)
        max_eigs.append(torch.linalg.eigvals(torch.stack(hessian)).real.max())
    max_eigs = torch.stack(max_eigs)
    gap = (y_pred - y_observed).squeeze(-1)
    mon_scalar = gradients.min(dim=1).values
    stats = {
        "MON": 100 * (mon_scalar < -tau).float().mean().item(),
        "CCV": 100 * (max_eigs > tau).float().mean().item(),
        "MEP": 100 * (gap < -tau).float().mean().item(),
    }
    records = {"MON": mon_scalar.detach().cpu(), "CCV": max_eigs.detach().cpu(), "MEP": gap.detach().cpu()}
    return stats, records


def evaluate(model, x, y_true, y_observed, elapsed=0.0):
    start = time.perf_counter()
    prediction = model(x)
    inference_time = time.perf_counter() - start
    metrics = regression_metrics(
        y_true.detach().cpu().numpy(), prediction.detach().cpu().numpy(), elapsed or inference_time
    )
    stats, records = axiom_violations(model, x, y_observed)
    metrics.update({f"{key}_vio": round(value, 4) for key, value in stats.items()})
    return metrics, records
