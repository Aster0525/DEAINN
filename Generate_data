import torch

def generate_data(func, input_dim, num_samples):
    """
    Generate synthetic data for a production function.

    :param func: The true production function to generate ground truth outputs.
    :param input_dim: The number of input dimensions (i.e., input variables).
    :param num_samples: The number of samples to generate.
    :return: Input tensor x, ground truth output y_true, and noise-added target y_tn.
    """
    # torch.manual_seed(42)  # Optional: Set random seed for reproducibility
    x = (1 + 9 * torch.rand(num_samples, input_dim)).requires_grad_()  # Input variables x1, x2, ..., scaled to [1, 10]
    y_true = func(x)  # Ground truth output without noise
    # y_true = 0.1 * x[:, 0:1] + 0.1 * x[:, 1:2] + 0.3 * torch.sqrt(x[:, 0:1]) * torch.sqrt(x[:, 1:2])  # Alternative explicit production function
    u = torch.abs(torch.randn_like(y_true) * 0.4)  # Additive noise (non-negative inefficiency term)
    y_tn = y_true - u  # Noisy target output
    return x, y_true, y_tn

