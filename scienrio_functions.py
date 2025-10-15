import torch
import numpy

def scenario_a(x):
    return torch.log(x) + 3

# Scenario (B)
def scenario_b(x):
    return 3 + x**(1/2) + torch.log(x)

def scenario_c(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return 0.1 * x1+ 0.1 * x2+ 0.3 * (x1 * x2)**(1/2)
#
# Scenario (D)
def scenario_d(x):
    x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.1 * x1 + 0.1 * x2 + 0.1 * x3 + 0.3 * (x1 * x2 * x3)**(1/3)

# Scenario (E)
def scenario_e(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return 0.1 * x1 + 0.1 * x2 + 0.3 * (x1 * x2)**(1/3)

# Scenario (F)
def scenario_f(x):
    x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y_true = 0.1 * x1 + 0.1 * x2 + 0.1 * x3 + 0.3 * (x1 * x2 * x3) ** (1 / 4)
    # print("x1 type:", type(y_true), "x1 shape:", y_true.shape)
    return y_true