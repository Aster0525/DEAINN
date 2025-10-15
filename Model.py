import torch.nn as nn

# 模型定义模块
class ProductionFunctionModel(nn.Module):
    def __init__(self, input_dim):
        super(ProductionFunctionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
