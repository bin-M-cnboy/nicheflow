from jaxtyping import Float
from torch import Tensor, nn


class CTClassifierNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim)
        ) # 采用极其经典的单隐层多层感知机 (MLP)

    def forward(
        self, x: Float[Tensor, "... {self.input_dim}"]
    ) -> Float[Tensor, "... {self.output_dim}"]:
        return self.net(x)
