import torch
from torch import nn

from src.utils.enums import ActivationFunction
from src.utils.settings import GlobalSettings


class CustomNetwork(nn.Module):
    """
    Uniwersalna sieć MLP używana jako podstawowy model PyTorch.

    Klasa odpowiada tylko za architekturę sieci neuronowej:
        input tensor -> MLP -> output tensor

    Nie wybiera akcji, nie wykonuje argmax, nie wykonuje softmax,
    nie zna logiki DQN / REINFORCE / PPO.

    Sieć będzie używana m.in. przez:
        - DQN
        - N-step DQN
        - Double DQN
        - PER DQN
        - REINFORCE
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        settings: GlobalSettings,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.settings = settings

        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        layers = []

        in_features = self.input_size

        for _ in range(self.settings.hidden_layers):
            layers.append(nn.Linear(in_features, self.settings.hidden_dims))
            layers.append(self._create_activation())
            in_features = self.settings.hidden_dims

        layers.append(nn.Linear(in_features, self.output_size))

        return nn.Sequential(*layers)

    def _create_activation(self) -> nn.Module:
        activation = self.settings.f_activation

        if activation == ActivationFunction.RELU:
            return nn.ReLU()

        if activation == ActivationFunction.TANH:
            return nn.Tanh()

        if activation == ActivationFunction.SILU:
            return nn.SiLU()

        raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wykonuje przejście przez sieć.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Dla pojedynczej obserwacji agent powinien wcześniej dodać batch dimension,
        np. obs.unsqueeze(0).
        """
        return self.model(x)
