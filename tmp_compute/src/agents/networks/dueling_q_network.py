import torch
from torch import nn

from src.utils.enums import ActivationFunction
from src.utils.settings import GlobalSettings


class DuelingQNetwork(nn.Module):
    """
    Sieć Q używana na potrzeby algorytmu Dueling DQN.

    Architektura rozdziela reprezentację stanu na dwa strumienie:
        - value stream: V(s)
        - advantage stream: A(s, a)

    Wynik końcowy to wartości Q dla wszystkich akcji:
        Q(s, a) = V(s) + A(s, a) - mean(A(s, ~))

    Klasa odpowiada tylko za architekturę sieci neuronowej.
    Nie wybiera akcji, nie wykonuje argmax i nie zna logiki eksploracji.
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

        if self.settings.hidden_layers < 1:
            raise ValueError("DuelingQNetwork wymaga hidden_layers >= 1.")

        self.feature_layer = self._build_feature_layer()
        self.value_stream = self._build_value_stream()
        self.advantage_stream = self._build_advantage_stream()

    def _build_feature_layer(self) -> nn.Sequential:
        layers = []

        in_features = self.input_size

        for _ in range(self.settings.hidden_layers):
            layers.append(nn.Linear(in_features, self.settings.hidden_dims))
            layers.append(self._create_activation())
            in_features = self.settings.hidden_dims

        return nn.Sequential(*layers)

    def _build_value_stream(self) -> nn.Linear:
        return nn.Linear(self.settings.hidden_dims, 1)

    def _build_advantage_stream(self) -> nn.Linear:
        return nn.Linear(self.settings.hidden_dims, self.output_size)

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
        Zwraca wartości Q dla wszystkich akcji.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Zwracany kształt:
            [batch_size, output_size]
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)

        return q_values
