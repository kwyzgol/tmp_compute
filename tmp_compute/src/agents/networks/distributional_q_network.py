import torch
from torch import nn
from torch.nn import functional as F

from src.utils.enums import ActivationFunction
from src.utils.settings import GlobalSettings


class DistributionalQNetwork(nn.Module):
    """
    Sieć Q używana na potrzeby algorytmu Distributional DQN.

    Zamiast przewidywać jedną wartość Q(s, a) dla każdej akcji,
    sieć przewiduje rozkład prawdopodobieństwa po atomach wartości.

    Wyjście sieci ma kształt:
        [batch_size, output_size, num_atoms]

    gdzie:
        - output_size = liczba akcji,
        - num_atoms = liczba atomów rozkładu.

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

        self.num_atoms = settings.num_atoms
        self.v_min = settings.v_min
        self.v_max = settings.v_max

        self.register_buffer(
            "support",
            torch.linspace(self.v_min, self.v_max, self.num_atoms),
        )

        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        layers = []

        in_features = self.input_size

        for _ in range(self.settings.hidden_layers):
            layers.append(nn.Linear(in_features, self.settings.hidden_dims))
            layers.append(self._create_activation())
            in_features = self.settings.hidden_dims

        layers.append(nn.Linear(in_features, self.output_size * self.num_atoms))

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
        Zwraca rozkład prawdopodobieństwa po atomach dla każdej akcji.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Zwracany kształt:
            [batch_size, output_size, num_atoms]
        """
        logits = self.model(x)
        logits = logits.view(-1, self.output_size, self.num_atoms)

        probabilities = F.softmax(logits, dim=2)

        return probabilities

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca oczekiwane wartości Q dla każdej akcji.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Zwracany kształt:
            [batch_size, output_size]
        """
        probabilities = self.forward(x)
        q_values = torch.sum(probabilities * self.support, dim=2)

        return q_values
