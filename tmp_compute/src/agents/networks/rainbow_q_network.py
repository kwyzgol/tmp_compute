import torch
from torch import nn
from torch.nn import functional as F

from src.agents.networks.noisy_q_network import NoisyLinear
from src.utils.enums import ActivationFunction
from src.utils.settings import GlobalSettings


class RainbowQNetwork(nn.Module):
    """
    Sieć Q używana na potrzeby algorytmu Rainbow DQN.

    Architektura łączy elementy:
        - Dueling DQN,
        - Noisy DQN,
        - Distributional DQN.

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

        self.feature_layer = self._build_feature_layer()
        self.value_stream = self._build_value_stream()
        self.advantage_stream = self._build_advantage_stream()

    def _build_feature_layer(self) -> nn.Sequential:
        layers = []

        in_features = self.input_size

        for _ in range(self.settings.hidden_layers):
            layers.append(
                NoisyLinear(
                    in_features=in_features,
                    out_features=self.settings.hidden_dims,
                    sigma_init=self.settings.sigma_init,
                )
            )
            layers.append(self._create_activation())
            in_features = self.settings.hidden_dims

        return nn.Sequential(*layers)

    def _build_value_stream(self) -> NoisyLinear:
        return NoisyLinear(
            in_features=self.settings.hidden_dims,
            out_features=self.num_atoms,
            sigma_init=self.settings.sigma_init,
        )

    def _build_advantage_stream(self) -> NoisyLinear:
        return NoisyLinear(
            in_features=self.settings.hidden_dims,
            out_features=self.output_size * self.num_atoms,
            sigma_init=self.settings.sigma_init,
        )

    def _create_activation(self) -> nn.Module:
        activation = self.settings.f_activation

        if activation == ActivationFunction.RELU:
            return nn.ReLU()

        if activation == ActivationFunction.TANH:
            return nn.Tanh()

        if activation == ActivationFunction.SILU:
            return nn.SiLU()

        raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

    def reset_noise(self):
        """
        Losuje nowy szum we wszystkich warstwach NoisyLinear.

        Metoda powinna być wywoływana przez agenta/trening w odpowiednich momentach,
        np. przed wyborem akcji lub przed krokiem optymalizacji.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca rozkład prawdopodobieństwa po atomach dla każdej akcji.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Zwracany kształt:
            [batch_size, output_size, num_atoms]
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        value = value.view(-1, 1, self.num_atoms)
        advantages = advantages.view(-1, self.output_size, self.num_atoms)

        logits = value + advantages - advantages.mean(dim=1, keepdim=True)
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
