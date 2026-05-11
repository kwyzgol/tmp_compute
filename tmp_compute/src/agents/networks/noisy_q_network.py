import math

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.enums import ActivationFunction
from src.utils.settings import GlobalSettings


class NoisyLinear(nn.Module):
    """
    Warstwa liniowa z parametryzowanym szumem używana w Noisy DQN.

    Implementacja korzysta z factorized Gaussian noise.
    Warstwa posiada osobne parametry dla wartości bazowych wag/biasów oraz
    dla skali szumu wag/biasów.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """
    Sieć Q używana na potrzeby algorytmu Noisy DQN.

    Architektura jest podobna do zwykłej sieci MLP, ale zamiast klasycznych
    warstw Linear wykorzystuje warstwy NoisyLinear.

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

        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
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

        layers.append(
            NoisyLinear(
                in_features=in_features,
                out_features=self.output_size,
                sigma_init=self.settings.sigma_init,
            )
        )

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
        Zwraca wartości Q dla wszystkich akcji.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Zwracany kształt:
            [batch_size, output_size]
        """
        return self.model(x)
