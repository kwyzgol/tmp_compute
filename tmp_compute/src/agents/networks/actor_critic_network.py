import torch
from torch import nn

from src.utils.enums import ActivationFunction
from src.utils.settings import GlobalSettings


class ActorCriticNetwork(nn.Module):
    """
    Sieć Actor-Critic używana na potrzeby algorytmów A2C i PPO.

    Architektura posiada wspólną część MLP oraz dwie osobne głowy:
        - actor head: logits polityki dla akcji,
        - critic head: wartość stanu V(s).

    Klasa odpowiada tylko za architekturę sieci neuronowej.
    Nie wybiera akcji, nie wykonuje softmax, nie losuje akcji
    i nie zna logiki A2C / PPO.
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

        self.feature_layer = self._build_feature_layer()
        self.actor_head = self._build_actor_head()
        self.critic_head = self._build_critic_head()

    def _build_feature_layer(self) -> nn.Sequential:
        layers = []

        in_features = self.input_size

        for _ in range(self.settings.hidden_layers):
            layers.append(nn.Linear(in_features, self.settings.hidden_dims))
            layers.append(self._create_activation())
            in_features = self.settings.hidden_dims

        return nn.Sequential(*layers)

    def _build_actor_head(self) -> nn.Linear:
        return nn.Linear(self.settings.hidden_dims, self.output_size)

    def _build_critic_head(self) -> nn.Linear:
        return nn.Linear(self.settings.hidden_dims, 1)

    def _create_activation(self) -> nn.Module:
        activation = self.settings.f_activation

        if activation == ActivationFunction.RELU:
            return nn.ReLU()

        if activation == ActivationFunction.TANH:
            return nn.Tanh()

        if activation == ActivationFunction.SILU:
            return nn.SiLU()

        raise ValueError(f"Nieznana funkcja aktywacji: {activation}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca logits polityki oraz wartość stanu.

        Oczekiwany kształt wejścia:
            [batch_size, input_size]

        Zwracane kształty:
            logits: [batch_size, output_size]
            value:  [batch_size, 1]
        """
        features = self.feature_layer(x)

        logits = self.actor_head(features)
        value = self.critic_head(features)

        return logits, value

    def get_policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca wyłącznie logits polityki.

        Przydatne, gdy agent potrzebuje tylko części actor,
        np. podczas wyboru akcji.
        """
        features = self.feature_layer(x)
        return self.actor_head(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca wyłącznie wartość stanu V(s).

        Przydatne, gdy algorytm potrzebuje tylko predykcji critic,
        np. przy obliczaniu przewag lub bootstrapu.
        """
        features = self.feature_layer(x)
        return self.critic_head(features)
