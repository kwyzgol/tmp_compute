import copy
import random

import torch
from torch.optim import Adam

from src.agents.dqn_agent import DqnAgent
from src.agents.networks.noisy_q_network import NoisyQNetwork
from src.utils.enums import Actions


class NoisyDqnAgent(DqnAgent):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "Noisy DQN",
        device=None,
    ):
        super().__init__(
            input_size=input_size,
            settings=settings,
            name=name,
            device=device,
        )

        self.model = NoisyQNetwork(
            input_size=self.input_size,
            output_size=self.output_size,
            settings=self.settings,
        ).to(self.device)

        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.settings.learning_rate,
        )

    def predict(self, obs=None) -> Actions:
        if obs is None:
            return Actions.NONE

        self.model.eval()

        with torch.no_grad():
            obs_t = self._prepare_obs(obs)
            q_values = self.model(obs_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return Actions(action_idx)

    def select_action(self, obs: torch.Tensor, step: int) -> Actions:
        """
        W Noisy DQN nie używamy epsilon-greedy.
        Eksploracja wynika z zaszumionych warstw NoisyLinear.
        """

        self.model.train()
        self.model.reset_noise()

        with torch.no_grad():
            obs_t = self._prepare_obs(obs)
            q_values = self.model(obs_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return Actions(action_idx)

    def optimize(self) -> float | None:
        self.model.train()
        self.model.reset_noise()

        loss = super().optimize()

        self.model.reset_noise()

        return loss

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def clone_for_eval(self) -> "NoisyDqnAgent":
        cloned = NoisyDqnAgent(
            input_size=self.input_size,
            settings=self.settings,
            name=f"{self.name}_eval_copy",
            device=self.device,
        )

        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        cloned.target_model.load_state_dict(copy.deepcopy(self.target_model.state_dict()))

        cloned.model.eval()
        cloned.target_model.eval()

        return cloned