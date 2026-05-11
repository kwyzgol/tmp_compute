import copy

import torch
from torch.distributions import Categorical
from torch.optim import Adam

from src.agents.agents import AgentNN
from src.agents.networks.custom_network import CustomNetwork
from src.utils.enums import Actions


class ReinforceAgent(AgentNN):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "REINFORCE",
        device=None,
    ):
        super().__init__(name)

        self.input_size = input_size
        self.output_size = len(Actions)
        self.settings = settings

        self.device = self.resolve_device(self.settings, device)

        self.model = CustomNetwork(
            input_size=self.input_size,
            output_size=self.output_size,
            settings=self.settings,
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.settings.learning_rate,
        )

        self.log_probs = []
        self.rewards = []
        self.entropies = []

        self.training_steps = 0

    def _prepare_obs(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.float()

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return obs.to(self.device)

    def predict(self, obs=None) -> Actions:
        if obs is None:
            return Actions.NONE

        self.model.eval()

        with torch.no_grad():
            obs_t = self._prepare_obs(obs)
            logits = self.model(obs_t)
            action_idx = int(torch.argmax(logits, dim=1).item())

        return Actions(action_idx)

    def select_action(self, obs) -> Actions:
        self.model.train()

        obs_t = self._prepare_obs(obs)
        logits = self.model(obs_t)

        distribution = Categorical(logits=logits)

        action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        entropy = distribution.entropy()

        self.log_probs.append(log_prob.squeeze())
        self.entropies.append(entropy.squeeze())

        action_idx = int(action_tensor.item())

        return Actions(action_idx)

    def remember_reward(self, reward: float):
        self.rewards.append(float(reward))

    def _calculate_returns(self) -> torch.Tensor:
        returns = []
        discounted_return = 0.0

        for reward in reversed(self.rewards):
            discounted_return = reward + self.settings.gamma * discounted_return
            returns.append(discounted_return)

        returns.reverse()

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def optimize_episode(self) -> float | None:
        if len(self.rewards) == 0:
            return None

        returns = self._calculate_returns()

        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        policy_loss = -(log_probs * returns).mean()
        entropy_bonus = entropies.mean()

        loss = policy_loss - self.settings.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.clear_episode_memory()

        return float(loss.item())

    def clear_episode_memory(self):
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def clone_for_eval(self) -> "ReinforceAgent":
        cloned = ReinforceAgent(
            input_size=self.input_size,
            settings=self.settings,
            name=f"{self.name}_eval_copy",
            device=self.device,
        )

        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        cloned.model.eval()

        return cloned
