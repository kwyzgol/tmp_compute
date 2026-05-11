import copy

import torch
from torch.distributions import Categorical
from torch.optim import Adam

from src.agents.agents import AgentNN
from src.agents.networks.actor_critic_network import ActorCriticNetwork
from src.utils.enums import Actions


class A2CAgent(AgentNN):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "A2C",
        device=None,
    ):
        super().__init__(name)

        self.input_size = input_size
        self.output_size = len(Actions)
        self.settings = settings

        self.device = self.resolve_device(self.settings, device)

        self.model = ActorCriticNetwork(
            input_size=self.input_size,
            output_size=self.output_size,
            settings=self.settings,
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.settings.learning_rate,
        )

        self.states = []
        self.actions_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
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
            logits = self.model.get_policy_logits(obs_t)
            action_idx = int(torch.argmax(logits, dim=1).item())

        return Actions(action_idx)

    def select_action(self, obs) -> Actions:
        self.model.train()

        obs_t = self._prepare_obs(obs)

        logits, value = self.model(obs_t)

        distribution = Categorical(logits=logits)

        action_tensor = distribution.sample()
        log_prob = distribution.log_prob(action_tensor)
        entropy = distribution.entropy()

        self.actions_log_probs.append(log_prob.squeeze())
        self.values.append(value.squeeze())
        self.entropies.append(entropy.squeeze())

        action_idx = int(action_tensor.item())

        return Actions(action_idx)

    def remember_reward(self, reward: float, done: bool):
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def _calculate_returns(self, next_value: torch.Tensor | None = None) -> torch.Tensor:
        returns = []

        if next_value is None:
            discounted_return = 0.0
        else:
            discounted_return = float(next_value.detach().cpu().item())

        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_return = 0.0

            discounted_return = reward + self.settings.gamma * discounted_return
            returns.append(discounted_return)

        returns.reverse()

        return torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

    def optimize_rollout(self, next_obs=None, done: bool = True) -> float | None:
        if len(self.rewards) == 0:
            return None

        next_value = None

        if not done and next_obs is not None:
            with torch.no_grad():
                next_obs_t = self._prepare_obs(next_obs)
                next_value = self.model.get_value(next_obs_t).squeeze()

        returns = self._calculate_returns(next_value=next_value)

        log_probs = torch.stack(self.actions_log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = (
            actor_loss
            + self.settings.value_loss_coef * critic_loss
            - self.settings.entropy_coef * entropy_bonus
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.clear_rollout_memory()

        return float(loss.item())

    def clear_rollout_memory(self):
        self.states.clear()
        self.actions_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.entropies.clear()

    def clone_for_eval(self) -> "A2CAgent":
        cloned = A2CAgent(
            input_size=self.input_size,
            settings=self.settings,
            name=f"{self.name}_eval_copy",
            device=self.device,
        )

        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        cloned.model.eval()

        return cloned
