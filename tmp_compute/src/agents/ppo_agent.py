import copy

import torch
from torch.distributions import Categorical
from torch.optim import Adam

from src.agents.agents import AgentNN
from src.agents.networks.actor_critic_network import ActorCriticNetwork
from src.utils.enums import Actions


class PPOAgent(AgentNN):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "PPO",
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
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

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

        self.states.append(obs_t.squeeze(0).detach())
        self.actions.append(action_tensor.squeeze().detach())
        self.old_log_probs.append(log_prob.squeeze().detach())
        self.values.append(value.squeeze().detach())

        return Actions(int(action_tensor.item()))

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

        states = torch.stack(self.states).to(self.device)
        actions = torch.stack(self.actions).long().to(self.device)
        old_log_probs = torch.stack(self.old_log_probs).to(self.device)
        old_values = torch.stack(self.values).to(self.device)

        returns = self._calculate_returns(next_value=next_value)

        advantages = returns - old_values

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout_size = states.size(0)
        batch_size = min(self.settings.ppo_batch_size, rollout_size)

        last_loss = None

        for _ in range(self.settings.ppo_epochs):
            permutation = torch.randperm(rollout_size, device=self.device)

            for start in range(0, rollout_size, batch_size):
                indices = permutation[start:start + batch_size]

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                logits, values = self.model(batch_states)
                values = values.squeeze(-1)

                distribution = Categorical(logits=logits)

                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                unclipped = ratio * batch_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.settings.clip_epsilon,
                    1.0 + self.settings.clip_epsilon,
                ) * batch_advantages

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = (batch_returns - values).pow(2).mean()

                loss = (
                    actor_loss
                    + self.settings.value_loss_coef * critic_loss
                    - self.settings.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                last_loss = float(loss.item())

        self.clear_rollout_memory()

        return last_loss

    def clear_rollout_memory(self):
        self.states.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def clone_for_eval(self) -> "PPOAgent":
        cloned = PPOAgent(
            input_size=self.input_size,
            settings=self.settings,
            name=f"{self.name}_eval_copy",
            device=self.device,
        )

        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        cloned.model.eval()

        return cloned
