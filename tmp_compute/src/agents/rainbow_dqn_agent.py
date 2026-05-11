import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam

from src.agents.dqn_agent import DqnAgent
from src.agents.networks.rainbow_q_network import RainbowQNetwork
from src.utils.enums import Actions


@dataclass
class OneStepTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


@dataclass
class RainbowTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    discount: float


class PrioritizedRainbowReplayBuffer:
    def __init__(self, capacity: int, alpha: float, epsilon: float):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, transition: RainbowTransition):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        priorities = self.priorities[:len(self.buffer)]

        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probabilities,
        )

        batch = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        weights = torch.tensor(weights, dtype=torch.float32)

        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = priorities.detach().abs().cpu().numpy()

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority + self.epsilon)

    def __len__(self):
        return len(self.buffer)


class RainbowDqnAgent(DqnAgent):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "Rainbow DQN",
        device=None,
    ):
        super().__init__(
            input_size=input_size,
            settings=settings,
            name=name,
            device=device,
        )

        self.model = RainbowQNetwork(
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

        self.n_step = max(1, int(self.settings.n_step))
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.replay_buffer = PrioritizedRainbowReplayBuffer(
            capacity=self.settings.replay_buffer_size,
            alpha=self.settings.per_alpha,
            epsilon=self.settings.per_epsilon,
        )

        self.num_atoms = self.settings.num_atoms
        self.v_min = self.settings.v_min
        self.v_max = self.settings.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.support = torch.linspace(
            self.v_min,
            self.v_max,
            self.num_atoms,
            device=self.device,
        )

    def _beta(self) -> float:
        progress = min(
            self.training_steps / self.settings.total_max_training_steps,
            1.0,
        )

        return (
            self.settings.per_beta_start
            + progress * (self.settings.per_beta_end - self.settings.per_beta_start)
        )

    def predict(self, obs=None) -> Actions:
        if obs is None:
            return Actions.NONE

        self.model.eval()

        with torch.no_grad():
            obs_t = self._prepare_obs(obs)
            q_values = self.model.get_q_values(obs_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return Actions(action_idx)

    def select_action(self, obs: torch.Tensor, step: int) -> Actions:
        """
        Rainbow korzysta z NoisyLinear, więc nie używa epsilon-greedy.
        Eksploracja wynika z szumu w sieci.
        """
        self.model.train()
        self.model.reset_noise()

        with torch.no_grad():
            obs_t = self._prepare_obs(obs)
            q_values = self.model.get_q_values(obs_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return Actions(action_idx)

    def remember(self, state, action, reward, next_state, done):
        transition = OneStepTransition(
            state=state.detach().cpu(),
            action=action.value,
            reward=float(reward),
            next_state=next_state.detach().cpu(),
            done=bool(done),
        )

        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        self._push_n_step_transition()

        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                self._push_n_step_transition()

            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.popleft()

    def _push_n_step_transition(self):
        reward_sum = 0.0
        discount = 1.0

        final_next_state = None
        final_done = False

        for transition in self.n_step_buffer:
            reward_sum += discount * transition.reward

            final_next_state = transition.next_state
            final_done = transition.done

            if final_done:
                break

            discount *= self.settings.gamma

        first = self.n_step_buffer[0]

        transition = RainbowTransition(
            state=first.state,
            action=first.action,
            reward=reward_sum,
            next_state=final_next_state,
            done=final_done,
            discount=discount,
        )

        self.replay_buffer.push(transition)

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.settings.warmup_steps:
            return None

        if len(self.replay_buffer) < self.settings.batch_size:
            return None

        self.model.train()
        self.model.reset_noise()
        self.target_model.reset_noise()

        beta = self._beta()

        batch, indices, weights = self.replay_buffer.sample(
            batch_size=self.settings.batch_size,
            beta=beta,
        )

        weights = weights.to(self.device)

        states = torch.stack([t.state for t in batch]).float().to(self.device)
        next_states = torch.stack([t.next_state for t in batch]).float().to(self.device)

        if states.dim() == 3:
            states = states.squeeze(1)

        if next_states.dim() == 3:
            next_states = next_states.squeeze(1)

        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            [t.done for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        discounts = torch.tensor(
            [t.discount for t in batch],
            dtype=torch.float32,
            device=self.device,
        )

        batch_size = states.size(0)

        distributions = self.model(states)

        action_distributions = distributions[
            torch.arange(batch_size, device=self.device),
            actions,
        ]

        action_distributions = action_distributions.clamp(min=1e-8)

        with torch.no_grad():
            # Double DQN:
            # online model wybiera akcję, target model ocenia rozkład tej akcji.
            next_online_q_values = self.model.get_q_values(next_states)
            next_actions = next_online_q_values.argmax(dim=1)

            next_target_distributions = self.target_model(next_states)

            next_action_distributions = next_target_distributions[
                torch.arange(batch_size, device=self.device),
                next_actions,
            ]

            target_support = (
                rewards.unsqueeze(1)
                + discounts.unsqueeze(1)
                * (1.0 - dones.unsqueeze(1))
                * self.support.unsqueeze(0)
            )

            target_support = target_support.clamp(
                min=self.v_min,
                max=self.v_max,
            )

            b = (target_support - self.v_min) / self.delta_z
            lower = b.floor().long()
            upper = b.ceil().long()

            lower_weight = upper.float() - b
            upper_weight = b - lower.float()

            same = lower == upper
            lower_weight[same] = 1.0
            upper_weight[same] = 0.0

            projected_distribution = torch.zeros(
                batch_size,
                self.num_atoms,
                device=self.device,
            )

            offset = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                * self.num_atoms
            )

            lower_flat = (lower + offset).view(-1)
            upper_flat = (upper + offset).view(-1)

            projected_flat = projected_distribution.view(-1)

            projected_flat.index_add_(
                0,
                lower_flat,
                (next_action_distributions * lower_weight).view(-1),
            )

            projected_flat.index_add_(
                0,
                upper_flat,
                (next_action_distributions * upper_weight).view(-1),
            )

        per_sample_loss = -torch.sum(
            projected_distribution * torch.log(action_distributions),
            dim=1,
        )

        loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            indices=indices,
            priorities=per_sample_loss,
        )

        self.model.reset_noise()
        self.target_model.reset_noise()

        return float(loss.item())

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def clone_for_eval(self) -> "RainbowDqnAgent":
        cloned = RainbowDqnAgent(
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