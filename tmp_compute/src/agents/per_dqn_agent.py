import copy
import random
from collections import deque

import numpy as np
import torch

from src.agents.dqn_agent import DqnAgent, Transition


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, epsilon: float):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0

        transition = Transition(
            state=state.detach().cpu(),
            action=int(action),
            reward=float(reward),
            next_state=next_state.detach().cpu(),
            done=bool(done),
        )

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

    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.detach().abs().cpu().numpy()

        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = float(error + self.epsilon)

    def __len__(self):
        return len(self.buffer)


class PerDqnAgent(DqnAgent):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "PER DQN",
        device=None,
    ):
        super().__init__(
            input_size=input_size,
            settings=settings,
            name=name,
            device=device,
        )

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.settings.replay_buffer_size,
            alpha=self.settings.per_alpha,
            epsilon=self.settings.per_epsilon,
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

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.settings.warmup_steps:
            return None

        if len(self.replay_buffer) < self.settings.batch_size:
            return None

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

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_max = next_q_values.max(dim=1).values

            target = rewards + self.settings.gamma * (1.0 - dones) * next_q_max

        td_errors = target - q_selected

        losses = torch.nn.functional.smooth_l1_loss(
            q_selected,
            target,
            reduction="none",
        )

        loss = (weights * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            indices=indices,
            td_errors=td_errors,
        )

        return float(loss.item())

    def clone_for_eval(self) -> "PerDqnAgent":
        cloned = PerDqnAgent(
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