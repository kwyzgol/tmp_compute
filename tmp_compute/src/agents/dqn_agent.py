import copy
import random
from collections import deque
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Adam

from src.agents.agents import AgentNN
from src.agents.networks.custom_network import CustomNetwork
from src.utils.enums import Actions
from src.utils.settings import GlobalSettings


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            Transition(
                state=state.detach().cpu(),
                action=int(action),
                reward=float(reward),
                next_state=next_state.detach().cpu(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DqnAgent(AgentNN):
    def __init__(
        self,
        input_size: int,
        settings: GlobalSettings,
        name: str = "DQN",
        device: str | torch.device | None = None,
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

        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.settings.learning_rate,
        )

        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.settings.replay_buffer_size)

        self.training_steps = 0

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.float()

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return obs.to(self.device)

    def _epsilon(self, step: int) -> float:
        epsilon_start = self.settings.epsilon_start
        epsilon_end = self.settings.epsilon_end
        epsilon_decay = self.settings.epsilon_decay

        progress = min(step / epsilon_decay, 1.0)

        return epsilon_start + progress * (epsilon_end - epsilon_start)

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
        epsilon = self._epsilon(step)

        if random.random() < epsilon:
            return random.choice(list(Actions))

        return self.predict(obs)

    def remember(
        self,
        state: torch.Tensor,
        action: Actions,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        self.replay_buffer.push(
            state=state,
            action=action.value,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.settings.warmup_steps:
            return None

        if len(self.replay_buffer) < self.settings.batch_size:
            return None

        batch = self.replay_buffer.sample(self.settings.batch_size)

        states = torch.stack([t.state for t in batch]).float().to(self.device)
        next_states = torch.stack([t.next_state for t in batch]).float().to(self.device)

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

        if states.dim() == 3:
            states = states.squeeze(1)

        if next_states.dim() == 3:
            next_states = next_states.squeeze(1)

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_max = next_q_values.max(dim=1).values

            target = rewards + self.settings.gamma * (1.0 - dones) * next_q_max

        loss = self.loss_fn(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def clone_for_eval(self) -> "DqnAgent":
        cloned = DqnAgent(
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
