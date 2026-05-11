from collections import deque
from dataclasses import dataclass

import torch

from src.agents.dqn_agent import DqnAgent


@dataclass
class NStepTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    discount: float


class NStepReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: NStepTransition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        import random
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


@dataclass
class OneStepTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class NStepDqnAgent(DqnAgent):
    def __init__(self, input_size, settings, name: str = "N-step DQN", device=None):
        super().__init__(
            input_size=input_size,
            settings=settings,
            name=name,
            device=device,
        )

        self.n_step = max(1, int(settings.n_step))

        self.replay_buffer = NStepReplayBuffer(settings.replay_buffer_size)
        self.n_step_buffer = deque(maxlen=self.n_step)

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

        n_step_transition = NStepTransition(
            state=first.state,
            action=first.action,
            reward=reward_sum,
            next_state=final_next_state,
            done=final_done,
            discount=discount,
        )

        self.replay_buffer.push(n_step_transition)

    def optimize(self) -> float | None:
        if len(self.replay_buffer) < self.settings.warmup_steps:
            return None

        if len(self.replay_buffer) < self.settings.batch_size:
            return None

        batch = self.replay_buffer.sample(self.settings.batch_size)

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

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_max = next_q_values.max(dim=1).values

            target = rewards + discounts * (1.0 - dones) * next_q_max

        loss = self.loss_fn(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def clone_for_eval(self) -> "NStepDqnAgent":
        cloned = NStepDqnAgent(
            input_size=self.input_size,
            settings=self.settings,
            name=f"{self.name}_eval_copy",
            device=self.device,
        )

        cloned.model.load_state_dict(self.model.state_dict())
        cloned.target_model.load_state_dict(self.target_model.state_dict())

        cloned.model.eval()
        cloned.target_model.eval()

        return cloned