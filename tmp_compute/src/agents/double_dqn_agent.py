import copy

import torch

from src.agents.dqn_agent import DqnAgent


class DoubleDqnAgent(DqnAgent):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "Double DQN",
        device=None,
    ):
        super().__init__(
            input_size=input_size,
            settings=settings,
            name=name,
            device=device,
        )

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

        q_values = self.model(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN:
            # 1. online model wybiera akcję
            next_online_q_values = self.model(next_states)
            next_actions = next_online_q_values.argmax(dim=1)

            # 2. target model ocenia tę akcję
            next_target_q_values = self.target_model(next_states)
            next_q_selected = next_target_q_values.gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)

            target = rewards + self.settings.gamma * (1.0 - dones) * next_q_selected

        loss = self.loss_fn(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def clone_for_eval(self) -> "DoubleDqnAgent":
        cloned = DoubleDqnAgent(
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