import copy

import torch
from torch.optim import Adam

from src.agents.dqn_agent import DqnAgent
from src.agents.networks.distributional_q_network import DistributionalQNetwork
from src.utils.enums import Actions


class DistributionalDqnAgent(DqnAgent):
    def __init__(
        self,
        input_size: int,
        settings,
        name: str = "Distributional DQN",
        device=None,
    ):
        super().__init__(
            input_size=input_size,
            settings=settings,
            name=name,
            device=device,
        )

        self.model = DistributionalQNetwork(
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

    def predict(self, obs=None) -> Actions:
        if obs is None:
            return Actions.NONE

        self.model.eval()

        with torch.no_grad():
            obs_t = self._prepare_obs(obs)
            q_values = self.model.get_q_values(obs_t)
            action_idx = int(torch.argmax(q_values, dim=1).item())

        return Actions(action_idx)

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

        batch_size = states.size(0)

        distributions = self.model(states)

        action_distributions = distributions[
            torch.arange(batch_size, device=self.device),
            actions,
        ]

        action_distributions = action_distributions.clamp(min=1e-8)

        with torch.no_grad():
            next_distributions = self.target_model(next_states)
            next_q_values = self.target_model.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)

            next_action_distributions = next_distributions[
                torch.arange(batch_size, device=self.device),
                next_actions,
            ]

            target_support = (
                rewards.unsqueeze(1)
                + self.settings.gamma
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

        loss = -torch.sum(
            projected_distribution * torch.log(action_distributions),
            dim=1,
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def clone_for_eval(self) -> "DistributionalDqnAgent":
        cloned = DistributionalDqnAgent(
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