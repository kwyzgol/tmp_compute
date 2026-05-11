import pickle
import random
from abc import ABC, abstractmethod

import torch

from src.utils.enums import Actions, AgentType


_TORCH_INTEROP_THREADS_CONFIGURED = False


def configure_torch_runtime(settings=None) -> None:
    global _TORCH_INTEROP_THREADS_CONFIGURED

    torch_num_threads = getattr(settings, "torch_num_threads", None)
    if torch_num_threads is None:
        return

    torch_num_threads = int(torch_num_threads)
    if torch_num_threads <= 0:
        return

    torch.set_num_threads(torch_num_threads)

    if _TORCH_INTEROP_THREADS_CONFIGURED:
        return

    try:
        torch.set_num_interop_threads(torch_num_threads)
    except RuntimeError:
        pass

    _TORCH_INTEROP_THREADS_CONFIGURED = True


class Agent:
    def __init__(self, name: str, agent_type: AgentType = AgentType.BASIC):
        self.name = name
        self.agent_type = agent_type

    def predict(self, obs=None) -> Actions:
        raise NotImplementedError

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class AgentNN(Agent):
    def __init__(self, name: str):
        super().__init__(name, AgentType.RL)

    @staticmethod
    def resolve_device(settings=None, device=None) -> torch.device:
        configure_torch_runtime(settings)

        if device is not None:
            return torch.device(device)

        if getattr(settings, "force_cpu_training", False):
            return torch.device("cpu")

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_save_object(self):
        clone_for_eval = getattr(self, "clone_for_eval", None)
        if not callable(clone_for_eval):
            return self

        saved_agent = clone_for_eval()
        saved_agent.name = self.name
        saved_agent.training_steps = getattr(self, "training_steps", 0)

        return saved_agent

    def save(self, filename: str, eval_only: bool = True):
        torch.save(self._get_save_object() if eval_only else self, filename)

    @classmethod
    def load(cls, filename: str, device='cpu'):
        return torch.load(filename, map_location=device, weights_only=False)


class LazyAgent(Agent):
    def __init__(self):
        super().__init__('LazyAgent')

    def predict(self, obs=None) -> Actions:
        return Actions.NONE


class RandomAgent(Agent):
    def __init__(self):
        super().__init__('RandomAgent')

    def predict(self, obs=None) -> Actions:
        rng = random.Random()
        action = rng.choice(list(Actions))
        return action
