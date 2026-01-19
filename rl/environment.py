from abc import ABC, abstractmethod
from typing import Hashable


class Environment(ABC):
    @abstractmethod
    def get_current_state(self) -> Hashable:
        pass

    @abstractmethod
    def sample_random_state(self) -> Hashable:
        pass

    @abstractmethod
    def update_state(self, state):
        pass

    @abstractmethod
    def at_terminal_state(self) -> bool:
        """Whether or not the agent is currently at a terminal state."""

    @abstractmethod
    def is_terminal_state(self, state: Hashable) -> bool:
        """Whether or not the provided state is a terminal state."""

    @abstractmethod
    def take_action(self, action) -> float:
        """Takes an action and returns the reward."""

    def visualize(self):
        raise NotImplementedError("This method has not been implemented by the subclass.")

    @property
    @abstractmethod
    def states(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def actions(self):
        raise NotImplementedError
