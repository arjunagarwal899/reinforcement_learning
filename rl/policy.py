from copy import deepcopy
from functools import lru_cache
from typing import Hashable, Self

import numpy as np
from prettytable import PrettyTable


class Policy:
    """Generic Policy class that stores a policy table and samples actions from it."""

    def __init__(self, action_probabilities: dict[tuple[Hashable, Hashable], float]):
        """Initialize the policy.

        Args:
            action_probabilities: A dictionary mapping state-action pairs to probabilities.
        """
        # Store action probabilities as a policy table
        self.policy_table: dict[Hashable, dict[Hashable, float]] = {}
        for (state, action), probability in action_probabilities.items():
            if state not in self.policy_table:
                self.policy_table[state] = {}
            self.policy_table[state][action] = probability

        # Ensure the policy table is valid
        self.check_policy_table()

    @classmethod
    def initialize_random(cls, states: list[Hashable], actions: list[Hashable]):
        """Initialize a random policy.

        Args:
            states: A list of states.
            actions: A list of actions.

        Returns:
            A random policy.
        """
        action_probabilities = {}
        for state in states:
            random_values = np.random.random(len(actions))
            normalized_random_values = random_values / random_values.sum()
            for i, action in enumerate(actions):
                action_probabilities[(state, action)] = normalized_random_values[i].item()
        return cls(action_probabilities)

    def check_policy_table(self):
        """Returns True if the policy table is initialized and complete (i.e. all states and actions have probabilities
        which sum to 1 for each state)."""
        for state in self.states:
            if state not in self.policy_table:
                raise ValueError(f"State {state} is not in the policy table.")

            total_probability = 0.0
            for action in self.actions:
                if action not in self.policy_table[state]:
                    raise ValueError(f"Action {action} is not in the policy table for state {state}.")
                if np.isnan(self.policy_table[state][action]):
                    raise ValueError(f"Probability for state {state} and action {action} is not a number.")
                if self.policy_table[state][action] < 0 or self.policy_table[state][action] > 1:
                    raise ValueError(f"Probability for state {state} and action {action} is not between 0 and 1.")
                total_probability += self.policy_table[state][action]

            if not np.isclose(total_probability, 1.0):
                raise ValueError(f"Sum of probabilities for state {state} is not 1.")

    def get_action(self, state: Hashable) -> Hashable:
        """For a given state, look up the policy table and return an action according to the policy.

        Args:
            state: The state to get the action for.

        Returns:
            The action to take.
        """
        self.check_policy_table()
        if state not in self.policy_table:
            raise ValueError(f"State {state} is not in the policy table.")
        action_probabilities = self.policy_table[state]
        return np.random.choice(list(action_probabilities.keys()), p=list(action_probabilities.values())).item()

    def get_action_probabilities(self, state: Hashable) -> dict[Hashable, float]:
        """Get the action probabilities for a given state.

        Args:
            state: The state to get the action probabilities for.

        Returns:
            A dictionary mapping actions to probabilities for that particular state.
        """
        return self.policy_table[state]

    def update_policy(self, state: Hashable, action_probabilities: dict[Hashable, float]):
        """Update the policy table for a given state.

        Args:
            state: The state to update the policy for.
            action_probabilities: A dictionary mapping actions to probabilities for that particular state.
        """
        if len(action_probabilities) != len(self.actions):
            raise ValueError("Please provide action probabilities for all actions.")

        if not np.isclose(sum(action_probabilities.values()), 1.0):
            raise ValueError("Sum of all actions for each state is not 1.")

        self.policy_table[state] = action_probabilities

    def visualize(self):
        """Prints a visual representation of the policy table."""
        print("Policy table:")
        table = PrettyTable()
        table.border = False
        table.field_names = ["state \\ action"] + list(self.actions)
        rows = []
        for state in self.states:
            row = [state]
            for action in self.actions:
                row.append(round(self.policy_table[state][action], 3))
            rows.append(row)
        table.add_rows(rows)
        print(table)

    def copy(self) -> Self:
        """Returns a deep copy of the policy."""
        return deepcopy(self)

    @property
    def states(self) -> set[Hashable]:
        return set(self.policy_table.keys())

    @property
    def actions(self) -> set[Hashable]:
        return set(sum([list(self.policy_table[state].keys()) for state in self.states], []))


class EpsilonGreedyPolicy(Policy):
    def __init__(self, action_probabilities: dict[tuple[Hashable, Hashable], float], epsilon: float, **kwargs):
        super().__init__(action_probabilities, **kwargs)

        self.epsilon = epsilon
        self.check_epsilon()

    @classmethod
    def initialize_random(cls, states: list[Hashable], actions: list[Hashable], epsilon: float):
        """Initialize a random policy in an epsilon-greedy fashion.

        Args:
            states: A list of states.
            actions: A list of actions.
            epsilon: The epsilon value. Measures the amount of exploration.

        Returns:
            A random policy.
        """
        action_probabilities = {}
        for state in states:
            random_action = np.random.choice(actions).item()
            for action in actions:
                if action == random_action:
                    action_probabilities[(state, random_action)] = cls.get_epsilon_greedy_probability(
                        epsilon, len(actions)
                    )
                else:
                    action_probabilities[(state, action)] = cls.get_epsilon_non_greedy_probability(
                        epsilon, len(actions)
                    )
        return cls(action_probabilities, epsilon=epsilon)

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon
        self.check_epsilon()

    def check_epsilon(self):
        """Raises an error for invalid epsilon values."""
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError("Epsilon must be between 0 and 1.")

    def update_policy(self, state: Hashable, chosen_action: Hashable):
        """Update the policy table for a given state.

        Args:
            state: The state to update the policy for.
            action: The epsilon greedy state.
        """
        action_probabilities = {}
        for action in self.actions:
            if action == chosen_action:
                action_probabilities[action] = self.get_epsilon_greedy_probability(self.epsilon, len(self.actions))
            else:
                action_probabilities[action] = self.get_epsilon_non_greedy_probability(self.epsilon, len(self.actions))

        return super().update_policy(state, action_probabilities)

    @staticmethod
    @lru_cache(maxsize=64)
    def get_epsilon_greedy_probability(epsilon: float, num_actions: int):
        return 1 - (epsilon / num_actions) * (num_actions - 1)

    @staticmethod
    @lru_cache(maxsize=64)
    def get_epsilon_non_greedy_probability(epsilon: float, num_actions: int):
        return epsilon / num_actions


class DeterministicPolicy(EpsilonGreedyPolicy):
    """Deterministic policy is a special case of epsilon-greedy policy where epsilon is 0 i.e. only one action is chosen
    for each state (a completely greedy policy)."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("epsilon", None)
        super().__init__(*args, epsilon=0.0, **kwargs)

    @classmethod
    def initialize_random(cls, states: list[Hashable], actions: list[Hashable]):
        return super().initialize_random(states, actions, epsilon=0.0)

    def visualize(self):
        print("Policy table:")
        for state in self.states:
            action = self.get_action(state)
            print(f"{state} -> {action}")


class ExploratoryPolicy(EpsilonGreedyPolicy):
    """An exploratory policy is a special case of epsilon-greedy policy where epsilon is 1 i.e. only all actions have
    equal probability to be chosen for each state."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("epsilon", None)
        super().__init__(*args, epsilon=1.0, **kwargs)

    @classmethod
    def initialize_random(cls, states: list[Hashable], actions: list[Hashable]):
        return super().initialize_random(states, actions, epsilon=1.0)


if __name__ == "__main__":
    print("Creating valid policy:")
    action_probabilities = {
        ("s0", "a0"): 0.5,
        ("s0", "a1"): 0.5,
        ("s1", "a0"): 0.25,
        ("s1", "a1"): 0.75,
    }
    print(action_probabilities)
    policy = Policy(action_probabilities)
    policy.visualize()
    print()
    print("Creating invalid policy:")
    action_probabilities = {
        ("s0", "a0"): 0.5,
        ("s0", "a1"): 0.5,
        ("s1", "a0"): 0.25,
        ("s1", "a1"): 0.99,
    }
    print(action_probabilities)
    try:
        policy = Policy(action_probabilities)
    except Exception as e:
        print(e)
    print()
    print("Creating invalid policy:")
    action_probabilities = {
        ("s0", "a0"): 0.5,
        ("s0", "a1"): 0.5,
        ("s1", "a0"): 0.25,
    }
    print(action_probabilities)
    try:
        policy = Policy(action_probabilities)
    except Exception as e:
        print(e)
    print()
    print("Creating random policy")
    policy = Policy.initialize_random(["s0", "s1", "s2", "s3", "s4"], ["a0", "a1", "a2", "a3"])
    policy.visualize()
    print()
    print("Sampling 10 actions from each state:")
    for state in policy.states:
        actions = []
        for _ in range(10):
            actions.append(policy.get_action(state))
        print(f"State {state}: {actions}")
    print()
    policy.update_policy("s0", {"a0": 1.0, "a1": 0.0, "a2": 0.0, "a3": 0.0})
    print("Updated policy for state s0 to be deterministic")
    policy.visualize()
    print(" Sampling 10 actions for state s0:")
    print(f"State s0: {[policy.get_action('s0') for _ in range(10)]}")
    print()
    print("Creating an epsilon greedy policy with different epsilons:")
    for epsilon in [0.25, 0.5, 0.75]:
        policy = EpsilonGreedyPolicy.initialize_random(
            ["s0", "s1", "s2", "s3", "s4"], ["a0", "a1", "a2", "a3"], epsilon
        )
        print(f"Epsilon: {epsilon}")
        policy.visualize()
        print()
    print("Creating a determinsitic policy:")
    policy = DeterministicPolicy.initialize_random(["s0", "s1", "s2", "s3", "s4"], ["a0", "a1", "a2", "a3"])
    policy.visualize()
    print()
    print("Creating an exploratory policy:")
    policy = ExploratoryPolicy.initialize_random(["s0", "s1", "s2", "s3", "s4"], ["a0", "a1", "a2", "a3"])
    policy.visualize()
    print()
