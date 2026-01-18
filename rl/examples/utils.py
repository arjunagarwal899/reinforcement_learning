from time import time
from typing import Hashable

from rl import Maze, Policy


def timeit():
    pass


def have_state_values_converged(
    old_state_values: dict[Hashable, float],
    new_state_values: dict[Hashable, float],
    tolerance: float = 0.01,
):
    for state in new_state_values:
        if new_state_values[state] - old_state_values[state] > tolerance:
            return False
    return True


def has_policy_converged(
    old_policy: Policy,
    new_policy: Policy,
    tolerance: float = 0.01,
):
    for state in new_policy.states:
        for action in new_policy.actions:
            if (
                new_policy.get_action_probabilities(state)[action] - old_policy.get_action_probabilities(state)[action]
                > tolerance
            ):
                return False
    return True


def calculate_action_value(
    state: Hashable,
    action: Hashable,
    environment: Maze,
    state_values: dict[Hashable, float],
    discount_factor: float,
):
    environment.update_state(state)
    r = environment.take_action(action)
    s_prime = environment.get_current_state()
    action_value = r + discount_factor * state_values[s_prime]
    return action_value


def calculate_state_value(
    state: Hashable,
    policy: Policy,
    environment: Maze,
    state_values: dict[Hashable, float],
    discount_factor: float,
):
    state_value = 0
    action_probabilities = policy.get_action_probabilities(state)
    for action, probability in action_probabilities.items():
        action_value = calculate_action_value(state, action, environment, state_values, discount_factor)
        state_value += probability * action_value
    return state_value


def visualize_maze_policy(policy: Policy, maze: Maze):
    # Easy visualization of the optimal policy
    for row in range(maze.grid_height):
        for col in range(maze.grid_width):
            tile = maze.environment[row][col]

            if tile == " ":
                tile = " "
            elif tile == "E":
                tile = "\033[32mE\033[0m"
            elif tile == "V":
                tile = "\033[31mV\033[0m"

            state = (row, col)
            action = policy.get_action(state)

            if action == "up":
                tile += "↑"
            elif action == "down":
                tile += "↓"
            elif action == "left":
                tile += "←"
            elif action == "right":
                tile += "→"
            elif action == "stay":
                tile += "↺"

            print(tile, end="  ")
        print()
