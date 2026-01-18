from typing import Hashable, Literal

from rl.environment import Environment
from rl.policy import Policy


def get_experience_sample(
    policy: Policy,
    environment: Environment,
    reset_to_original_state: bool = False,
    style: Literal["sars", "sarsa"] = "sars",
):
    s = environment.get_current_state()
    a = policy.get_action(s)
    r_prime = environment.take_action(a)
    s_prime = environment.get_current_state()
    if style == "sars":
        experience_sample = (s, a, r_prime, s_prime)
    elif style == "sarsa":
        a_prime = policy.get_action(s_prime)
        experience_sample = (s, a, r_prime, s_prime, a_prime)
    if reset_to_original_state:
        environment.update_state(s)
    return experience_sample


def generate_trajectory(
    policy: Policy,
    environment: Environment,
    max_steps: int,
    end_at_terminal_steps: bool = True,
    reset_to_original_state: bool = False,
    style: Literal["sars", "sarsa"] = "sars",
) -> list[Hashable | float]:
    original_state = environment.get_current_state()

    trajectory = []
    for _ in range(max_steps):
        experience_sample = get_experience_sample(policy, environment, reset_to_original_state=False, style=style)

        if style == "sars":
            pass
        elif style == "sarsa":
            # update a_prime of previous sample to match action of current sample
            _, a, _, _, _ = experience_sample
            if len(trajectory) > 0:
                trajectory[-1] = tuple(list(trajectory[-1])[:-1] + [a])
        else:
            raise NotImplementedError(f"Unknown style: {style}")

        trajectory.append(experience_sample)

        if end_at_terminal_steps and environment.at_terminal_state():
            break

    if reset_to_original_state:
        environment.update_state(original_state)

    return trajectory


if __name__ == "__main__":
    from rl.games import Maze
    from rl.policy import ExploratoryPolicy

    environment = Maze.initialize_random(grid_height=5, grid_width=5, num_exits=3, num_vortexes=7)
    policy = ExploratoryPolicy.initialize_random(states=environment.states, actions=environment.actions)

    environment.visualize()
    print()
    policy.visualize()
    print()
    print("Trajectory (ending at terminal state, sars, reset original state):")
    trajectory1 = generate_trajectory(
        policy,
        environment,
        max_steps=100,
        reset_to_original_state=True,
        end_at_terminal_steps=True,
        style="sars",
    )
    for t in trajectory1:
        print(t)
    print()
    environment.visualize()
    print()
    print("Trajectory (non-ending, sarsa, not resetting to original state):")
    trajectory2 = generate_trajectory(
        policy,
        environment,
        max_steps=10,
        reset_to_original_state=False,
        end_at_terminal_steps=False,
        style="sarsa",
    )
    for t in trajectory2:
        print(t)
    print()
    environment.visualize()
    print()
    print()
