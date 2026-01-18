from rl import DeterministicPolicy, Maze
from rl.examples.utils import (
    calculate_action_value,
    calculate_state_value,
    has_policy_converged,
    have_state_values_converged,
    visualize_maze_policy,
)


def policy_iteration(
    policy: DeterministicPolicy,
    environment: Maze,
    discount_factor: float = 0.9,
    max_overall_iterations: int = 100,
    max_policy_evaluation_iterations: int = 100,
):
    for overall_iteration in range(max_overall_iterations):
        old_policy = policy.copy()

        # Policy evaluation
        state_values = {s: 0 for s in environment.states}
        for policy_evaluation_iteration in range(max_policy_evaluation_iterations):
            old_state_values = state_values.copy()
            for s in environment.states:
                state_values[s] = calculate_state_value(s, policy, environment, state_values, discount_factor)

            if have_state_values_converged(old_state_values, state_values):
                break

        # Policy improvement
        for s in environment.states:
            action_values = {}
            for a in environment.actions:
                action_values[a] = calculate_action_value(s, a, environment, state_values, discount_factor)

            chosen_action, _ = max(action_values.items(), key=lambda x: x[1])
            policy.update_policy(s, chosen_action)

        print(
            f"Iteration {overall_iteration} complete. Ran policy evalutation for {policy_evaluation_iteration} "
            f"iterations. Sum of state_values: {sum(state_values.values())}"
        )

        if has_policy_converged(old_policy, policy):
            print(f"Converged in {overall_iteration} iterations.")
            break

    return policy


if __name__ == "__main__":
    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = DeterministicPolicy.initialize_random(maze.states, maze.actions)

    maze.visualize()
    print()
    print("Running Policy Iteration...")
    optimal_policy = policy_iteration(policy, maze)
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
