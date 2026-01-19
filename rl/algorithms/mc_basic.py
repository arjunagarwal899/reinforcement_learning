from rl import DeterministicPolicy, Environment, generate_trajectory
from rl.algorithms.utils import get_state_value_from_action_values, have_action_values_converged


def mc_basic(
    policy: DeterministicPolicy,
    environment: Environment,
    discount_factor: float = 0.9,
    max_iterations: int = 100,
    num_samples_per_iteration: int = 10,
    max_steps_in_trajectory: int = 10,
    end_trajectory_at_terminal_state: bool = True,
):
    action_values = {s: {a: -1000 for a in environment.actions} for s in environment.states}
    for iteration in range(max_iterations):
        old_action_values = action_values.copy()

        for s in environment.states:
            # Policy evaluation
            for a in environment.actions:
                returns = []
                for _ in range(num_samples_per_iteration):
                    trajectory = generate_trajectory(
                        policy,
                        environment,
                        start_state=s,
                        start_action=a,
                        max_steps=max_steps_in_trajectory,
                        end_at_terminal_steps=end_trajectory_at_terminal_state,
                    )
                    g = 0
                    for _, _, r, _ in reversed(trajectory):
                        g = discount_factor * g + r
                    returns.append(g)
                action_values[s][a] = sum(returns) / len(returns)

            # Policy improvement
            chosen_action, _ = max(action_values[s].items(), key=lambda x: x[1])
            policy.update_policy(s, chosen_action)

        state_values = {
            state: get_state_value_from_action_values(action_values, state, policy) for state in environment.states
        }
        print(f"Iteration {iteration} complete. Sum of state_values: {sum(state_values.values())}")

        if have_action_values_converged(old_action_values, action_values):
            print(f"Converged in {iteration + 1} iterations.")
            break

    return policy
