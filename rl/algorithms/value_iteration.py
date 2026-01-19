from rl import DeterministicPolicy, Environment
from rl.algorithms.utils import calculate_action_value, have_state_values_converged


def value_iteration(
    policy: DeterministicPolicy,
    environment: Environment,
    discount_factor: float,
    max_iterations: int,
):
    state_values = {s: 0 for s in environment.states}
    for iteration in range(max_iterations):
        old_state_values = state_values.copy()
        for s in environment.states:
            action_values = {}
            for a in environment.actions:
                action_values[a] = calculate_action_value(s, a, environment, state_values, discount_factor)

            chosen_action, action_value = max(action_values.items(), key=lambda x: x[1])

            # Policy Update
            policy.update_policy(s, chosen_action)

            # Value update
            state_values[s] = action_value

        if iteration % 10 == 0:
            print(f"Iteration {iteration} complete. Sum of state_values: {sum(state_values.values())}")

        if have_state_values_converged(old_state_values, state_values):
            print(f"Converged in {iteration + 1} iterations.")
            break

    return policy
