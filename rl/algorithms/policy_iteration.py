from rl import DeterministicPolicy, Environment
from rl.algorithms.utils import (
    calculate_action_value,
    calculate_state_value,
    has_policy_converged,
    have_state_values_converged,
)


def policy_iteration(
    policy: DeterministicPolicy,
    environment: Environment,
    discount_factor: float = 0.9,
    max_overall_iterations: int = 100,
    truncate_policy_evaluation_iterations: int | None = None,
):
    for overall_iteration in range(max_overall_iterations):
        old_policy = policy.copy()

        # Policy evaluation
        state_values = {s: 0 for s in environment.states}
        policy_evaluation_iteration = 0
        while True:
            old_state_values = state_values.copy()
            for s in environment.states:
                state_values[s] = calculate_state_value(s, policy, environment, state_values, discount_factor)

            if have_state_values_converged(old_state_values, state_values):
                break

            if (
                truncate_policy_evaluation_iterations is not None
                and policy_evaluation_iteration >= truncate_policy_evaluation_iterations
            ):
                break

            policy_evaluation_iteration += 1

        # Policy improvement
        for s in environment.states:
            action_values = {}
            for a in environment.actions:
                action_values[a] = calculate_action_value(s, a, environment, state_values, discount_factor)

            chosen_action, _ = max(action_values.items(), key=lambda x: x[1])
            policy.update_policy(s, chosen_action)

        print(
            f"Iteration {overall_iteration} complete. Ran policy evalutation for {policy_evaluation_iteration + 1} "
            f"iterations. Sum of state_values: {sum(state_values.values())}"
        )

        if has_policy_converged(old_policy, policy):
            print(f"Converged in {overall_iteration + 1} iterations.")
            break

    return policy
