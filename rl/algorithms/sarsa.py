from random import choice

from rl import Environment, EpsilonGreedyPolicy, get_experience_sample
from rl.algorithms.utils import get_state_value_from_action_values


def sarsa(
    policy: EpsilonGreedyPolicy,
    environment: Environment,
    discount_factor: float,
    learning_rate: float,
    num_episodes: int,
    max_steps_in_trajectory: int,
    end_trajectory_at_terminal_state: bool,
):
    q = {s: {a: 0 for a in environment.actions} for s in environment.states}

    for episode_no in range(num_episodes):
        while True:
            s_prime = environment.sample_random_state()
            if not environment.is_terminal_state(s_prime):
                break

        for _ in range(max_steps_in_trajectory):
            s, a, r_prime, s_prime, a_prime = get_experience_sample(
                policy, environment, state=s_prime, style="sarsa", reset_to_original_state=False
            )

            # Policy evaluation
            q[s][a] = q[s][a] - learning_rate * (q[s][a] - r_prime - discount_factor * q[s_prime][a_prime])

            # Policy improvement
            chosen_action = max(q[s], key=q[s].get)
            policy.update_policy(s, chosen_action)

            if end_trajectory_at_terminal_state and environment.at_terminal_state():
                break

        if episode_no % 20 == 0:
            state_values = {state: get_state_value_from_action_values(q, state, policy) for state in environment.states}
            print(f"Iteration {episode_no} complete. Sum of state_values: {sum(state_values.values())}")

    return policy
