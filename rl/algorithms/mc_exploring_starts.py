from rl import DeterministicPolicy, Environment, generate_trajectory
from rl.algorithms.utils import get_state_value_from_action_values


def mc_exploring_starts(
    policy: DeterministicPolicy,
    environment: Environment,
    discount_factor: float,
    num_episodes: int,
    max_steps_in_trajectory: int,
    end_trajectory_at_terminal_state: bool,
):
    returns = {s: {a: [] for a in environment.actions} for s in environment.states}
    action_values = {s: {a: 0 for a in environment.actions} for s in environment.states}

    start_state_index = 0
    start_action_index = 0
    for episode_no in range(num_episodes):
        start_state = environment.states[start_state_index % len(environment.states)]
        start_action = environment.actions[start_action_index % len(environment.actions)]
        start_action_index += 1
        if start_action_index % len(environment.actions) == 0:
            start_state_index += 1

        trajectory = generate_trajectory(
            policy,
            environment,
            start_state=start_state,
            start_action=start_action,
            max_steps=max_steps_in_trajectory,
            end_at_terminal_steps=end_trajectory_at_terminal_state,
        )

        # Policy evaluation using every-visit
        g = 0
        for s, a, r, _ in reversed(trajectory):
            g = discount_factor * g + r
            returns[s][a].append(g)

        # Policy improvement
        for s in environment.states:
            for a in environment.actions:
                action_values[s][a] = sum(returns[s][a]) / max(1, len(returns[s][a]))
            chosen_action, _ = max(action_values[s].items(), key=lambda x: x[1])
            policy.update_policy(s, chosen_action)

        if episode_no % 60 == 0:
            state_values = {
                state: get_state_value_from_action_values(action_values, state, policy) for state in environment.states
            }
            print(f"Iteration {episode_no} complete. Sum of state_values: {sum(state_values.values())}")

    # Ensure every state-action-pair has been visited at least once
    not_visited = []
    for s in environment.states:
        for a in environment.actions:
            if len(returns[s][a]) == 0:
                not_visited.append((s, a))
    if len(not_visited) > 0:
        raise RuntimeError(f"State-action pairs {not_visited} have/has not been visited.")

    return policy
