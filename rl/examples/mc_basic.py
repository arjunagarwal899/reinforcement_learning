if __name__ == "__main__":
    from rl import DeterministicPolicy, Maze
    from rl.algorithms import mc_basic
    from rl.examples.utils import visualize_maze_policy

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = DeterministicPolicy.initialize_random(maze.states, maze.actions)

    maze.visualize(print_rewards=True)
    print()
    print("Running MC Basic...")
    # Since the environment is deterministic, all trajectories from state s and action a will give the same return
    optimal_policy = mc_basic(
        policy,
        maze,
        discount_factor=0.9,
        max_iterations=100,
        num_samples_per_iteration=1,
        max_steps_in_trajectory=10,
        end_trajectory_at_terminal_state=True,
    )
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
