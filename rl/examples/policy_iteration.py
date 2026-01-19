if __name__ == "__main__":
    from rl import DeterministicPolicy, Maze
    from rl.algorithms import policy_iteration
    from rl.examples.utils import visualize_maze_policy

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = DeterministicPolicy.initialize_random(maze.states, maze.actions)

    maze.visualize(print_rewards=True)
    print()
    print("Running Policy Iteration...")
    optimal_policy = policy_iteration(
        policy,
        maze,
        discount_factor=0.9,
        max_overall_iterations=100,
        truncate_policy_evaluation_iterations=None,
    )
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
    print()

    print("-" * 70)
    print()

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = DeterministicPolicy.initialize_random(maze.states, maze.actions)

    maze.visualize(print_rewards=True)
    print()
    print("Running Truncated Policy Iteration...")
    optimal_policy = policy_iteration(
        policy,
        maze,
        discount_factor=0.9,
        max_overall_iterations=100,
        truncate_policy_evaluation_iterations=10,
    )
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
