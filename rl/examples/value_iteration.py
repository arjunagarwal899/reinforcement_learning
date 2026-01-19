if __name__ == "__main__":
    from rl import DeterministicPolicy, Maze
    from rl.algorithms import value_iteration
    from rl.examples.utils import visualize_maze_policy

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = DeterministicPolicy.initialize_random(maze.states, maze.actions)

    maze.visualize(print_rewards=True)
    print()
    print("Running Value Iteration...")
    optimal_policy = value_iteration(
        policy,
        maze,
        discount_factor=0.9,
        max_iterations=100,
    )
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
