if __name__ == "__main__":
    from rl import EpsilonGreedyPolicy, Maze
    from rl.algorithms import sarsa
    from rl.examples.utils import visualize_maze_policy

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = EpsilonGreedyPolicy.initialize_random(maze.states, maze.actions, epsilon=0.5)

    maze.visualize(print_rewards=True)
    print()
    print("Running SARSA...")
    optimal_policy = sarsa(
        policy,
        maze,
        discount_factor=0.9,
        learning_rate=0.1,
        num_episodes=100,
        max_steps_in_trajectory=100,
        end_trajectory_at_terminal_state=False,
    )
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
