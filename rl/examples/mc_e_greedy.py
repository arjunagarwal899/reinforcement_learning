if __name__ == "__main__":
    from rl import EpsilonGreedyPolicy, Maze
    from rl.algorithms import mc_e_greedy
    from rl.examples.utils import visualize_maze_policy

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = EpsilonGreedyPolicy.initialize_random(maze.states, maze.actions, epsilon=0.2)

    maze.visualize(print_rewards=True)
    print()
    print("Running MC ε-greedy...")
    optimal_policy = mc_e_greedy(
        policy,
        maze,
        discount_factor=0.9,
        num_episodes=500,
        max_steps_in_trajectory=10,
        end_trajectory_at_terminal_state=False,
    )
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
