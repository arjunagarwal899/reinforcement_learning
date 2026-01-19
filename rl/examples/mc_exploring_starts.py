if __name__ == "__main__":
    from rl import DeterministicPolicy, Maze
    from rl.algorithms import mc_exploring_starts
    from rl.examples.utils import visualize_maze_policy

    maze = Maze.initialize_random(6, 6, 3, 7)
    policy = DeterministicPolicy.initialize_random(maze.states, maze.actions)

    maze.visualize(print_rewards=True)
    print()
    print("Running MC Basic...")
    optimal_policy = mc_exploring_starts(policy, maze)
    print()
    optimal_policy.visualize()
    print()
    visualize_maze_policy(optimal_policy, maze)
