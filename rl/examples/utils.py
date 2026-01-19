from rl import Maze, Policy


def visualize_maze_policy(policy: Policy, maze: Maze):
    # Easy visualization of the optimal policy
    for row in range(maze.grid_height):
        for col in range(maze.grid_width):
            tile = maze.environment[row][col]

            if tile == " ":
                tile = " "
            elif tile == "E":
                tile = "\033[32mE\033[0m"
            elif tile == "V":
                tile = "\033[31mV\033[0m"

            state = (row, col)
            action_probabilities = policy.get_action_probabilities(state)
            action = max(action_probabilities, key=action_probabilities.get)

            if action == "up":
                tile += "↑"
            elif action == "down":
                tile += "↓"
            elif action == "left":
                tile += "←"
            elif action == "right":
                tile += "→"
            elif action == "stay":
                tile += "↺"

            print(tile, end="  ")
        print()
