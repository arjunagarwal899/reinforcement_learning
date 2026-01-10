import numpy as np


class Maze:
    """A grid-like maze from which an agent has to exit. The agent always starts from the top left corner i.e.
    (1, 1). There are multiple exits in the maze. There are also multiple vortexes in the maze that will injure the
    agent if it steps on them.

    Note: Even if the agent reaches a vortex or an exit, it will still be able to take actions. While they can be
    considered as terminal states, I leave it to the user to choose to do so. There is a utility function for the same.
    """

    def __init__(
        self,
        environment: np.ndarray,
        reward_to_exit: float = 1.0,
        reward_to_enter_vortex: float = -1.0,
        reward_to_move: float = -0.1,
    ):
        self.ensure_initial_environment_correctness(environment)
        self.environment = environment

        self.agent_location = (0, 0)

        self.reward_to_exit = reward_to_exit
        self.reward_to_enter_vortex = reward_to_enter_vortex
        self.reward_to_move = reward_to_move

    @classmethod
    def initialize_random(
        cls,
        grid_height: int,
        grid_width: int,
        num_exits: int,
        num_vortexes: int,
        *args,
        **kwargs,
    ) -> "Maze":
        assert grid_height > 1 and grid_width > 1
        assert num_exits > 0 and num_vortexes >= 0

        environment = np.full((grid_height, grid_width), " ")

        available_locations = np.array(range(grid_height * grid_width - 1)) + 1
        locations_required = num_exits + num_vortexes
        selected_locations = np.random.choice(available_locations, locations_required, replace=False)

        locations_for_exits = selected_locations[:num_exits]
        locations_for_vortexes = selected_locations[num_exits:]

        environment[locations_for_exits // grid_width, locations_for_exits % grid_width] = "E"
        environment[locations_for_vortexes // grid_width, locations_for_vortexes % grid_width] = "V"

        return cls(environment, *args, **kwargs)

    def get_current_state(self) -> tuple[int, int]:
        """Returns the current location of the agent."""
        return self.agent_location

    def at_terminal_state(self) -> bool:
        """Returns True if the agent is at a terminal state (i.e. an exit or a vortex)."""
        return self.environment[self.agent_location] in {"E", "V"}

    def take_action(self, action: int | str) -> float:
        """Takes an action and returns the reward.

        Args:
            action: The action to take. Must be one of 0 / "stay" / 1 / "right" / 2 / "down" / 3 / "left" / 4 / "up".

        Returns:
            The reward for the state action pair.
        """
        # Basic checks
        if not isinstance(action, (int, str)):
            raise ValueError("Action must be an integer or a string.")
        if (isinstance(action, int) and action not in range(5)) or (
            isinstance(action, str) and action not in {"stay", "right", "down", "left", "up"}
        ):
            raise ValueError("Action must be one of 0/stay, 1/right, 2/down, 3/left, 4/up.")

        # Initialize reward
        reward = 0

        # Update the agent state
        match action:
            case "stay" | 0:
                pass
            case "right" | 1:
                reward = self.reward_to_move
                self.agent_location = (self.agent_location[0], min(self.agent_location[1] + 1, self.grid_width - 1))
            case "down" | 2:
                reward = self.reward_to_move
                self.agent_location = (min(self.agent_location[0] + 1, self.grid_height - 1), self.agent_location[1])
            case "left" | 3:
                reward = self.reward_to_move
                self.agent_location = (self.agent_location[0], max(self.agent_location[1] - 1, 0))
            case "up" | 4:
                reward = self.reward_to_move
                self.agent_location = (max(self.agent_location[0] - 1, 0), self.agent_location[1])

        # Update reward
        if self.environment[self.agent_location] == "E":
            reward += self.reward_to_exit
        elif self.environment[self.agent_location] == "V":
            reward += self.reward_to_enter_vortex

        return reward

    def visualize(self):
        """Prints a visual representation of the maze. The agent is represented by 'A' exits by 'E', and vortexes by
        'V'."""
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                tile = self.environment[row][col]

                if tile == " ":
                    tile = "·  "
                elif tile == "E":
                    tile = "\033[32mE\033[0m  "
                elif tile == "V":
                    tile = "\033[31mV\033[0m  "

                if (row, col) == self.agent_location:
                    tile = tile[:-2] + "A "

                if tile == "·A ":
                    tile = "A  "

                print(tile.ljust(3), end="")
            print()

    @staticmethod
    def ensure_initial_environment_correctness(environment: np.ndarray):
        # Check environment shape
        if environment.ndim != 2:
            raise ValueError("Environment must be a 2D array.")

        # Check environment contents
        if not set(np.unique(environment).tolist()).issubset({"E", "V", " "}):
            raise ValueError(
                "Environment must contain only the characters 'E' (Exit), 'V' (Vortex), and ' ' (empty tile)."
            )

        # Ensure top-left corner is empty
        if environment[0, 0] != " ":
            raise ValueError("The top-left corner of the environment must be empty.")

    @property
    def grid_height(self):
        return self.environment.shape[0]

    @property
    def grid_width(self):
        return self.environment.shape[1]


if __name__ == "__main__":
    print("Random maze:")
    maze = Maze.initialize_random(5, 5, 3, 7)
    maze.visualize()

    print("\nCustom maze:")
    sample = np.array(
        [
            [" ", "V", "E"],
            [" ", "V", " "],
            [" ", " ", " "],
            ["V", "V", " "],
            ["E", " ", " "],
        ]
    )

    maze = Maze(sample)
    maze.visualize()

    print("\nCheck reward system")
    for action in [
        "up",  # try hitting the wall
        2,  # down, move in the right direction
        "left",  # try hitting a different wall
        "down",  # move in the right direction
        "stay",  # don't move
        "right",  # move in the right direction
        "up",  # move into a vortex
        "right",  # exit the vortex in the right direction
        "up",  # exit the environment
        "stay",  # don't move
        "up",  # hit the wall
        "left",  # enter the vortex
    ]:
        reward = maze.take_action(action)
        print(f"Agent took action '{action}' and received reward {reward}")
        maze.visualize()
