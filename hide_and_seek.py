from gym import Env, logger, spaces, utils
import numpy as np
import random

class HideAndSeekEnv(Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()

        # Map and Grid size
        self.map = [
            "+-----------------+",
            "| | : | : : | : : |",
            "| : : |-| |-| |-| |",
            "| | : : : : : : : |",
            "|-| : : : : : |-| |",
            "| : | | : : : : : |",
            "| | : : : : : |-| |",
            "| | : : : : : | : |",
            "| |-|-|-| : : |-| |",
            "| : : : : : : : |-|",
            "+-----------------+",
        ]
        self.grid_size = (len(self.map), len(self.map[0]))

        # Action space: stand, move east, move west, move south, move north, catch
        self.action_space = spaces.Discrete(6)

        # Observation space: police positions, thief positions, thief direction (0: east, 1: west, 2: south, 3: north)
        self.observation_space = spaces.Box(low=0, high=max(self.grid_size) - 1, shape=(5,), dtype=np.int32)

        # Initialize state
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None
    ):
        # Find all empty spaces
        empty_spaces = [(y, x) for y, row in enumerate(self.map) for x, cell in enumerate(row) if cell == ' ']

        # Initialize positions: [police_x, police_y, thief_x, thief_y, thief_direction]
        police_y, police_x = 5, 9  # Initial police position at 'x'
        thief_y, thief_x = random.choice(empty_spaces)  # Random initial thief position
        thief_dir = random.randint(0, 3)  # Random initial thief direction (0: east, 1: west, 2: south, 3: north)

        self.state = np.array([police_x, police_y, thief_x, thief_y, thief_dir])

        # Time step
        self.steps = 0

        return self.state

    def step(self, action):
        police_x, police_y, thief_x, thief_y, thief_dir = self.state

        # Reward and done flag
        reward = -1  # time penalty
        done = False

        # Police action
        if action == 1:
            if self.map[police_y][police_x + 1] == ":":
                police_x = min(police_x + 2, self.grid_size[1] - 1)
            else:
                reward = -10
        elif action == 2:
            if self.map[police_y][police_x - 1] == ":":
                police_x = max(police_x - 2, 0)
            else:
                reward = -10
        elif action == 3:
            if self.map[police_y + 1][police_x] == " ":
                police_y = min(police_y + 1, self.grid_size[0] - 1)
            else:
                reward = -10
        elif action == 4:
            if self.map[police_y - 1][police_x] == " ":
                police_y = max(police_y - 1, 0)
            else:
                reward = -10
        elif action == 5:  # catch action
            if ((police_x == thief_x + 1 and police_y == thief_y)
                or (police_x == thief_x - 1 and police_y == thief_y)
                or (police_x == thief_x and police_y == thief_y + 1)
                or (police_x == thief_x and police_y == thief_y - 1)
                or (police_x == thief_x and police_y == thief_y)):
                if not self.is_thief_hiding():
                    reward = 100
                    done = True
                    return np.array([police_x, police_y, thief_x, thief_y, thief_dir]), reward, done, {}
            else:
                reward = -10  # penalty for failed catch


        # Thief action (random)
        thief_action_list = [1, 2, 3, 4]
        while True:
            thief_action = np.random.choice(thief_action_list)
            if thief_action == 1:
                if self.map[thief_y][thief_x + 1] == ":":
                    thief_x = min(thief_x + 2, self.grid_size[1] - 1)
                    thief_dir = 0
                    break
                else:
                    thief_action_list.remove(thief_action)
                    continue
            elif thief_action == 2:
                if self.map[thief_y][thief_x - 1] == ":":
                    thief_x = max(thief_x - 2, 0)
                    thief_dir = 1
                    break
                else:
                    thief_action_list.remove(thief_action)
                    continue
            elif thief_action == 3:
                if self.map[thief_y + 1][thief_x] == " ":
                    thief_y = min(thief_y + 1, self.grid_size[0] - 1)
                    thief_dir = 2
                    break
                else:
                    thief_action_list.remove(thief_action)
                    continue
            elif thief_action == 4:
                if self.map[thief_y - 1][thief_x] == " ":
                    thief_y = max(thief_y - 1, 0)
                    thief_dir = 3
                    break
                else:
                    thief_action_list.remove(thief_action)
                    continue

        # Update state
        self.state = np.array([police_x, police_y, thief_x, thief_y, thief_dir])

        self.steps += 1

        return self.state, reward, done, {}

    def is_thief_hiding(self):
        police_x, police_y, thief_x, thief_y, thief_dir = self.state

        # Check if police is in the line of sight of thief
        if thief_dir == 0:  # Thief looking east
            if police_y == thief_y and police_x >= thief_x:
                return True
        elif thief_dir == 1:  # Thief looking west
            if police_y == thief_y and police_x <= thief_x:
                return True
        elif thief_dir == 2:  # Thief looking south
            if police_x == thief_x and police_y >= thief_y:
                return True
        elif thief_dir == 3:  # Thief looking north
            if police_x == thief_x and police_y <= thief_y:
                return True

        return False

    def render(self, mode='human'):
        grid = [list(row) for row in self.map]

        police_x, police_y, thief_x, thief_y, thief_dir = self.state
        grid[police_y][police_x] = 'P'
        if not self.is_thief_hiding():
            grid[thief_y][thief_x] = f'{thief_dir}'
        else:
            grid[thief_y][thief_x] = f'{thief_dir+5}'  # Hidden state

        for row in grid:
            print(' '.join(row))
        print()

# Create and use the environment
if __name__ == "__main__":
    env = HideAndSeekEnv()
    env.reset()
    env.render()

    done = False
    while not done:
        action = env.action_space.sample()  # Sample random action
        state, reward, done, *_ = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
