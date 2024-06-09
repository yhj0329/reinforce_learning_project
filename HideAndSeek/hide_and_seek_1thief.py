from gym import Env, logger, spaces, utils
import numpy as np
import random
from typing import Optional


class HideAndSeekEnv(Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()

        # Map and Grid size
        self.map = [
            "o-----------------o",
            "| |   |     |     |",
            "|     |-| | |-|-| |",
            "| |       | |     |",
            "|-|           |-| |",
            "|   | |           |",
            "| |           |-| |",
            "| |           |   |",
            "| |-|-|-|     |-| |",
            "|               |-|",
            "o-----------------o",
        ]
        self.grid_size = (len(self.map), len(self.map[0]))

        # Action space: stand, move east, move west, move south, move north, catch
        self.action_space = spaces.Discrete(5)

        # Observation space: police positions, thief positions, thief direction (0: east, 1: west, 2: south, 3: north)
        self.observation_space = spaces.Discrete(26244) # spaces.Box(low=0, high=max(self.grid_size) - 1, shape=(5,), dtype=np.int32)

        # Initialize state
        self.reset()

    def encode(self, police_x, police_y, thief_x, thief_y, thief_dir):
        police_x = (police_x - 1) // 2
        police_y = police_y - 1
        thief_x = (thief_x - 1) // 2
        thief_y = thief_y - 1
        i = police_x
        i *= 9
        i += police_y
        i *= 9
        i += thief_x
        i *= 9
        i += thief_y
        i *= 4
        i += thief_dir
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 9 + 1)
        i = i // 9
        out.append((i % 9 * 2) + 1)
        i = i // 9
        out.append(i % 9 + 1)
        i = i // 9
        out.append((i * 2) + 1)
        return reversed(out)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        # Find all empty spaces
        empty_spaces = [(y, x) for y, row in enumerate(self.map) for x, cell in enumerate(row) if cell == ' ' and x % 2 != 0]

        # Initialize positions: [police_x, police_y, thief_x, thief_y, thief_direction]
        police_y, police_x = 1, 1  # Initial police position at 'x'
        thief_y, thief_x = random.choice(empty_spaces)  # Random initial thief position
        thief_dir = random.randint(0, 3)  # Random initial thief direction (0: east, 1: west, 2: south, 3: north)

        self.state = self.encode(police_x, police_y, thief_x, thief_y, thief_dir)

        # Time step
        self.steps = 0

        return self.state

    def step(self, action):
        police_x, police_y, thief_x, thief_y, thief_dir = self.decode(self.state)

        # Reward and done flag
        reward = -1  # time penalty
        done = False

        # Police action
        if action == 0:
            if self.map[police_y][police_x + 1] == " ":
                police_x = min(police_x + 2, self.grid_size[1] - 1)
            else:
                reward = -10
        elif action == 1:
            if self.map[police_y][police_x - 1] == " ":
                police_x = max(police_x - 2, 0)
            else:
                reward = -10
        elif action == 2:
            if self.map[police_y + 1][police_x] == " ":
                police_y = min(police_y + 1, self.grid_size[0] - 1)
            else:
                reward = -10
        elif action == 3:
            if self.map[police_y - 1][police_x] == " ":
                police_y = max(police_y - 1, 0)
            else:
                reward = -10
        elif action == 4:  # catch action
            if thief_dir != 4:
                if ((police_x == thief_x + 2 and police_y == thief_y and police_x + 1 == " ")
                    or (police_x == thief_x - 2 and police_y == thief_y and police_x - 1 == " ")
                    or (police_x == thief_x and police_y == thief_y + 1)
                    or (police_x == thief_x and police_y == thief_y - 1)
                    or (police_x == thief_x and police_y == thief_y)) and not self.is_thief_hiding():
                    reward = 100
                    thief_dir = 4
                    done = True
                    return self.encode(police_x, police_y, thief_x, thief_y, thief_dir), reward, done, {}
                else:
                    reward = -10  # penalty for failed catch


        # Thief action (random)
        thief_action_list = [1, 2, 3, 4]
        while True:
            thief_action = np.random.choice(thief_action_list)
            if thief_action == 1:
                if self.map[thief_y][thief_x + 1] == " ":
                    thief_x = min(thief_x + 2, self.grid_size[1] - 1)
                    thief_dir = 0
                    break
                else:
                    thief_action_list.remove(thief_action)
                    continue
            elif thief_action == 2:
                if self.map[thief_y][thief_x - 1] == " ":
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
        self.state = self.encode(police_x, police_y, thief_x, thief_y, thief_dir)

        self.steps += 1

        return self.state, reward, done, {}

    def is_thief_hiding(self):
        police_x, police_y, thief_x, thief_y, thief_dir = self.decode(self.state)

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

    def render(self, mode='none'):
        grid = [list(row) for row in self.map]

        police_x, police_y, thief_x, thief_y, thief_dir = self.decode(self.state)
        grid[police_y][police_x] = 'P'
        if not self.is_thief_hiding():
            grid[thief_y][thief_x] = '{0}'.format(thief_dir)
        else:
            grid[thief_y][thief_x] = '{0}'.format(thief_dir + 5)  # Hidden state

        if mode == 'dqn':
            return np.array([ord(col) for row in grid for col in row]).reshape(11, 19, 1)
        else:
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
        print("Action: {0}, Reward: {1}, Done: {2}".format(action, reward, done))
