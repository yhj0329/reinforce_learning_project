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

        # Action space: move east, move west, move south, move north, catch
        self.action_space = spaces.Discrete(5)

        # Observation space: police positions, thief1 positions, thief1 direction, thief2 positions, thief2 direction (0: east, 1: west, 2: south, 3: north 4: catched)
        self.observation_space = spaces.Discrete(13286025)

        # Initialize state
        self.reset()

    def encode(self, police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir):
        police_x = (police_x - 1) // 2
        police_y = police_y - 1
        thief1_x = (thief1_x - 1) // 2
        thief1_y = thief1_y - 1
        thief2_x = (thief2_x - 1) // 2
        thief2_y = thief2_y - 1
        i = police_x
        i *= 9
        i += police_y
        i *= 9
        i += thief1_x
        i *= 9
        i += thief1_y
        i *= 9
        i += thief2_x
        i *= 9
        i += thief2_y
        i *= 5
        i += thief1_dir
        i *= 5
        i += thief2_dir
        return i

    def decode(self, i):
        out = []
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i % 9 + 1)
        i = i // 9
        out.append((i % 9 * 2) + 1)
        i = i // 9
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

        # Initialize positions: [police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir]
        police_y, police_x = 1, 1
        thief1_y, thief1_x = random.choice(empty_spaces)  # Random initial thief1 position
        thief1_dir = random.randint(0, 3)  # Random initial thief1 direction (0: east, 1: west, 2: south, 3: north)

        thief2_y, thief2_x = random.choice(empty_spaces)  # Random initial thief2 position
        thief2_dir = random.randint(0, 3)  # Random initial thief1 direction (0: east, 1: west, 2: south, 3: north)

        self.state = self.encode(police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir)

        # Time step
        self.steps = 0

        return self.state

    def step(self, action):
        police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir = self.decode(self.state)

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
            if thief1_dir != 4:
                if ((police_x == thief1_x + 2 and police_y == thief1_y and police_x + 1 == " ")
                    or (police_x == thief1_x - 2 and police_y == thief1_y and police_x - 1 == " ")
                    or (police_x == thief1_x and police_y == thief1_y + 1)
                    or (police_x == thief1_x and police_y == thief1_y - 1)
                    or (police_x == thief1_x and police_y == thief1_y)) and not self.is_thief_hiding():
                        if reward == -1 or reward == -10:
                            reward = 100
                        else:
                            reward += 100
                        thief1_dir = 4
                else:
                    if reward == -1:
                        reward = -10  # penalty for failed catch


            if thief2_dir != 4:
                if ((police_x == thief2_x + 2 and police_y == thief2_y and police_x + 1 == " ")
                        or (police_x == thief2_x - 2 and police_y == thief2_y and police_x - 1 == " ")
                        or (police_x == thief2_x and police_y == thief2_y + 1)
                        or (police_x == thief2_x and police_y == thief2_y - 1)
                        or (police_x == thief2_x and police_y == thief2_y)) and not self.is_thief_hiding():
                        if reward == -1 or reward == -10:
                            reward = 100
                        else:
                            reward += 100
                        thief1_dir = 4
                else:
                    if reward == -1:
                        reward = -10  # penalty for failed catch


            if thief1_dir == 4 and thief2_dir == 4:
                done = True
                return self.encode(police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir), reward, done, {}


        if thief1_dir != 4:
            # thief1 action (random)
            thief1_action_list = [1, 2, 3, 4]
            while True:
                thief1_action = np.random.choice(thief1_action_list)
                if thief1_action == 1:
                    if self.map[thief1_y][thief1_x + 1] == " ":
                        thief1_x = min(thief1_x + 2, self.grid_size[1] - 1)
                        thief1_dir = 0
                        break
                    else:
                        thief1_action_list.remove(thief1_action)
                        continue
                elif thief1_action == 2:
                    if self.map[thief1_y][thief1_x - 1] == " ":
                        thief1_x = max(thief1_x - 2, 0)
                        thief1_dir = 1
                        break
                    else:
                        thief1_action_list.remove(thief1_action)
                        continue
                elif thief1_action == 3:
                    if self.map[thief1_y + 1][thief1_x] == " ":
                        thief1_y = min(thief1_y + 1, self.grid_size[0] - 1)
                        thief1_dir = 2
                        break
                    else:
                        thief1_action_list.remove(thief1_action)
                        continue
                elif thief1_action == 4:
                    if self.map[thief1_y - 1][thief1_x] == " ":
                        thief1_y = max(thief1_y - 1, 0)
                        thief1_dir = 3
                        break
                    else:
                        thief1_action_list.remove(thief1_action)
                        continue

        if thief2_dir != 4:
            # thief2 action (random)
            thief2_action_list = [1, 2, 3, 4]
            while True:
                thief2_action = np.random.choice(thief2_action_list)
                if thief2_action == 1:
                    if self.map[thief2_y][thief2_x + 1] == " ":
                        thief2_x = min(thief2_x + 2, self.grid_size[1] - 1)
                        thief2_dir = 0
                        break
                    else:
                        thief2_action_list.remove(thief2_action)
                        continue
                elif thief2_action == 2:
                    if self.map[thief2_y][thief2_x - 1] == " ":
                        thief2_x = max(thief2_x - 2, 0)
                        thief2_dir = 1
                        break
                    else:
                        thief2_action_list.remove(thief2_action)
                        continue
                elif thief2_action == 3:
                    if self.map[thief2_y + 1][thief2_x] == " ":
                        thief2_y = min(thief2_y + 1, self.grid_size[0] - 1)
                        thief2_dir = 2
                        break
                    else:
                        thief2_action_list.remove(thief2_action)
                        continue
                elif thief2_action == 4:
                    if self.map[thief2_y - 1][thief2_x] == " ":
                        thief2_y = max(thief2_y - 1, 0)
                        thief2_dir = 3
                        break
                    else:
                        thief2_action_list.remove(thief2_action)
                        continue

        # Update state
        self.state = self.encode(police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir)

        self.steps += 1

        return self.state, reward, done, {}

    def is_thief_hiding(self):
        police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir = self.decode(self.state)

        # Check if police is in the line of sight of thief1
        if thief1_dir == 0:  # thief1 looking east
            if police_y == thief1_y and police_x >= thief1_x:
                return True
        elif thief1_dir == 1:  # thief1 looking west
            if police_y == thief1_y and police_x <= thief1_x:
                return True
        elif thief1_dir == 2:  # thief1 looking south
            if police_x == thief1_x and police_y >= thief1_y:
                return True
        elif thief1_dir == 3:  # thief1 looking north
            if police_x == thief1_x and police_y <= thief1_y:
                return True

        # Check if police is in the line of sight of thief1
        if thief2_dir == 0:  # thief1 looking east
            if police_y == thief2_y and police_x >= thief2_x:
                return True
        elif thief2_dir == 1:  # thief1 looking west
            if police_y == thief2_y and police_x <= thief2_x:
                return True
        elif thief2_dir == 2:  # thief1 looking south
            if police_x == thief2_x and police_y >= thief2_y:
                return True
        elif thief2_dir == 3:  # thief1 looking north
            if police_x == thief2_x and police_y <= thief2_y:
                return True

        return False

    def render(self, mode='none'):
        grid = [list(row) for row in self.map]

        police_x, police_y, thief1_x, thief1_y, thief2_x, thief2_y, thief1_dir, thief2_dir = self.decode(self.state)
        grid[police_y][police_x] = 'P'
        if not self.is_thief_hiding():
            grid[thief1_y][thief1_x] = '{0}'.format(thief1_dir)
            grid[thief2_y][thief2_x] = '{0}'.format(thief2_dir)
        else:
            if thief1_dir != 4:
                grid[thief1_y][thief1_x] = '{0}'.format(thief1_dir+5)
            if thief2_dir != 4:
                grid[thief2_y][thief2_x] = '{0}'.format(thief2_dir+5)

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
