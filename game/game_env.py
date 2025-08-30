import pygame 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from game_agent import MarbleGameAgent
from tqdm import tqdm
from matplotlib import pyplot as plt



 


register(
    id="MarbleGameEnv",
    entry_point="game_env:MarbleGameEnv",
)


class MarbleGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    def __init__(self, size=7, render_mode=None, fps=None):

        self.size = size 
        self.board = None 
        
        self.cell_size = 64 
        self.margin = 24
        self.window = None
        self.clock = None
        self._canvas = None
        self.render_mode = render_mode
        if fps is not None:
            self.metadata["render_fps"] = fps
        
        self.wrong_move_counter = 0
        self.max_wrong_moves = 10


        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.int8)
        self.action_space = spaces.MultiDiscrete([7,7,4])
        
        self.movement_vectors = {
            0: np.array([0,-2]),
            1: np.array([0,2]),
            2: np.array([-2,0]),
            3: np.array([2,0])
        }
        

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = self.create_board(self.size)   

        self.wrong_move_counter = 0
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def _get_obs(self):
        return {"board":self.board.copy()}
    
    def _get_info(self):
        valid_actions = self.get_valid_moves()
        mask = np.zeros(self.size * self.size * 4, dtype=np.int8)
        for a in valid_actions:
            mask[self.action_to_index(a)] = 1

        
        return {
            "remaining marbles":(np.sum(self.board == 1)),
            "action_mask":mask,
            "valid_actions":valid_actions
        }



    def create_board(self, size=7):
        board = np.full((size,size),1, dtype=np.int8)
        
        board[size//2][size//2] = 0

        for row in range(size):
            for column in range(size):
                if (row < size // 3 or row >= size - size//3) and (column < size //3 or column >= size - size // 3):
                    board[row, column] = 2
        return board



    def check_movement_validity(self, selected_marble:np.array, new_position:np.array, midpoint:np.array):

        
        for position in (selected_marble, new_position, midpoint):
            if position[0] < 0 or position[0] >= self.size or position[1] < 0 or position[1] >= self.size:
                return False

        if (self.board[selected_marble[1]][selected_marble[0]] != 1):
            return False

        if self.board[midpoint[1]][midpoint[0]] != 1:
            return False

        if (self.board[new_position[1]][new_position[0]] != 0):
            return False

        return True

    def get_valid_moves(self):
        
        valid_moves = []

        for y in range(self.size):
            for x in range(self.size):
                if self.board[y, x] != 1:
                    continue
                for direction_index, vec in self.movement_vectors.items():
                    new_position = np.array([y + vec[0], x + vec[1]])
                    midpoint = np.array([y + vec[0] // 2, x + vec[1] // 2])
                    selected_marble = np.array([y,x])
                    if self.check_movement_validity(selected_marble, new_position, midpoint):
                        valid_moves.append((int(y), int(x), int(direction_index)))
        return valid_moves
    
    
    def action_to_index(self, action):
        y, x, direction_index = action
        return int((y * self.size * 4) + (x * 4) + direction_index)
    
    def index_to_action(self, index:int):
        y = index // (self.size * 4)
        x = (index % (self.size * 4)) // 4
        direction_index = index % 4

        return (int(y),int(x),int(direction_index))


    def step(self, step_input):
        
        direction = int(step_input[2])
        selected_marble = np.array([int(step_input[0]), int(step_input[1])])
        vec = self.movement_vectors[direction]
        new_position = selected_marble + vec
        midpoint = selected_marble + (vec // 2).astype(int)



        reward = 0.0

        if self.check_movement_validity(selected_marble, new_position, midpoint):


            self.board[midpoint[0], midpoint[1]] = 0
            self.board[new_position[0], new_position[1]] = 1
            self.board[selected_marble[0], selected_marble[1]] = 0

            self.wrong_move_counter = 0

            reward += 1.0

        else:
            self.wrong_move_counter += 1
            reward -= 0.5

        observation = self._get_obs()
        info = self._get_info()

        remaining = info["remaining marbles"]

        if remaining == 1:
            terminated = True
            reward += 50
        elif self.wrong_move_counter > self.max_wrong_moves:
            terminated = True
            reward -= float(len(np.where(self.board == 1))) * 0.1
        else: 

            if len(self.get_valid_moves()) == 0:
                terminated = True
                if remaining > 1:
                    reward -= float(len(np.where(self.board == 1))) * 0.1
                else: 
                    reward += 20.0
            else:
                terminated = False

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), bool(terminated), False, info





    def render(self):

        if self.board is None:
            return None

        rows, cols = self.board.shape
        width  = self.margin * 2 + cols * self.cell_size
        height = self.margin * 2 + rows * self.cell_size

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Marble Solitaire")
                self.clock = pygame.time.Clock()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            if self._canvas is None:
                self._canvas = pygame.Surface((width, height))
            surface = self._canvas

        else:
            if self._canvas is None:
                pygame.init()
                self._canvas = pygame.Surface((width, height))
            surface = self._canvas

        BG = (40, 40, 45)
        WOOD = (210, 180, 140)
        BLOCKED = (60, 60, 65)
        HOLE_RIM = (160, 120, 80)
        HOLE_INNER = (230, 210, 170)
        MARBLE = (30, 70, 180)
        MARBLE_HL = (220, 240, 255)
        MARBLE_SH = (10, 10, 15)
        GRID = (120, 100, 80)

        surface.fill(BG)

        board_rect = pygame.Rect(self.margin, self.margin, cols * self.cell_size, rows * self.cell_size)
        pygame.draw.rect(surface, WOOD, board_rect, border_radius=16)

        for r in range(rows + 1):
            y = self.margin + r * self.cell_size
            pygame.draw.line(surface, GRID, (self.margin, y), (self.margin + cols * self.cell_size, y), 1)
        for c in range(cols + 1):
            x = self.margin + c * self.cell_size
            pygame.draw.line(surface, GRID, (x, self.margin), (x, self.margin + rows * self.cell_size), 1)

        
        for r in range(rows):
            for c in range(cols):
                val = int(self.board[r, c])
                cell_x = self.margin + c * self.cell_size
                cell_y = self.margin + r * self.cell_size
                cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)

                if val == 2:  
                    pygame.draw.rect(surface, BLOCKED, cell_rect)
                    continue

                
                cx = cell_x + self.cell_size // 2
                cy = cell_y + self.cell_size // 2
                hole_r_outer = int(self.cell_size * 0.42)
                hole_r_inner = int(self.cell_size * 0.34)
                pygame.draw.circle(surface, HOLE_RIM, (cx, cy), hole_r_outer)
                pygame.draw.circle(surface, HOLE_INNER, (cx, cy), hole_r_inner)

                if val == 1: 
                    r_marble = int(self.cell_size * 0.30)
                    pygame.draw.circle(surface, MARBLE_SH, (cx + r_marble // 6, cy + r_marble // 6), r_marble)
                    pygame.draw.circle(surface, MARBLE, (cx, cy), r_marble)
                    pygame.draw.circle(surface, MARBLE_HL, (cx - r_marble // 3, cy - r_marble // 3), max(2, r_marble // 5))

        pygame.draw.rect(surface, GRID, board_rect, width=2, border_radius=16)

        if self.render_mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata.get("render_fps", 30))
            return None

        elif self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(surface) 
            return np.transpose(arr, (1, 0, 2))
        else:
            return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        if self.clock is not None:
            self.clock = None



if __name__=="__main__":
    learning_rate = 0.01        
    n_episodes = 1_000_000        
    start_epsilon = 1.0         
    epsilon_decay = start_epsilon / (n_episodes / 2)  
    final_epsilon = 0.1         





    env = gym.make("MarbleGameEnv")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = MarbleGameAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )


    counter = 0
    for episode in tqdm(range(n_episodes)):

        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)

            next_obs, reward, terminated, _, info = env.step(action)

            agent.update(obs=obs, action=action, reward=reward, terminated=terminated, next_obs=next_obs)


            done = terminated
            obs = next_obs

        agent.decay_epsilon()
        counter += 1


    agent.save("MarbleGameAgent.pkl")

    env.close()

    """
    # TEST CODE
    env = gym.make("MarbleGameEnv", render_mode="human")  # headless test
    agent = MarbleGameAgent.load("MarbleGameAgent.pkl", env)

    agent.test(env, n_episodes=1_000, render=False)
    """



    def get_moving_avgs(arr, window, convolution_mode):
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()        



