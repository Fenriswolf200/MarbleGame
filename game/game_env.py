import pygame 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from game_agent import MarbleGameAgent
from tqdm import tqdm
from matplotlib import pyplot as plt

# 1. Clean up code
# 2. Add action masking 
# - write function to get all valid moves and return a list with their np coordinates
# - encode mask for your action space
# - provide mask in reset and step()
# - modify get_action to use mask
# - Make function to terminate when no moves remain
# 3. Find a better reward system for the agent so that it learns faster
# 4. Add logs to improve debugging


 
"""



board =
[
[2,2,1,1,1,2,2]
[2,2,1,1,1,2,2]
[1,1,1,1,1,1,1]
[1,1,1,0,1,1,1]
[1,1,1,1,1,1,1]
[2,2,1,1,1,2,2]
[2,2,1,1,1,2,2]
]
"""


register(
    id="MarbleGameEnv",
    entry_point="game_env:MarbleGameEnv",
)


class MarbleGameEnv(gym.Env):


    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    # --------------Initialization-------------------
    # The init function creates the variables that will be used across the 
    # program and sets them to their default state.
    def __init__(self, size=7, render_mode=None, fps=None):

        # These two variables control the creation and management of 
        # the board.        
        self.size = size # Size of the board array. 
        self.board = None # Will store a numpy 2D array that contains the "Board".
        
        # These variables control the rendering of the game in pygame.
        self.cell_size = 64 
        self.margin = 24
        self.window = None
        self.clock = None
        self._canvas = None
        self.render_mode = render_mode
        if fps is not None:
            self.metadata["render_fps"] = fps
        
        # These variables will control the amount of wrong moves that the model will 
        # be allowed to make consecutively before it is terminated.
        self.wrong_move_counter = 0
        self.max_wrong_moves = 10

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.int8)

        

        # An action space is an object that tells the agent what actions it has available. 
        # row (0-6)
        # column (0-6)
        # direction (0-3) up, down, left, right
        self.action_space = spaces.MultiDiscrete([7,7,4])
        


        # This dictionary stores the movement vectors for each of the directions that a marble can move.
        self.movement_vectors = {
            0: np.array([0,-2]),
            1: np.array([0,2]),
            2: np.array([-2,0]),
            3: np.array([2,0])
        }
        

        # This function will initialize all the variables to their default values
        self.reset()

    # The reset function will set all of the variables in the environment 
    # to their default state and it will return the info on the environment.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = self.create_board(self.size)   

        observation = self._get_obs()
        info = self._get_info()


        return observation, info
    

    # --------------Analytics/Logs-------------------
    # _get_obs and _get_info are functions that will return info about the environment
    # _get_obs is used by the agent to view the environment and make it's decisions from there.
    # _get_info is used for debugging
    def _get_obs(self):
        return {"board":self.board}
    
    def _get_info(self):
        valid_actions = self.get_valid_moves()
        mask = np.zeros(self.size * self.size * 4, dtype=np.int8)
        for a in valid_actions:
            mask[self.action_to_index(a)] = 1

        
        return {
            "remaining marbles":(np.sum(self.board == 1)),
            "action_mask":mask
        }


    # --------------Board----------------------------
    # This function will create a 2D array that will act as the board for the game.
    # - The size of the board will be defined by the parameter size with the default being 7x7
    # - Each tile will have a value in the range of [0,2]
    #   - 0 means that the tile is empty.
    #   - 1 means that the tile has a marble in it.
    #   - 2 means that the tile is blocked and not playable
    # - When creating the board the function will set every tile in the array to 1. 
    #   Then it will set the center tile to 0. Finally it will block all of the 
    #   tiles on the edges by setting them to 2
    # - Once all of this is done the function will return the board to be used in the game

    def create_board(self, size=7):
        board = np.full((size,size),1, dtype=np.int8)
        
        board[size//2][size//2] = 0

        row = 0
        col = 0
        for i in board:
            if row < size//3 or row > (size-size//3-1):
                for ii in i:
                    if col < size//3 or col > (size-size//3-1):
                        board[row][col] = 2
                    col += 1
            row += 1
            col = 0

        return board



    # --------------Movement-------------------------
    # The check_movement_validity, as the name implies, will run tests to see if a movement is valid. 
    def check_movement_validity(self, selected_marble:np.array, new_position:np.array, midpoint:np.array):

        # 1. Check that selected token is a token and it is not empty or a wall
        if (self.board[selected_marble[1]][selected_marble[0]] != 1):
            return False
        
        # 2. Check that the new position is inside the bounds of the board.
        if (new_position[0]<0) or (new_position[0] >= self.size) or (new_position[1]<0) or (new_position[1] >= self.size):
            return False

        # 3. Check that there is a marble in the midpoint
        if self.board[midpoint[1]][midpoint[0]] != 1:
            return False

        # 4. Check that the new position is not a wall or blocked by another piece
        if (self.board[new_position[1]][new_position[0]] != 0):
            return False

        return True


    def get_valid_moves(self):
        valid_moves = []
        
        for y in range(self.size):
            for x in range(self.size):
                for direction_index in self.movement_vectors:
                    selected_marble = np.array([y,x])
                    new_position = np.add(selected_marble, self.movement_vectors[direction_index])
                    midpoint = np.add(selected_marble, (self.movement_vectors[direction_index]//2).astype(int))
                    if self.check_movement_validity(selected_marble, new_position, midpoint):
                        valid_moves.append((y,x,direction_index))
        return valid_moves
    
    def action_to_index(self, action):
        y, x, direction_index = action
        return (y * self.size * 4) + (x * 4) + direction_index
    
    def index_to_action(self, index:int):
        y = index // (self.size * 4)
        x = (index % (self.size * 4)) // 4
        direction_index = index % 4

        return (y,x,direction_index)

    


    # MARBLE GAME STEP FUNCTION
    # 1. User inputs coordinates for the selected piece and the action Enum that it is going to take (x,y,direction)
    # 2. Direction object is converted to an np vector
    # 3. Check to see if movement is valid
    #     if it is +1 points if it isn't -0.1 points
    # 4. Update Board and current position
    # 5. Check to see if only one piece remains in the board np.sum and terminate if so,
    # 6. Screen is rendered 
    def step(self, step_input):
        
        direction = int(step_input[2])
        selected_marble = np.array([step_input[0], step_input[1]])

        new_position = np.add(selected_marble, self.movement_vectors[direction])
        midpoint = np.add(selected_marble, (self.movement_vectors[direction] // 2).astype(int))

        reward = 0


        if self.check_movement_validity(selected_marble=selected_marble, new_position=new_position, midpoint=midpoint):
            self.board[midpoint[1]][midpoint[0]] = 0
            self.board[new_position[1]][new_position[0]] = 1
            self.board[selected_marble[1]][selected_marble[0]] = 0
            reward += self.max_wrong_moves
            self.wrong_move_counter = 0
        else:
            self.wrong_move_counter += 1
            reward -= 0.75
        

        observation = self._get_obs()
        info = self._get_info()

        if info["remaining marbles"] == 1:
            terminated = True
            reward += 100.0
        elif self.wrong_move_counter > self.max_wrong_moves:
            terminated = True
            reward -= 1.0

        else:
            terminated = False



        if self.render_mode == "human":
            self.render()


        return observation, reward, terminated, False, info



    # --------------Rendering-----------------------

    def render(self):

        if self.board is None:
            return None

        rows, cols = self.board.shape
        width  = self.margin * 2 + cols * self.cell_size
        height = self.margin * 2 + rows * self.cell_size

        # Choose the drawing surface
        if self.render_mode == "human":
            # Lazy-init window and clock
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Marble Solitaire")
                self.clock = pygame.time.Clock()
            # Keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            # Ensure a canvas exists to draw on
            if self._canvas is None:
                self._canvas = pygame.Surface((width, height))
            surface = self._canvas

        else:
            # Headless draw to an off-screen surface (rgb_array mode or None)
            if self._canvas is None:
                pygame.init()
                self._canvas = pygame.Surface((width, height))
            surface = self._canvas

        # ---------- Drawing ----------
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

        # Optional grid
        for r in range(rows + 1):
            y = self.margin + r * self.cell_size
            pygame.draw.line(surface, GRID, (self.margin, y), (self.margin + cols * self.cell_size, y), 1)
        for c in range(cols + 1):
            x = self.margin + c * self.cell_size
            pygame.draw.line(surface, GRID, (x, self.margin), (x, self.margin + rows * self.cell_size), 1)

        # Cells
        for r in range(rows):
            for c in range(cols):
                val = int(self.board[r, c])
                cell_x = self.margin + c * self.cell_size
                cell_y = self.margin + r * self.cell_size
                cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)

                if val == 2:  # blocked
                    pygame.draw.rect(surface, BLOCKED, cell_rect)
                    continue

                # hole
                cx = cell_x + self.cell_size // 2
                cy = cell_y + self.cell_size // 2
                hole_r_outer = int(self.cell_size * 0.42)
                hole_r_inner = int(self.cell_size * 0.34)
                pygame.draw.circle(surface, HOLE_RIM, (cx, cy), hole_r_outer)
                pygame.draw.circle(surface, HOLE_INNER, (cx, cy), hole_r_inner)

                if val == 1:  # marble
                    r_marble = int(self.cell_size * 0.30)
                    pygame.draw.circle(surface, MARBLE_SH, (cx + r_marble // 6, cy + r_marble // 6), r_marble)
                    pygame.draw.circle(surface, MARBLE, (cx, cy), r_marble)
                    pygame.draw.circle(surface, MARBLE_HL, (cx - r_marble // 3, cy - r_marble // 3), max(2, r_marble // 5))

        pygame.draw.rect(surface, GRID, board_rect, width=2, border_radius=16)
        # ---------- /Drawing ----------

        if self.render_mode == "human":
            # Blit the canvas to the window
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata.get("render_fps", 30))
            return None

        elif self.render_mode == "rgb_array":
            # Return an HxWxC uint8 array
            arr = pygame.surfarray.array3d(surface)  # (W, H, 3)
            return np.transpose(arr, (1, 0, 2))
        else:
            return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        if self.clock is not None:
            self.clock = None
        # Don't call pygame.quit() globally if other envs/windows might exist






if __name__=="__main__":
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 1_000_000        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration



    """

    # CODE TO TRAIN WITHOUT RENDERING

    # Create environment and agent
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
        #print(counter)


    agent.save("MarbleGameAgent.pkl")

    env.close()





    env = gym.make("MarbleGameEnv", render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = MarbleGameAgent.load("MarbleGameAgent.pkl", env)


    
    counter = 0
    for episode in tqdm(range(100)):

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
        #print(counter)




    env.close()

    

    """

    env = gym.make("MarbleGameEnv", render_mode="human")  # headless test
    agent = MarbleGameAgent.load("MarbleGameAgent.pkl", env)

    agent.test(env, n_episodes=1_000, render=False)


    



    """




    def get_moving_avgs(arr, window, convolution_mode):
        #Compute moving average to smooth noisy data.
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
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



    """
