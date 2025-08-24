import pygame 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from game_agent import BlackjackAgent
from tqdm import tqdm
from matplotlib import pyplot as plt


# add action masking 
# - make a function to obtain allowed moves and return a list of those allowed moves
# - modify action space to only come from that list of allowed moves 
# - Make function to terminate when no moves remain
#


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
        self.wrong_move_counter = 10
        
        

        self.render_mode = render_mode

        if fps is not None:
            self.metadata["render_fps"] = fps


        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.int8)

        
        # row (0-6)
        # column (0-6)
        # direction (0-3) up, down, left, right
        self.action_space = spaces.MultiDiscrete([7,7,4])
        


    
        self.movement_vectors = {
            0: np.array([0,-2]),
            1: np.array([0,2]),
            2: np.array([-2,0]),
            3: np.array([2,0])
        }
        
        self.reset()

    def _get_obs(self):
        return {"board":self.board}
    
    def _get_info(self):
        return {
            "remaining marbles":(np.sum(self.board == 1))
        }

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


        
    def check_movement_validity(self, selected_marble:np.array, new_position:np.array, midpoint:np.array):

        # 1. Check that selected token is a token and it is not empty or a wall
        if (self.board[selected_marble[1]][selected_marble[0]] != 1):
            return False
        
        # 2. Check that the direction that the new position is in bounds
        if (new_position[0]<0) or (new_position[0] >= self.size) or (new_position[1]<0) or (new_position[1] >= self.size):
            return False

        # 3. Check that there is a marble in the midpoint
        if self.board[selected_marble[1]][selected_marble[0]] != 1:
            return False

        # 4. Check that the new position is not a wall or blocked by another piece
        if (self.board[new_position[1]][new_position[0]] != 1):
            return False

        return True
            


    def step(self, step_input):


        """
        MARBLE GAME STEP FUNCTION
        1. User inputs coordinates for the selected piece and the action Enum that it is going to take (x,y,direction)
        2. Direction object is converted to an np vector
        3. Check to see if movement is valid
            if it is +1 points if it isn't -0.1 points
        4. Update Board and current position
        5. Check to see if only one piece remains in the board np.sum and terminate if so,
        6. Screen is rendered 
        """





        direction = int(step_input[2])
        selected_marble = np.array([step_input[0], step_input[1]])



        


        new_position = np.add(selected_marble, self.movement_vectors[direction])
        midpoint = np.add(selected_marble, (self.movement_vectors[direction] // 2).astype(int))


        if self.check_movement_validity(selected_marble=selected_marble, new_position=new_position, midpoint=midpoint):
            self.board[midpoint[1]][midpoint[0]] = 0
            self.board[new_position[1]][new_position[0]] = 1
            self.board[selected_marble[1]][selected_marble[0]] = 0
            reward = 1.0
            self.wrong_move_counter = 0

        else:

             
            self.wrong_move_counter += 1
            reward = -0.5
            #print("WRONG MOVE")

        
        observation = self._get_obs()
        info = self._get_info()
        terminated = True if (info["remaining marbles"] == 1 or self.wrong_move_counter > 5) else False
        


        self.render()


        return observation, reward, terminated, False, info

    def render(self):
        import pygame
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = self.create_board(self.size)   

        observation = self._get_obs()
        info = self._get_info()
        reward = 0
        terminated = False


        return observation, info

        
if __name__=="__main__":
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 100_00        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration


    # Create environment and agent
    env = gym.make("MarbleGameEnv", render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
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



    env.close()

    def get_moving_avgs(arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
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




