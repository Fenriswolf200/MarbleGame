from collections import defaultdict
import gymnasium as gym
import numpy as np
 # hello world
print("test")





class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env

        # MultiDiscrete([7,7,4]) -> nvec = [7,7,4]

        self.nvec = np.array(getattr(env.action_space, "nvec"))
        self.num_actions = int(np.prod(self.nvec))

        # Q(s,a) table: state_key -> np.array([Q for each flat action])
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    # ---- Helpers ----
    def encode_obs(self, obs: dict) -> bytes:
        """Make observation hashable. Expect obs = {'board': np.ndarray}."""
        board = obs["board"]
        # Ensure C-contiguous for stable tobytes
        board = np.ascontiguousarray(board)
        return board.tobytes()

    def flatten_action(self, a3) -> int:
        a3 = np.asarray(a3, dtype=int)
        # Use positional arg for the shape
        return int(np.ravel_multi_index(a3, tuple(self.nvec)))

    def unflatten_action(self, idx: int):
        # Use positional arg (or shape=...) — NOT dims=
        return tuple(np.unravel_index(int(idx), tuple(self.nvec)))
        # equivalently:
        # return tuple(np.unravel_index(int(idx), shape=tuple(self.nvec)))


    # ---- Policy ----
    def get_action(self, obs: dict):
        s = self.encode_obs(obs)

        if np.random.random() < self.epsilon:
            # Random legal action in env's expected format (length-3 array/tuple)
            return self.env.action_space.sample()
        else:
            # Greedy flat index -> convert back to (r,c,d)
            best_flat = int(np.argmax(self.q_values[s]))
            return np.array(self.unflatten_action(best_flat), dtype=int)

    # ---- Learning ----
    def update(
        self,
        obs: dict,
        action,               # array-like of shape (3,) from env.step
        reward: float,
        terminated: bool,
        next_obs: dict,
    ):
        s = self.encode_obs(obs)
        s_next = self.encode_obs(next_obs)
        a_flat = self.flatten_action(action)

        # Max over next-state actions unless terminated
        future_q = 0.0 if terminated else float(np.max(self.q_values[s_next]))
        target = reward + self.discount_factor * future_q

        td = target - self.q_values[s][a_flat]
        self.q_values[s][a_flat] += self.lr * td
        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


