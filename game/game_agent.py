from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle





class MarbleGameAgent:
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


        self.nvec = np.array(getattr(env.action_space, "nvec"))
        self.num_actions = int(np.prod(self.nvec))

        self.q_values = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float32))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def encode_obs(self, obs: dict) -> bytes:
        """Make observation hashable. Expect obs = {'board': np.ndarray}."""
        board = obs["board"]
        board = np.ascontiguousarray(board)
        return board.tobytes()

    def flatten_action(self, a3) -> int:
        a3 = np.asarray(a3, dtype=int)
        return int(np.ravel_multi_index(a3, tuple(self.nvec)))

    def unflatten_action(self, idx: int):
        return tuple(np.unravel_index(int(idx), tuple(self.nvec)))


    def get_action(self, obs: dict, info: dict = None):

        s = self.encode_obs(obs)

        if info is not None and "action_mask" in info:
            mask = info["action_mask"]
            valid_indices = np.where(mask == 1)[0]

            if len(valid_indices) == 0:
                valid_indices = np.arange(self.num_actions)
        else:
            valid_indices = np.arange(self.num_actions)


        if np.random.random() < self.epsilon:
            flat_action = np.random.choice(valid_indices)
        else:
            q_vals = self.q_values[s]
            flat_action = valid_indices[np.argmax(q_vals[valid_indices])]

        return np.array(self.unflatten_action(int(flat_action)), dtype=int)


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

        future_q = 0.0 if terminated else float(np.max(self.q_values[s_next]))
        target = reward + self.discount_factor * future_q

        td = target - self.q_values[s][a_flat]
        self.q_values[s][a_flat] += self.lr * td
        self.training_error.append(float(td))


    def save(self, path: str):
        """Save agent parameters and Q-table to disk."""
        data = {
            "q_values": dict(self.q_values),   # convert defaultdict -> normal dict
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            "nvec": self.nvec,
            "num_actions": self.num_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, env: gym.Env):
        """Load agent from disk and attach to env."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            env=env,
            learning_rate=data["lr"],
            initial_epsilon=data["epsilon"], 
            epsilon_decay=data["epsilon_decay"],
            final_epsilon=data["final_epsilon"],
            discount_factor=data["discount_factor"],
        )

        agent.q_values = defaultdict(
            lambda: np.zeros(agent.num_actions, dtype=np.float32),
            data["q_values"],
        )

        return agent


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def test(self, env, n_episodes=100, render=False):
        rewards = []
        successes = 0

        for ep in range(n_episodes):
            print(ep)
            obs, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Always greedy (epsilon = 0 in test mode)
                action = self.get_action(obs, info)

                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                obs = next_obs

                if render:
                    env.render()

                # 🚨 Ensure we break immediately when episode ends
                done = terminated or truncated
                if done:
                    break

            rewards.append(total_reward)

            # success condition (custom to your env)
            if "remaining marbles" in info and info["remaining marbles"] == 1:
                successes += 1

        avg_reward = np.mean(rewards)
        success_rate = successes / n_episodes

        print(f"✅ Test Results: {n_episodes} episodes")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Success Rate:   {success_rate*100:.1f}%")

        return avg_reward, success_rate

