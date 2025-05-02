import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from renju_engine import GameEngine

class RenjuEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        self.game = GameEngine()
        self.action_space = gym.spaces.Discrete(self.game.size * self.game.size)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.game.size, self.game.size, 3), dtype=np.float32)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = GameEngine()
        return self._get_obs(), {}

    def _get_obs(self):
        # Создаем тензор с каналами в последнем измерении
        # Kanal 1: stones of the current player
        # Kanal 2: stones of the opponent
        # Kanal 3: 1 if it's the current player's turn, 0 otherwise (not used in this implementation)
        obs = np.zeros((self.game.size, self.game.size, 3), dtype=np.float32)
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y, x] == self.game.current_player:
                    obs[y, x, 0] = 1.0
                elif self.game.board[y, x] == -self.game.current_player:
                    obs[y, x, 1] = 1.0
                if self.game.current_player == -1: # Swap for the neural network to always see the board from the perspective of player -1 (Black)
                    obs[y, x, 0], obs[y, x, 1] = obs[y, x, 1], obs[y, x, 0]
        obs[:, :, 2] = 1.0 if self.game.current_player == -1 else 0.0 # Not necessary as we are swapping the first two channels based on current player
        return obs

    def step(self, action):
        x = action % self.game.size
        y = action // self.game.size
        status = self.game.make_move(x, y)
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False

        if "Победа" in status:
            if "Черных" in status:
                reward = 10.0 if self.game.current_player == 1 else -10.0 # Reward for winning
            else:
                reward = 10.0 if self.game.current_player == -1 else -10.0 # Reward for winning
            terminated = True
        elif status != "Ход сделан":
            reward = -1000.0 # Penalty for invalid move
            terminated = True
        else:
            reward = -0.01 # No reward for a valid move

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            print(self.game.board)

if __name__ == "__main__":
    # Проверка среды
    env = RenjuEnv()
    check_env(env)

    # Создание модели с использованием MLP политики
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        normalize_advantage=False
    )

    # Обучение модели
    model.learn(total_timesteps=400_000)

    # Тестирование модели
    obs, _ = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    
    print("Обучение завершено!")
    
    # Сохранение модели
    model.save("renju_model")
    print("Модель сохранена!")

