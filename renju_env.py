import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from renju_engine import GameEngine, Point

from icecream import ic

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Вход: (3, 15, 15)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (16, 15, 15)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (64, 15, 15)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Вычисляем размер после Flatten
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs))

class RenjuEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        self.game = GameEngine()
        self.action_space = gym.spaces.Discrete(self.game.size * self.game.size)
        # Используем channel-first формат (channels, height, width)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(3, self.game.size, self.game.size),  # (channels, height, width)
            dtype=np.float32
        )
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = GameEngine()
        return self._get_masked_obs(), {}

    def _get_obs(self):
        # Создаем тензор с каналами первыми (channels, height, width)
        obs = np.zeros((3, self.game.size, self.game.size), dtype=np.float32)
        
        for y in range(self.game.size):
            for x in range(self.game.size):
                if self.game.board[y, x] == self.game.current_player:
                    obs[0, y, x] = 1.0  # Текущий игрок
                elif self.game.board[y, x] == -self.game.current_player:
                    obs[1, y, x] = 1.0  # Противник
                
                # Для Black (current_player=-1) меняем каналы местами
                if self.game.current_player == -1:
                    obs[0, y, x], obs[1, y, x] = obs[1, y, x], obs[0, y, x]
        
        # Канал 2: маска доступных ходов (будет заполнен в _get_masked_obs)
        return obs

    def _get_masked_obs(self):
        obs = self._get_obs()
        mask = self.game.get_allowed_mask() * float(self.game.current_player)
        obs[2, :, :] = mask  # Добавляем маску в третий канал
        return obs

    def step(self, action):
        coords = self.game.get_coordinates(action)
        if coords == 'Ничья':
            return self._get_masked_obs(), -0.5, True, False, {}
        
        x, y = coords

        if x is None or y is None:
            reward = -1000
            return self._get_masked_obs(), reward, True, False, {}

        status = self.game.make_move(x, y)
        obs = self._get_masked_obs()
        reward = 0.0
        terminated = False
        truncated = False

        if status:
            if self.game.is_win(x, y):
                reward = 10.0
                terminated = True
            else:
                reward = -0.1
                
                # Проверка на проигрыш следующим ходом
                self.game.current_player = -self.game.current_player
                
                allowed_moves = self.game.get_allowed_moves()
                if len(allowed_moves) == 0:
                    reward = -0.5
                    terminated = True
                else:
                    for move in allowed_moves:
                        if self.game.is_win(move[0], move[1]):
                            reward = -10.0
                            break
                
                self.game.current_player = -self.game.current_player
        else:
            reward = -1000.0 

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            print(self.game.board)

if __name__ == "__main__":
    ic.disable()
    env = RenjuEnv()
    check_env(env)

    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        normalize_advantage=False,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,  # Используем кастомный CNN
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False  # Отключаем автоматическую нормализацию
        )
    )

    model.learn(total_timesteps=100_000)

    obs, _ = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    
    print("Обучение завершено!")
    model.save("renju_model")
    print("Модель сохранена!")
