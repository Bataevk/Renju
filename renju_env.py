# pip install numpy icecream gymnasium torch stable-baselines3

'''

Значение total_timesteps зависит от сложности задачи, размера пространства состояний и желаемого качества агента. Для настольных игр вроде рэндзю/гомоку обычно используют диапазон:

Минимум для теста: 10 000 – 100 000 (агент учится делать хоть что-то)
Базовое обучение: 500 000 – 2 000 000 (агент начинает играть осмысленно)
Для сильного агента: 5 000 000 – 20 000 000+ (агент учится стратегиям, но обучение может занять много времени)
Рекомендация:
Начните с 500 000 – 1 000 000. Если агент играет слабо — увеличивайте.
Для финального обучения на мощной видеокарте можно ставить 5–10 миллионов и больше, если есть время.

Важно:

Следите за метриками обучения (reward, winrate).
Можно сохранять чекпоинты и дообучать модель позже.

'''


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
    def __init__(self, observation_space: gym.Space, features_dim: int = 225):
        super().__init__(observation_space, features_dim)
        
        # Вход: (3, 15, 15)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 15, kernel_size=3, padding=1),  # (16, 15, 15)
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1), # (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(15, 15, kernel_size=3, padding=1), # (64, 15, 15)
            # nn.RNN(15, 3, 1),
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

    def game_reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = GameEngine()
        return self._get_masked_obs(), {}

    def reset(self, seed=None, options=None):
        def append_move(x, y, player):
            board[x, y] = player
            self.game.history.append((player, (x, y)))

        super().reset(seed=seed)
        self.game = GameEngine()
        board = self.game.board
        size = self.game.size
        center = size // 2

        # 1. Первый ход: черный (-1) в центр
        append_move(center, center, -1)

        # 2. Второй ход: белый (1) в любую клетку с радиусом 1 от центра (но не центр)
        board_3_3 = board[center - 1:center + 2, center - 1:center + 2]
        # Получаем индексы свободных клеток
        free_indices = np.argwhere(board_3_3 == 0)
        # Если нет свободные клетки, кидаем ошибку
        assert free_indices.size > 0, "Нет свободных клеток в 3x3 вокруг центра"
        idx = np.random.choice(free_indices.shape[0])
        y2, x2 = free_indices[idx]
        y2 += center - 1
        x2 += center - 1
        # Проверяем, что клетка свободна
        assert board[y2, x2] == 0, "Клетка занята"
        append_move(x2, y2, 1)

        # 3. Третий ход: черный (-1) в любую свободную клетку 5x5 вокруг центра (кроме занятых)
        board_5_5 = board[center - 2:center + 3, center - 2:center + 3]
        # Получаем индексы свободных клеток
        free_indices = np.argwhere(board_5_5 == 0)
        # Если нет свободные клетки, кидаем ошибку 
        assert free_indices.size > 0, "Нет свободных клеток в 5x5 вокруг центра"

        idx = np.random.choice(free_indices.shape[0])
        y3, x3 = free_indices[idx]
        y3 += center - 2
        x3 += center - 2

        # Проверяем, что клетка свободна
        assert board[y3, x3] == 0, "Клетка занята"
        append_move(x3, y3, -1)

        # После этого ход белых
        self.game.current_player = 1
        return self._get_masked_obs(), {}

    # def _get_obs(self):
    #     # Создаем тензор с каналами первыми (channels, height, width)
    #     obs = np.zeros((3, self.game.size, self.game.size), dtype=np.float32)
        
    #     for y in range(self.game.size):
    #         for x in range(self.game.size):
    #             if self.game.board[y, x] == self.game.current_player:
    #                 obs[0, y, x] = 1.0  # Текущий игрок
    #             elif self.game.board[y, x] == -self.game.current_player:
    #                 obs[1, y, x] = 1.0  # Противник
                
    #             # Для Black (current_player=-1) меняем каналы местами
    #             if self.game.current_player == -1:
    #                 obs[0, y, x], obs[1, y, x] = obs[1, y, x], obs[0, y, x]
        
    #     # Канал 2: маска доступных ходов (будет заполнен в _get_masked_obs)
    #     return obs

    def _get_obs(self):
        board = self.game.board
        current = self.game.current_player
        # Если current игрок - Белые (1)

        channel0 = (board == (1 * current)).astype(np.float32) # Чёрные = текущий игрок (но обозначения все-равно 1)
        channel1 = (board == (-1 * current)).astype(np.float32) # Белые = противник (для противника всегда -1)

        # Третий канал оставляем нулевым (будет заполнен маской доступных ходов)
        channel2 = np.zeros_like(board, dtype=np.float32)
        obs = np.stack([channel0, channel1, channel2], axis=0)
        return obs


    def _get_masked_obs(self):
        obs = self._get_obs()
        mask = self.game.get_allowed_mask() * float(self.game.current_player)
        obs[2, :, :] = mask  # Добавляем маску в третий канал
        return obs

    def step(self, action):
        # Размеры штрафов
        illegal_move_penalty = -1.0
        move_penalty = -0.001
        lose_penalty = -0.5  # Штраф за проигрыш (если есть возможность проиграть в следующем ходе)
        # Размеры вознаграждений
        win_reward = 2.0
        draw_reward = -0.25


        coords = self.game.get_coordinates(action)
        if coords == 'Ничья':
            return self._get_masked_obs(), draw_reward, True, False, {}
        
        x, y = coords

        if x is None or y is None:
            return self._get_masked_obs(), illegal_move_penalty, False, True, {}

        status = self.game.make_move(x, y)
        obs = self._get_masked_obs()
        reward = move_penalty  # Штраф за ход
        terminated = False
        truncated = False

        if status:
            if self.game.is_win(x, y):
                reward = win_reward
                terminated = True
            else:
                # Проверка на проигрыш следующим ходом
                self.game.current_player = -self.game.current_player
                
                allowed_moves = self.game.get_allowed_moves()
                if len(allowed_moves) == 0:
                    reward = draw_reward
                    terminated = True
                else:
                    for move in allowed_moves:
                        if self.game.is_win(move[0], move[1]):
                            reward = -lose_penalty
                            # Если есть возможность проиграть в следующем ходе, то не завершаем игру, чтобы агент учился
                            # terminated = True
                            break
                
                self.game.current_player = -self.game.current_player
        else:
            reward = illegal_move_penalty
            truncated = True 

        self.game.current_player = -self.game.current_player
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            print(self.game.board)

if __name__ == "__main__":
    ic("cuda" if th.cuda.is_available() else "cpu")
    ic.disable()
    env = RenjuEnv()
    check_env(env)

    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1,
        device=ic("cuda" if th.cuda.is_available() else "cpu"),
        normalize_advantage=False,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,  # Используем кастомный CNN
            features_extractor_kwargs=dict(features_dim=225),
            normalize_images=False  # Отключаем автоматическую нормализацию
        ),
        n_steps=4096  
    )


    model.learn(total_timesteps=4_000_000)

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
