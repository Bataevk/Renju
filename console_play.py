import numpy as np
from stable_baselines3 import PPO
from renju_env import RenjuEnv

def print_board(board):
    print("   " + " ".join(f"{i:2}" for i in range(board.shape[1])))
    for y in range(board.shape[0]):
        row = []
        for x in range(board.shape[1]):
            if board[y, x] == -1:
                row.append("X")
            elif board[y, x] == 1:
                row.append("O")
            else:
                row.append(".")
        print(f"{y:2} " + "  ".join(row))

def main():
    env = RenjuEnv()
    model = PPO.load("renju_model.zip", env=env)
    obs, _ = env.reset()
    human_player = int(input("За кого играть? (X = -1, O = 1): ").strip())
    if human_player not in [-1, 1]:
        print("Некорректный выбор. По умолчанию вы играете за X (чёрных).")
        human_player = -1

    done = False
    while not done:
        print_board(env.game.board)
        if env.game.current_player == human_player:
            while True:
                try:
                    move = input("Ваш ход (формат: x y): ")
                    x, y = map(int, move.strip().split())
                    status = env.game.make_move(x, y)
                    if status == "Ход сделан" or "Победа" in status:
                        break
                    else:
                        print("Ошибка:", status)
                except Exception as e:
                    print("Ошибка ввода:", e)
            obs = env._get_obs()
        else:
            action, _ = model.predict(obs, deterministic=True)
            x = action % env.game.size
            y = action // env.game.size
            print(f"Агент ходит: {x} {y}")
            status = env.game.make_move(x, y)
            print("Статус:", status)
            obs = env._get_obs()
        if "Победа" in status:
            print_board(env.game.board)
            print(status)
            done = True
        elif status == "Недопустимый ход":
            print("Недопустимый ход. Игра завершена.")
            done = True

if __name__ == "__main__":
    main()