import numpy as np
from stable_baselines3 import PPO
from renju_env import RenjuEnv
from icecream import ic

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
    obs, _ = env.game_reset()
    human_player = int(input("За кого играть? (X = -1, O = 1): ").strip())
    if human_player not in [-1, 1]:
        print("Некорректный выбор. По умолчанию вы играете за X (чёрных).")
        human_player = -1

    print("Вы играете за", "X" if human_player == -1 else "O")
    x, y = None, None
    done = False
    while not done:
        print_board(env.game.board)
        if env.game.current_player == human_player:
            while True:
                try:
                    move = input("Ваш ход (формат: x y): ")
                    x, y = map(int, move.strip().split())
                    status = env.game.make_move(x, y)
                    if status:
                        break
                    else:
                        print("Недопустимый ход.", env.game.logs[-1], " Попробуйте снова.", sep="\n")
                except Exception as e:
                    print("Ошибка ввода:", e)
            obs = env._get_obs()
        else:
            match len(env.game.history):
                case 0:
                    print("Агент делает первый ход.")
                    env.game.step_1()
                    status = True
                case 1:
                    print("Агент делает второй ход.")
                    env.game.step_2()
                    status = True
                case 2:
                    print("Агент делает третий ход.")
                    env.game.step_3()
                    status = True
                case _:
                    print("Агент делает ход.")
                    action, _ = model.predict(obs, deterministic=True)
                    x = action % env.game.size
                    y = action // env.game.size
                    print(f"Агент ходит: {x} {y}")
                    status = env.game.make_move(x, y)
                    if not status:
                        print("Агент выполнил недопустимый ход.", env.game.logs[-1], "Выполнение случайного хода.", sep="\n")
                        # небольшая корректировка входа для следующего хода
                        coord = env.game.random_step()
                        assert coord != None, "Ошибка: координаты не найдены."
                        x, y = coord
                        print(f"Агент ходит: {x} {y}")
                        status = env.game.make_move(x, y)
                    
                    obs = env._get_masked_obs()
        if status:
            # Проверка на победу и существование x и y
            if x is None or y is None:
                pass
            elif env.game.is_draw():
                print_board(env.game.board)
                print("Ничья!")
                done = True
            elif env.game.is_win(x, y):
                print_board(env.game.board)
                print("Победа!")
                done = True
        else:
            print("Недопустимый ход.")
            done = True
        
        env.game.current_player *= -1

if __name__ == "__main__":
    # ic.disable()
    main()