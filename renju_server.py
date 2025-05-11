# renju_server.py
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Предполагается, что renju_env.py находится в той же директории
# и содержит RenjuEnv, который в свою очередь содержит RenjuGame.
from renju_env import RenjuEnv 
from stable_baselines3 import PPO
import os

# --- Инициализация Flask приложения ---
app = Flask(__name__)
CORS(app) # Разрешаем CORS для всех доменов (для разработки)

# --- Глобальный контекст игры ---
# В реальном многопользовательском приложении состояние нужно управлять иначе (например, сессии)
# Для одного пользователя или демонстрации глобальные переменные допустимы.
game_context = {
    "env": None,
    "model": None,
    "human_player_value": None, # -1 для черных (X), 1 для белых (O)
    "ai_player_value": None,
    "board_size": 15 # Размер доски по умолчанию
}

# --- Вспомогательные функции ---
def player_value_to_string(player_value):
    if player_value == -1:
        return "black"
    elif player_value == 1:
        return "white"
    return None

def string_to_player_value(player_string):
    if player_string == "black":
        return -1
    elif player_string == "white":
        return 1
    return None

def initialize_game_engine():
    """Загружает среду и модель AI."""
    if game_context["env"] is None:
        try:
            # Убедитесь, что RenjuEnv инициализируется правильно
            game_context["env"] = RenjuEnv()
            print(f"RenjuEnv initialized with board size: {game_context['board_size']}")
        except Exception as e:
            print(f"Error initializing RenjuEnv: {e}")
            raise
    
    if game_context["model"] is None:
        model_path = "renju_model.zip"
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            # В этом случае AI не будет работать, можно либо вернуть ошибку,
            # либо предусмотреть режим игры без AI или с более простым AI.
            # Для данного примера, мы предполагаем, что модель есть.
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        try:
            # При загрузке модели, env должен быть уже инициализирован и соответствовать тому,
            # на котором модель обучалась (особенно observation_space и action_space)
            game_context["model"] = PPO.load(model_path, env=game_context["env"])
            print(f"PPO model '{model_path}' loaded successfully.")
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            # Если модель не загрузилась, AI не сможет делать ходы.
            # Можно установить game_context["model"] = None и обрабатывать это в make_ai_move
            raise

# Попытка инициализировать при старте сервера
try:
    initialize_game_engine()
except Exception as e:
    print(f"Failed to initialize game engine on startup: {e}")
    # Сервер запустится, но API могут возвращать ошибки, пока движок не будет готов.

# --- API Эндпоинты ---

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Начинает новую игру."""
    global game_context
    try:
        if game_context["env"] is None or game_context["model"] is None:
             # Повторная попытка инициализации, если не удалось при старте
            print("Re-initializing game engine for new_game...")
            initialize_game_engine()

        data = request.get_json()
        human_plays_as_str = data.get('human_plays_as', 'black') # 'black' or 'white'

        game_context["human_player_value"] = string_to_player_value(human_plays_as_str)
        game_context["ai_player_value"] = -1 * game_context["human_player_value"]

        # Сброс игры в среде
        # game_reset должен вернуть начальное наблюдение и информацию
        obs, _ = game_context["env"].game_reset() 
        
        # Важно: game_reset в RenjuEnv должен устанавливать game.current_player в -1 (черные)
        # Если current_player после сброса не -1, это нужно исправить в RenjuEnv.
        # print(f"Game reset. Initial current_player: {game_context['env'].game.current_player}")


        ai_move_made = None
        game_over = False
        winner = None
        
        # Если AI играет за черных (ходит первым)
        if game_context["env"].game.current_player == game_context["ai_player_value"]:
            print("AI plays as Black, making the first move.")
            # AI делает ход
            # В вашем коде есть специальная логика для первых ходов AI
            history_len = len(game_context["env"].game.history)
            ai_x, ai_y = None, None

            if history_len == 0 and hasattr(game_context["env"].game, 'step_1'):
                print("AI making step_1")
                game_context["env"].game.step_1() 
                ai_x, ai_y = game_context["env"].game.history[-1][1]

            else: # Общая логика хода AI, если step_1 не применим или не существует
                return jsonify({"error": "AI: No random moves available."}), 500
            
            ai_move_made = {"row": ai_y, "col": ai_x} # row это y, col это x

            
            game_context["env"].game.current_player *= -1 # Смена игрока

        return jsonify({
            "message": "New game started.",
            "board": game_context["env"].game.board.tolist(),
            "currentPlayer": player_value_to_string(game_context["env"].game.current_player),
            "ai_player_value_server": game_context["ai_player_value"], # для отладки
            "human_player_value_server": game_context["human_player_value"], # для отладки
            "ai_move": ai_move_made,
            "game_over": game_over,
            "winner": winner
        })

    except Exception as e:
        print(f"Error in /api/new_game: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Обрабатывает ход человека, затем делает ход AI."""
    global game_context
    if not game_context["env"] or not game_context["model"]:
        return jsonify({"error": "Game engine not initialized. Try starting a new game."}), 500

    data = request.get_json()
    row, col = data.get('row'), data.get('col') # row это y, col это x

    # Убедимся, что это ход человека
    if game_context["env"].game.current_player != game_context["human_player_value"]:
        return jsonify({"error": "Not human's turn.", "human_move_valid": False}), 400

    # 1. Ход человека
    human_x, human_y = col, row 
    status = game_context["env"].game.make_move(human_x, human_y)
    
    if not status:
        log_message = game_context["env"].game.logs[-1] if game_context["env"].game.logs else "Unknown error"
        return jsonify({
            "error": f"Invalid human move: {log_message}", 
            "human_move_valid": False,
            "board": game_context["env"].game.board.tolist(), # Отправляем текущее состояние доски
            "currentPlayer": player_value_to_string(game_context["env"].game.current_player)
        }), 400


    # Проверка на победу/ничью после хода человека
    if game_context["env"].game.is_win(human_x, human_y):
        return jsonify({
            "board": game_context["env"].game.board.tolist(),
            "human_move_valid": True,
            "ai_move": None,
            "game_over": True,
            "winner": player_value_to_string(game_context["human_player_value"]),
            "currentPlayer": player_value_to_string(game_context["env"].game.current_player) 
        })

    if game_context["env"].game.is_draw():
        return jsonify({
            "board": game_context["env"].game.board.tolist(),
            "human_move_valid": True,
            "ai_move": None,
            "game_over": True,
            "winner": "draw",
            "currentPlayer": player_value_to_string(game_context["env"].game.current_player)
        })
    
    # Смена игрока (теперь ход AI)
    game_context["env"].game.current_player *= -1

    # 2. Ход AI
    ai_x, ai_y = None, None
    history_len = len(game_context["env"].game.history)
    
    if history_len == 1 and hasattr(game_context["env"].game, 'step_2'): # AI - второй ход в игре
        print("AI making step_2")
        game_context["env"].game.step_2()
        ai_x, ai_y = game_context["env"].game.history[-1][1]
    elif history_len == 2 and hasattr(game_context["env"].game, 'step_3'): # AI - третий ход в игре
        print("AI making step_3")
        game_context["env"].game.step_3()
        ai_x, ai_y = game_context["env"].game.history[-1][1]
    else:
        print("AI making a standard move.")
        current_ai_obs = game_context["env"]._get_masked_obs()

        action, _ = game_context["model"].predict(current_ai_obs, deterministic=True)
        ai_x = int(action % game_context["env"].game.size)
        ai_y = int(action // game_context["env"].game.size)
        
        print(f"AI predicted move: x={ai_x}, y={ai_y}")
        status_ai = game_context["env"].game.make_move(ai_x, ai_y)
        if not status_ai:
            print(f"AI made an invalid move: {ai_x},{ai_y}. Log: {game_context['env'].game.logs[-1] if game_context['env'].game.logs else 'Unknown'}. Trying random.")
            coord = game_context["env"].game.random_step()
            if coord:
                ai_x, ai_y = coord
                print(f"AI random fallback move: x={ai_x}, y={ai_y}")
                game_context["env"].game.make_move(ai_x, ai_y) # Предполагаем, что случайный ход всегда валиден, если доступен
            else: # Ничья или нет доступных ходов
                 # Если AI не может сделать ход, это может быть ничья, которая должна была быть поймана ранее,
                 # или ошибка в логике. Возвращаем состояние доски как есть.
                return jsonify({
                    "board": game_context["env"].game.board.tolist(),
                    "human_move_valid": True,
                    "ai_move": None, # AI не смог сделать ход
                    "game_over": game_context["env"].game.is_draw(), # Проверяем на ничью
                    "winner": "draw" if game_context["env"].game.is_draw() else None,
                    "message": "AI could not make a move. Board might be full or error.",
                    "currentPlayer": player_value_to_string(game_context["env"].game.current_player) # Чей ход был бы, если бы AI не смог
                })

    ai_move_response = {"row": int(ai_y), "col": int(ai_x)}

    print(f"AI move made: x={ai_x}, y={ai_y}")

    # Проверка на победу/ничью после хода AI
    if game_context["env"].game.is_win(ai_x, ai_y):
        return jsonify({
            "board": game_context["env"].game.board.tolist(),
            "human_move_valid": True,
            "ai_move": ai_move_response,
            "game_over": True,
            "winner": player_value_to_string(game_context["ai_player_value"]),
            "currentPlayer": player_value_to_string(game_context["env"].game.current_player)
        })

    if game_context["env"].game.is_draw():
        return jsonify({
            "board": game_context["env"].game.board.tolist(),
            "human_move_valid": True,
            "ai_move": ai_move_response,
            "game_over": True,
            "winner": "draw",
            "currentPlayer": player_value_to_string(game_context["env"].game.current_player)
        })

    # Смена игрока (теперь ход человека)
    game_context["env"].game.current_player *= -1
    
    return jsonify({
        "board": game_context["env"].game.board.tolist(),
        "human_move_valid": True,
        "ai_move": ai_move_response,
        "game_over": False,
        "winner": None,
        "currentPlayer": player_value_to_string(game_context["env"].game.current_player)
    })
@app.route('/api/get_board', methods=['GET'])
def get_board():
    """Возвращает текущее состояние доски."""
    if game_context["env"] is None:
        return jsonify({"error": "Game engine not initialized."}), 500
    board = game_context["env"].game.board.tolist()
    return jsonify({"board": board})


@app.route('/', methods=['GET'])
def index():
    """Возвращает страницу renju_3d.html"""
    return app.send_static_file('./renju_3d.html')


# --- Запуск Flask приложения ---
if __name__ == '__main__':
    print("Starting Flask server for Renju AI...")
    # Перед запуском убедимся, что движок инициализирован, если это не произошло глобально.
    if game_context["env"] is None or game_context["model"] is None:
        try:
            print("Attempting to initialize game engine before starting server...")
            initialize_game_engine()
        except Exception as e:
            print(f"CRITICAL: Failed to initialize game engine at final attempt: {e}")
            # Можно решить не запускать сервер, если критические компоненты не загружены.
            # exit(1) 
    
    app.run(debug=True, host='0.0.0.0', port=5000)

