# main.py
from renju_engine import GameEngine
from kivy_gui import KivyApp
from persistence import PersistenceManager
from stable_baselines3 import PPO

class Manager:
    def __init__(self):
        self.model = PPO.load("renju_model.zip")
        self.persistence_manager = PersistenceManager()
        self.game_engine = GameEngine()
        self.gui = KivyApp(self)

    def start_game_with_ai(self):
        self.game_engine = GameEngine()
        self.gui.show_game_with_ai()

    def save_game(self, moves, winner):
        self.persistence_manager.save_game(moves, winner)

    def load_game(self, game_id):
        return self.persistence_manager.load_game(game_id)

    def delete_game(self, game_id):
        return self.persistence_manager.delete_game(game_id)

    def get_all_games(self):
        return self.persistence_manager.get_all_games()

    def make_move(self, x, y):
        status = self.game_engine.make_move(x, y)
        if "Победа" in status:
            self.save_game(self.game_engine.history, "Черные" if self.game_engine.current_player == 1 else "Белые")
        return status

    def get_ai_move(self):
        obs = self.game_engine._get_obs()
        action, _states = self.model.predict(obs)
        x = action % self.game_engine.size
        y = action // self.game_engine.size
        return x, y

    def run(self):
        self.gui.run()

if __name__ == "__main__":
    manager = Manager()
    manager.run()
