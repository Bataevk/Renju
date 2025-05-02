# kivy_gui.py
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRectangleFlatButton
from kivy.lang import Builder


class MenuScreen(Screen):
    pass

class GameWithAIScreen(Screen):
    def on_enter(self):
        grid_layout = self.ids.grid_layout
        for y in range(15):
            for x in range(15):
                button = MDRectangleFlatButton(text="", size_hint_y=None, height="30dp")
                button.bind(on_press=lambda instance, x=x, y=y: print(MDApp.get_running_app().ai_manager.make_move(x, y)))
                grid_layout.add_widget(button)

class ViewGamesScreen(Screen):
    pass



class KivyApp(MDApp):
    def __init__(self, manager):
        super().__init__()
        self.ai_manager = manager

    def build(self):
        self.theme_cls.theme_style = "Light"
        return Builder.load_file('./kivyapp.kv')
        # return ScreenManager()

    def make_move(self, x, y):
        status = self.manager.make_move(x, y)
        if "Победа" in status:
            print(status)
        # Update GUI to reflect the move
        game_with_ai_screen = self.root.get_screen('game_with_ai')
        grid_layout = game_with_ai_screen.ids.grid_layout
        grid_layout.children[225 - (y * 15 + x) - 1].text = "X" if self.manager.game_engine.board[y, x] == -1 else "O"
        # Get AI move and update GUI
        ai_x, ai_y = self.manager.get_ai_move()
        self.manager.make_move(ai_x, ai_y)
        grid_layout.children[225 - (ai_y * 15 + ai_x) - 1].text = "X" if self.manager.game_engine.board[ai_y, ai_x] == -1 else "O"

    def view_games(self):
        view_games_screen = self.root.get_screen('view_games')
        scroll_view = view_games_screen.ids.scroll_view
        layout = view_games_screen.ids.layout
        layout.clear_widgets()
        layout.height = len(self.manager.get_all_games()) * 50
        for game in self.manager.get_all_games():
            layout.add_widget(MDLabel(text=f"Game {game['id']} - {game['timestamp']} - {game['winner']}"))

if __name__ == "__main__":
    from main import Manager
    KivyApp(Manager()).run() 