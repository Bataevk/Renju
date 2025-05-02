import sqlite3
import json

class PersistenceManager:
    def __init__(self, db_path='games.db'):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    moves TEXT,
                    winner TEXT
                )
            ''')

    def save_game(self, moves, winner):
        encoded_moves = self.encode_moves(moves)
        with self.conn:
            self.conn.execute(
                'INSERT INTO games (moves, winner) VALUES (?, ?)',
                (encoded_moves, winner)
            )

    def load_game(self, game_id):
        cursor = self.conn.execute('SELECT moves, winner FROM games WHERE id=?', (game_id,))
        row = cursor.fetchone()
        if row:
            moves = self.decode_moves(row[0])
            return moves, row[1]
        return None, None
    
    def delete_game(self, game_id):
        with self.conn:
            self.conn.execute('DELETE FROM games WHERE id=?', (game_id,))
            return self.conn.total_changes > 0
    
    def get_all_games(self):
        cursor = self.conn.execute('SELECT id, timestamp, winner FROM games')
        games = cursor.fetchall()
        return [{'id': row[0], 'timestamp': row[1], 'winner': row[2]} for row in games]

    def encode_moves(self, moves):
        # Преобразует  List of (player, (x, y)) в строку
        return json.dumps(moves)

    def decode_moves(self, encoded_moves):
        # Преобразует строку обратно в список ходов
        moves = json.loads(encoded_moves)
        return moves
