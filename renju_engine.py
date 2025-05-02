import numpy as np
from typing import List, Tuple

# Определяем типы для лучшей читаемости
Point = Tuple[int, int] # (x, y) coordinate
History = List[Tuple[int, Point]] # List of (player, (x, y))
MoveStatus = str # Type for move result strings
Board = np.ndarray # Type for the game board

class GameEngine:
    """
    Основной класс для представления доски и логики правил Рендзю (Гомоку с фолами).
    Координаты: (x, y), где x - столбец (0..size-1), y - строка (0..size-1).
               (0, 0) - левый верхний угол.
    Игроки: -1 - Черные (Black), 1 - Белые (White).
    Клетки: 0 - Пусто, -1 - Черный камень, 1 - Белый камень.
    """

    DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0),  # Горизонталь ->
        (0, 1),  # Вертикаль v
        (1, 1),  # Диагональ \
        (1, -1)  # Диагональ /
    ]

    def __init__(self, size: int = 15):
        if size < 5: # Минимальный размер для игры 5 в ряд
            raise ValueError("Board size must be at least 5")
        self.size: int = size
        self.center: Point = (size // 2, size // 2)
        self.history: History = [] # История ходов (игрок, (x, y))
        
        self.board: Board = np.zeros((size, size), dtype=int)  # 0 - пусто, -1 - черный, 1 - белый
        
        self.current_player: int = -1  # -1 (Black) или 1 (White)

    def is_valid_move(self, x: int, y: int) -> bool:
        # отладка: выводим все
        # print(f"Проверка хода: ({x}, {y})")
        # print(f"Текущий игрок: {'Черные' if self.current_player == -1 else 'Белые'}")
        # print(f"История ходов: {self.history}")
        # print(f"Доска:\n{self.board}")


        # Проверка на существование клетки
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        # Проверка, что клетка пуста
        if self.board[y, x] != 0:
            return False
        return True

    def make_move(self, x: int, y: int) -> MoveStatus:
        if not self.is_valid_move(x, y):
            return f"Недопустимый ход: {x}, {y}"
        
        # Специальные условия для первых ходов
        if len(self.history) == 0: # Первый ход Черных
            if (x, y) != self.center:
                return "Первый ход Черных должен быть в центр"
        elif len(self.history) == 1: # Второй ход Белых
            if abs(x - self.center[0]) > 1 or abs(y - self.center[1]) > 1:
                return "Второй ход Белых должен быть вплотную к первому камню Черных"
        elif len(self.history) == 2: # Третий ход Черных
            if abs(x - self.center[0]) > 2 or abs(y - self.center[1]) > 2:
                return "Третий ход Черных должен быть в пределах центрального квадрата 5x5"
        
        # Запрещенные ходы для Черных
        if self.current_player == -1:
            # Проверка на построение длинного ряда
            if self.is_long_row(x, y):
                return "Черным запрещено строить длинные ряды"
            # Проверка на вилки
            if self.is_forbidden_fork(x, y):
                return "Черным запрещена вилка"
        
        # Сделать ход
        self.board[y, x] = self.current_player
        self.history.append((self.current_player, (x, y)))
        
        # Проверить победу
        if self.is_win(x, y):
            return "Победа" + (" Черных" if self.current_player == -1 else " Белых")
        
        # Сменить игрока
        self.current_player *= -1
        
        return "Ход сделан"

    def is_long_row(self, x: int, y: int) -> bool:
        for direction in self.DIRECTIONS:
            count = 1
            for i in range(1, 6):
                nx, ny = x + i * direction[0], y + i * direction[1]
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                    break
                count += 1
            for i in range(1, 6):
                nx, ny = x - i * direction[0], y - i * direction[1]
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                    break
                count += 1
            if count >= 6:
                return True
        return False

    def is_win(self, x: int, y: int) -> bool:
        for direction in self.DIRECTIONS:
            count = 1
            for i in range(1, 6):
                nx, ny = x + i * direction[0], y + i * direction[1]
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                    break
                count += 1
            for i in range(1, 6):
                nx, ny = x - i * direction[0], y - i * direction[1]
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                    break
                count += 1
            if count == 5:
                return True
        return False

    def is_forbidden_fork(self, x: int, y: int) -> bool:
        # Проверка вилки четыре-на-четыре
        if self.count_fork_four(x, y) > 1:
            return True
        # Проверка вилки три-на-три
        if self.count_fork_three(x, y) > 1:
            return True
        return False

    def count_fork_four(self, x: int, y: int) -> int:
        count = 0
        for direction in self.DIRECTIONS:
            # Проверка, можно ли построить четверку в этом направлении
            if self.is_four(x, y, direction):
                count += 1
        return count

    def is_four(self, x: int, y: int, direction: Tuple[int, int]) -> bool:
        # Проверка, можно ли построить четверку в заданном направлении
        count = 1
        for i in range(1, 5):
            nx, ny = x + i * direction[0], y + i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                break
            count += 1
        for i in range(1, 5):
            nx, ny = x - i * direction[0], y - i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                break
            count += 1
        # Если count == 4, то это четверка
        if count == 4:
            # Проверка, можно ли достроить до пятерки
            if self.can_build_to_five(x, y, direction):
                return True
        return False

    def can_build_to_five(self, x: int, y: int, direction: Tuple[int, int]) -> bool:
        # Проверка, можно ли достроить до пятерки в заданном направлении
        for i in range(1, 5):
            nx, ny = x + i * direction[0], y + i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                if self.is_valid_move(nx, ny):
                    return True
                break
        for i in range(1, 5):
            nx, ny = x - i * direction[0], y - i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                if self.is_valid_move(nx, ny):
                    return True
                break
        return False

    def count_fork_three(self, x: int, y: int) -> int:
        count = 0
        for direction in self.DIRECTIONS:
            # Проверка, можно ли построить тройку в этом направлении
            if self.is_three(x, y, direction):
                count += 1
        return count

    def is_three(self, x: int, y: int, direction: Tuple[int, int]) -> bool:
        # Проверка, можно ли построить тройку в заданном направлении
        count = 1
        for i in range(1, 4):
            nx, ny = x + i * direction[0], y + i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                break
            count += 1
        for i in range(1, 4):
            nx, ny = x - i * direction[0], y - i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                break
            count += 1
        # Если count == 3, то это тройка
        if count == 3:
            # Проверка, можно ли достроить до открытой четверки
            if self.can_build_to_open_four(x, y, direction):
                return True
        return False

    def can_build_to_open_four(self, x: int, y: int, direction: Tuple[int, int]) -> bool:
        # Проверка, можно ли достроить до открытой четверки в заданном направлении
        for i in range(1, 4):
            nx, ny = x + i * direction[0], y + i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                if self.is_valid_move(nx, ny):
                    # Проверка, что после достраивания не образуется длинный ряд или вилка четыре-на-четыре
                    new_x, new_y = nx, ny
                    self.board[new_y, new_x] = self.current_player
                    if not self.is_long_row(new_x, new_y) and self.count_fork_four(new_x, new_y) <= 1:
                        self.board[new_y, new_x] = 0
                        return True
                    self.board[new_y, new_x] = 0
                break
        for i in range(1, 4):
            nx, ny = x - i * direction[0], y - i * direction[1]
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size or self.board[ny, nx] != self.current_player:
                if self.is_valid_move(nx, ny):
                    # Проверка, что после достраивания не образуется длинный ряд или вилка четыре-на-четыре
                    new_x, new_y = nx, ny
                    self.board[new_y, new_x] = self.current_player
                    if not self.is_long_row(new_x, new_y) and self.count_fork_four(new_x, new_y) <= 1:
                        self.board[new_y, new_x] = 0
                        return True
                    self.board[new_y, new_x] = 0
                break
        return False
