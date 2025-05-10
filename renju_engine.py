import numpy as np
from typing import List, Tuple
from icecream import ic
from collections import deque

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
        self.logs = deque(maxlen=100) # Логи для отладки
        
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
    
    def get_valid_moves(self) -> List[Point]:
        """
        Возвращает список всех допустимых ходов для текущего игрока (только пустые клетки).
        """
        empty = np.argwhere(self.board == 0)
        # Возвращаем список (x, y), чтобы соответствовать вашему формату
        return [(x, y) for y, x in empty]

    
    def get_allowed_mask(self) -> Board:
        """
        Возвращает маску допустимых ходов для текущего игрока с учетом правил.
        """
        mask = np.zeros((self.size, self.size), dtype=np.int32)
        empty = np.argwhere(self.board == 0)
        for y, x in empty:
            if self._check_allowed_move(x, y):
                mask[y, x] = 1
        return mask

    def get_allowed_moves(self) -> List[Point]:
        # 1. Быстро находим все пустые клетки
        empty = np.argwhere(self.board == 0)                
        # 2. Применяем сложные проверки только к ним
        allowed_moves = [
            (x, y) for y, x in empty
            if self._check_allowed_move(x, y)
        ]
        return allowed_moves

    # def get_valid_moves(self) -> List[Point]:
    #     """
    #     Возвращает список всех допустимых ходов для текущего игрока.
    #     """
    #     valid_moves = []
    #     for y in range(self.size):
    #         for x in range(self.size):
    #             if self.is_valid_move(x, y):
    #                 valid_moves.append((x, y))
    #     return valid_moves
    # def get_allowed_mask(self) -> Board:
    #     """
    #     Возвращает список всех допустимых ходов для текущего игрока с учетом правил.
    #     """
    #     mask = np.zeros((self.size, self.size))
    #     for y in range(self.size):
    #         for x in range(self.size):
    #             if self.is_valid_move(x, y) and self._check_allowed_move(x, y):
    #                 mask[y, x] = 1
    #     return mask
    
    # def get_allowed_moves(self) -> List[Point]:
    #     """
    #     Возвращает список всех допустимых ходов для текущего игрока с учетом правил.
    #     """
    #     allowed_moves = []
    #     for y in range(self.size):
    #         for x in range(self.size):
    #             if self.is_valid_move(x, y) and self._check_allowed_move(x, y):
    #                 allowed_moves.append((x, y))
    #     return allowed_moves
    
    def get_coordinates(self, action:int) -> Point:
        # Convert action to coordinates (x, y) and return None, None if invalid
        valid_moves = self.get_allowed_moves()

        if len(valid_moves) == 0:
            return 'Ничья'

        point: Point = (action % self.size, action // self.size)
        if point in valid_moves:
            # print(f"-----------------------------Допустимый ход: {point}------------------------------")
            return point
        # Если действие не является допустимым ходом, возвращаем None
        # print(f"Недопустимый ход: {point}")
        return (None, None)


    
    def _check_allowed_move(self, x: int, y: int) -> bool:
        # Проверяем условия для первого и второго хода
        if len(self.history) == 0 and (x, y) != self.center:
            ic("Первый ход Черных должен быть в центр")
            self.logs.append(f"Первый ход Черных должен быть в центр: {x}, {y}")
            return False
        elif len(self.history) == 1:
            if abs(x - self.center[0]) > 1 or abs(y - self.center[1]) > 1:
                ic("Второй ход Белых должен быть вплотную к первому камню Черных")
                self.logs.append(f"Второй ход Белых должен быть вплотную к первому камню Черных: {x}, {y}")
                return False
        elif len(self.history) == 2:
            if abs(x - self.center[0]) > 2 or abs(y - self.center[1]) > 2:
                ic("Третий ход Черных должен быть в пределах центрального квадрата 5x5")
                self.logs.append(f"Третий ход Черных должен быть в пределах центрального квадрата 5x5: {x}, {y}")
                return False

        # Проверяем условия для черных
        if self.current_player == -1:
            # Проверка на построение длинного ряда
            if self.is_long_row(x, y):
                ic("Черным запрещено строить длинные ряды")
                self.logs.append(f"Черным запрещено строить длинные ряды: {x}, {y}")
                return False
            # Проверка на вилки
            if self.is_forbidden_fork(x, y):
                ic("Черным запрещена вилка")
                self.logs.append(f"Черным запрещена вилка: {x}, {y}")
                return False

        return True


    def make_move(self, x: int, y: int) -> bool:
        if not self.is_valid_move(x, y):
            ic(f"Недопустимый ход: {x}, {y}")
            self.logs.append(f"Недопустимый ход: {x}, {y}")
            return False
        
        if not self._check_allowed_move(x, y):
            ic(f"Запрещенный ход: {x}, {y}")
            return False
        
        # Сделать ход
        self.board[y, x] = self.current_player
        self.history.append((self.current_player, (x, y)))
        
        # Проверить победу
        if self.is_win(x, y):
            ic("Победа" + (" Черных" if self.current_player == -1 else " Белых"))
            self.logs.append(f"Победа" + (" Черных" if self.current_player == -1 else " Белых"))
            return True
        
        # Сменить игрока
        # self.current_player *= -1
        
        return True

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
            if count >= 5:
                return True
        return False
    
    def is_draw(self) -> bool:
        # Проверяем, есть ли свободные клетки
        # Если нет свободных клеток, то ничья
        return not np.any(self.board == 0)

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
    
    def __append_move(self, x, y, player):
        self.board[x, y] = player
        self.history.append((player, (x, y)))
    def step_1(self):
        # 1. Первый ход: черный (-1) в центр
        self.__append_move(self.center[0], self.center[1], -1)
        print(f"Первый ход Черных в центр: {self.center}")
    
    def step_2(self):
        # 2. Второй ход: белый (1) в любую клетку с радиусом 1 от центра (но не центр)
        board_3_3 = self.board[self.center[0] - 1:self.center[0] + 2, self.center[1] - 1:self.center[1] + 2]
        # Получаем индексы свободных клеток
        free_indices = np.argwhere(board_3_3 == 0)
        # Если нет свободные клетки, кидаем ошибку
        assert free_indices.size > 0, "Нет свободных клеток в 3x3 вокруг центра"
        idx = np.random.choice(free_indices.shape[0])
        y2, x2 = free_indices[idx]
        y2 += self.center[0] - 1
        x2 += self.center[1] - 1
        # Проверяем, что клетка свободна
        assert self.board[y2, x2] == 0, "Клетка занята"
        self.__append_move(x2, y2, 1)
        print(f"Второй ход Белых: {x2}, {y2}")

    def step_3(self):
        # 3. Третий ход: черный (-1) в любую свободную клетку 5x5 вокруг центра (кроме занятых)
        board_5_5 = self.board[self.center[0] - 2:self.center[0] + 3, self.center[1] - 2:self.center[1] + 3]
        # Получаем индексы свободных клеток
        free_indices = np.argwhere(board_5_5 == 0)
        # Если нет свободные клетки, кидаем ошибку 
        assert free_indices.size > 0, "Нет свободных клеток в 5x5 вокруг центра"

        idx = np.random.choice(free_indices.shape[0])
        y3, x3 = free_indices[idx]
        y3 += self.center[0] - 2
        x3 += self.center[1] - 2

        # Проверяем, что клетка свободна
        assert self.board[y3, x3] == 0, "Клетка занята"
        self.__append_move(x3, y3, -1)
        print(f"Третий ход Черных: {x3}, {y3}")

    def random_step(self) -> Point:
        """
        Возвращает случайный допустимый ход для текущего игрока.
        """
        valid_moves = self.get_allowed_moves()
        if len(valid_moves) == 0:
            return None
        return valid_moves[np.random.choice(len(valid_moves))]
