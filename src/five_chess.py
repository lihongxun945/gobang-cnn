import numpy as np


class FiveChess:
    def __init__(self, board_size=15, win_count=5, max_steps=1000):
        self.board_size = board_size
        self.win_count = win_count
        self.max_steps = max_steps
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.current_player = 1
        self.moves = []

    def encode_board(self):
        encoded_board = np.zeros((self.board_size, self.board_size, 2), dtype=np.float32)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    encoded_board[i][j][0] = 1.0
                elif self.board[i][j] == -1:
                    encoded_board[i][j][1] = 1.0
        return encoded_board

    def make_move(self, move):
        x, y = move
        self.board[x][y] = self.current_player
        self.moves.append((self.current_player, move))

    def predict(self, model, temperature=1.0):
        encoded_board = self.encode_board()
        prediction = model.predict(np.array([encoded_board]))[0]
        prediction = prediction.reshape((self.board_size, self.board_size))
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != 0:
                    prediction[i][j] = 0.0
        flattened_prediction = prediction.reshape(-1)
        if temperature == 0.0:
            index = np.argmax(flattened_prediction)
        else:
            distribution = flattened_prediction ** (1.0 / temperature)
            distribution /= np.sum(distribution)
            index = np.random.choice(np.arange(self.board_size*self.board_size), p=distribution)
        move = (index // self.board_size, index % self.board_size)
        return move

    def is_legal_move(self, move):
        x, y = move
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        if self.board[x][y] != 0:
            return False
        return True

    def check_win(self):
        for i in range(self.board_size):
            count = 0
            for j in range(self.board_size):
                if self.board[i][j] == self.current_player:
                    count += 1
                    if count == self.win_count:
                        return True
                else:
                    count = 0
        for j in range(self.board_size):
            count = 0
            for i in range(self.board_size):
                if self.board[i][j] == self.current_player:
                    count += 1
                    if count == self.win_count:
                        return True
                else:
                    count = 0
        for i in range(self.board_size-self.win_count+1):
            count = 0
            for j in range(self.board_size-i):
                if self.board[i+j][j] == self.current_player:
                    count += 1
                    if count == self.win_count:
                        return True
                else:
                    count = 0
        for j in range(1, self.board_size-self.win_count+1):
            count = 0
            for i in range(self.board_size-j):
                if self.board[i][self.board_size-1-j-i] == self.current_player:
                   
                    count += 1
                    if count == self.win_count:
                        return True
                else:
                    count = 0
        return False

    def play_game(self, player1, player2, show_board=True):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = 1
        self.moves = []
        if show_board:
            self.print_board()
        while len(self.moves) < self.max_steps:
            if self.current_player == 1:
                move = player1.predict(self)
            else:
                move = player2.predict(self)
            if self.is_legal_move(move):
                self.make_move(move)
                if show_board:
                    self.print_board()
                if self.check_win():
                    if show_board:
                        print("Player %d wins!" % self.current_player)
                    return self.current_player, self.moves
                self.current_player *= -1
            else:
                if show_board:
                    print("Illegal move by player %d: %s" % (self.current_player, move))
                return -self.current_player, self.moves
        if show_board:
            print("Game ended in a draw")
        return 0, self.moves

    def print_board(self):
        print("+" + "-" * self.board_size + "+")
        for i in range(self.board_size):
            line = "|"
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    line += "X"
                elif self.board[i][j] == -1:
                    line += "O"
                else:
                    line += " "
            line += "|"
            print(line)
        print("+" + "-" * self.board_size + "+")

    def collect_self_play_data(self, model, temperature=1.0, count=1):
        input_list, policy_list, value_list = [], [], []
        for i in range(count):
            winner, moves = self.self_play_game(model, temperature)
            if winner == 0:
                continue
            for move in moves:
                input_list.append(self.encode_board())
                policy = np.zeros((self.board_size * self.board_size), dtype=np.float32)
                policy[move[0] * self.board_size + move[1]] = 1.0
                policy_list.append(policy)
                value_list.append(winner)
        return input_list, policy_list, value_list

    def self_play_game(self, model, temperature=1.0):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.current_player = 1
        self.moves = []
        while len(self.moves) < self.max_steps:
            move = self.predict(model, temperature)
            if self.is_legal_move(move):
                self.make_move(move)
                if self.check_win():
                    return self.current_player, self.moves
                self.current_player *= -1
            else:
                return -self.current_player, self.moves
        return 0, self.moves
