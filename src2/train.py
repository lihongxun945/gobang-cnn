import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
import copy

class Board:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)

    def play(self, row, col, player):
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = player
        return True

    def is_win(self, player):
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == player:
                    if j <= self.size - 5 and np.all(self.board[i][j:j+5] == player):
                        return True
                    if i <= self.size - 5 and np.all(self.board[i:i+5, j] == player):
                        return True
                    if i <= self.size - 5 and j <= self.size - 5 and np.all(np.diag(self.board[i:i+5, j:j+5]) == player):
                        return True
                    if i >= 4 and j <= self.size - 5 and np.all(np.diag(self.board[i-4:i+1, j:j+5]) == player):
                        return True
        return False

class Node:
    def __init__(self, parent, move):
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0

    def select_child(self, c_param):
        return max(self.children, key=lambda c: c.wins / c.visits + c_param * np.sqrt(2 * np.log(self.visits) / c.visits))

    def expand(self, board):
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == 0:
                    self.children.append(Node(self, (i, j)))

    def update(self, result):
        self.visits += 1
        self.wins += result

class MCTS:
    def __init__(self, board, model):
        self.board = board
        self.model = model
        self.root = Node(None, None)

    def search(self, num_simulations):
        for i in range(num_simulations):
            node = self.root
            board_copy = copy.deepcopy(self.board)
            while node.children:
                node = node.select_child(c_param=1.4)
                row, col = node.move
                board_copy.play(row, col, player=2)
            if node.visits > 0:
                self.expand(node, board_copy)
                node = random.choice(node.children)
                row, col = node.move
                board_copy.play(row, col, player=1)
            value = self.evaluate(board_copy)
            while node:
                node.update(value)
                node = node.parent

    def evaluate(self, board):
        state = self.get_state(board)
        value = self.model.predict(np.array([state]))[0][0]
        return value

    def expand(self, node, board):
        node.expand(board)
       
        for child in node.children:
            result = self.simulate(child, board)
            child.update(result)

    def simulate(self, node, board):
        board_copy = copy.deepcopy(board)
        row, col = node.move
        board_copy.play(row, col, player=1)
        while not board_copy.is_win(1) and not board_copy.is_win(2):
            row, col = self.select_move(board_copy)
            board_copy.play(row, col, player=2)
            if board_copy.is_win(2):
                return -1
            row, col = self.select_move(board_copy)
            if board_copy.play(row, col, player=1):
                if board_copy.is_win(1):
                    return 1
        if board_copy.is_win(1):
            return 1
        else:
            return -1

    def select_move(self, board):
        state = self.get_state(board)
        q_values = self.model.predict(np.array([state]))[0]
        valid_moves = np.array([[i, j] for i in range(board.size) for j in range(board.size) if board.board[i][j] == 0])
        q_values = q_values[:len(valid_moves)]
        move = valid_moves[np.argmax(q_values)]
        return move

    def get_state(self, board):
        state = np.zeros((board.size, board.size, 3), dtype=int)
        state[:, :, 0] = (board.board == 1).astype(int)
        state[:, :, 1] = (board.board == 2).astype(int)
        state[:, :, 2] = (board.board == 0).astype(int)
        return state
board = Board()
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(board.size, board.size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='tanh')
])
model.compile(optimizer='adam', loss='mse')

mcts = MCTS(board, model)

for i in range(1000):
    mcts.search(100)

board.reset()
while True:
    print(board.board)
    if board.is_win(1):
        print('Player 1 wins!')
        break
    if board.is_win(2):
        print('Player 2 wins!')
        break
    row, col = mcts.select_move(board)
    board.play(row, col, player=1)
