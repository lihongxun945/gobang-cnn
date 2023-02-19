import tensorflow as tf
import numpy as np
import random

gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)

# 定义棋盘大小和游戏规则
board_size = 8
win_count = 4

# 定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(board_size, board_size, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(board_size*board_size, activation='softmax')
])

# 定义训练参数
model.compile(optimizer='adam', loss='categorical_crossentropy')

# 定义棋盘状态编码函数
def encode_board(board):
    encoded_board = np.zeros((board_size, board_size, 2), dtype=np.float32)
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 1:
                encoded_board[i][j][0] = 1.0
            elif board[i][j] == -1:
                encoded_board[i][j][1] = 1.0
    return encoded_board

# 定义落子函数
def make_move(board, player, move):
    x, y = move
    board[x][y] = player

# 定义预测函数
def predict(board, player):
    encoded_board = encode_board(board)
    prediction = model.predict(np.array([encoded_board]))
    prediction = prediction.reshape((board_size, board_size))
    # 对于已经落子的位置，预测值设为0
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] != 0:
                prediction[i][j] = 0.0
    # 对于非法位置，预测值设为0
    for i in range(board_size):
        for j in range(board_size):
            if not is_legal_move(board, (i, j)):
                prediction[i][j] = 0.0
    # 返回最大预测值对应的位置
    flattened_prediction = prediction.reshape(-1)
    index = np.argmax(flattened_prediction)
    move = (index // board_size, index % board_size)
    return move

# 定义判断合法性函数
def is_legal_move(board, move):
    x, y = move
    return x >= 0 and x < board_size and y >= 0 and y < board_size and board[x][y] == 0

# 定义判断胜负函数
def check_win(board, player):
    # 判断行
    for i in range(board_size):
        count = 0
        for j in range(board_size):
            if board[i][j] == player:
                count += 1
                if count == win_count:
                    return True
            else:
                count = 0
    # 判断列
    for j in range(board_size):
        count = 0
        for i in range(board_size):
            if board[i][j] == player:
                count += 1
                if count == win_count:
                    return True
            else:
                count = 0
    # 判断正对角线
    for i in range(win_count-1, board_size):
        count = 0
        for j in range(win_count):
            if board[i-j][j] == player:
                count += 1
                if count == win_count:
                    return True
            else:
                count = 0
    for j in range(1, board_size-win_count+1):
        count = 0
        for i in range(board_size-j):
            if board[i+j][i] == player:
                count += 1
                if count == win_count:
                    return True
            else:
                count = 0
    # 判断反对角线
    for i in range(win_count-1, board_size):
        count = 0
        for j in range(win_count):
            if board[i-j][board_size-1-j] == player:
                count += 1
                if count == win_count:
                    return True
            else:
                count = 0
    for j in range(board_size-win_count):
        count = 0
        for i in range(board_size-j-1):
            if board[i+j+1][board_size-1-i] == player:
                count += 1
                if count == win_count:
                    return True
            else:
                count = 0
    return False

# 定义自我对弈函数
def self_play():
    # 初始化棋盘和当前玩家
    board = np.zeros((board_size, board_size), dtype=np.int32)
    current_player = 1
    # 初始化落子记录
    moves = []
    while True:
        # 让当前玩家落子
        move = predict(board, current_player)
        make_move(board, current_player, move)
        moves.append((current_player, move))
        # 判断胜负
        if check_win(board, current_player):
            winner = current_player
            break
        if np.count_nonzero(board) == board_size*board_size:
            winner = 0  # 平局
            break
        # 切换玩家
        current_player = -current_player
    # 根据胜者来更新训练数据
    if winner == 1:
        winner_moves = [(1, move) if player == 1 else (0, move) for (player, move) in moves]
    elif winner == -1:
        winner_moves = [(0, move) if player == 1 else (1, move) for (player, move) in moves]
    else:
        winner_moves = [(0.5, move) for (player, move) in moves]
    inputs = []
    outputs = []
    for (result, move) in winner_moves:
        encoded_board = encode_board(board)
        inputs.append(encoded_board)
        output = np.zeros(board_size*board_size, dtype=np.float32)
        x, y = move
        index = x*board_size + y
        output[index] = result
        outputs.append(output)
    return inputs, outputs

# 进行自我对弈训练
for i in range(100):
    inputs, outputs = self_play()
    # 更新模型
    model.fit(np.array(inputs), np.array(outputs), batch_size=32, epochs=1)
