from five_chess import FiveChess;
from model import FiveChessModel;

game = FiveChess()

# 假设棋盘状态为 board_state
board_state = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 1, -1, 0],
               [0, 0, -1, 1, 0],
               [0, 0, 0, 0, 0]]

# 调用实例的 make_board() 方法更新棋盘状态
game.make_board(board_state)

# 调用实例的 encode_board() 方法将棋盘状态转换为神经网络的输入格式
encoded_board = game.encode_board()

# 创建一个五子棋模型实例
model = FiveChessModel()

# 加载之前训练好的模型参数
model.load_weights("model.h5")

# 调用模型的 predict() 方法，得到落子建议
move = game.predict(model)

# 打印落子建议
print("AI suggests move:", move)