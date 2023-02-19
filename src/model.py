import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model


class FiveChessModel:
    def __init__(self, board_size=15, learning_rate=0.01):
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.input = Input(shape=(board_size, board_size, 2))
        self.hidden1 = Conv2D(32, 3, padding='same', activation='relu')(self.input)
        self.hidden2 = Conv2D(64, 3, padding='same', activation='relu')(self.hidden1)
        self.hidden3 = Conv2D(128, 3, padding='same', activation='relu')(self.hidden2)
        self.policy_output = Conv2D(2, 1, padding='same', activation='softmax')(self.hidden3)
        self.flatten = Flatten()(self.hidden3)
        self.value_hidden = Dense(64, activation='relu')(self.flatten)
        self.value_output = Dense(1, activation='tanh')(self.value_hidden)
        self.model = Model(self.input, [self.policy_output, self.value_output])
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=opt)

    def predict(self, board):
        encoded_board = board.encode_board()
        policy, value = self.model.predict(np.array([encoded_board]))
        policy = policy[0]
        value = value[0][0]
        candidates = np.argsort(-policy)[:board.board_size * board.board_size]
        for c in candidates:
            move = divmod(c, board.board_size)
            if board.is_legal_move(move):
                return move, value
        return None, value
