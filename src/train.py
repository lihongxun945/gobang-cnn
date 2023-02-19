from five_chess import FiveChess
from model import FiveChessModel
import numpy as np


def train(model_path='best_model.h5', board_size=15, win_count=5, max_steps=1000,
          self_play_count=100, max_buffer_size=100000, batch_size=128, epochs=10, learning_rate=0.01):
    game = FiveChess(board_size=board_size, win_count=win_count, max_steps=max_steps)
    model = FiveChessModel(board_size=board_size, learning_rate=learning_rate)
    buffer_input, buffer_policy, buffer_value = [], [], []
    for i in range(self_play_count):
        input_list, policy_list, value_list = game.collect_self_play_data(model=model)
        buffer_input += input_list
        buffer_policy += policy_list
        buffer_value += value_list
        while len(buffer_input) > max_buffer_size:
            buffer_input.pop(0)
            buffer_policy.pop(0)
            buffer_value.pop(0)
        if len(buffer_input) > batch_size:
            indices = np.random.choice(len(buffer_input), batch_size, replace=False)
            batch_input = [buffer_input[i] for i in indices]
            batch_policy = [buffer_policy[i] for i in indices]
            batch_value = [buffer_value[i] for i in indices]
            model.model.fit(np.array(batch_input), [np.array(batch_policy), np.array(batch_value)], epochs=epochs, verbose=1)
        if i % 10 == 0:
            model.model.save(model_path)  # Save the model every 10 iterations
    model.model.save(model_path)  # Save the final model after training

if __name__ == '__main__':
    train()