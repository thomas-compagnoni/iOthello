from os.path import join, dirname
import os
import warnings
from itertools import chain
from multiprocessing import cpu_count
from random import choice

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

warnings.simplefilter('ignore', UserWarning)


class Othello:

    def __init__(self):
        self.board = self.reset_board()

        self.x_board = (1, 2, 3, 4, 5, 6)
        self.y_board = (1, 2, 3, 4, 5, 6)

        self.x_square = (-1, 0, 1)
        self.y_square = (-1, 0, 1)

        self.no_possible_moves = [False, False]

        self.models = self.load_models()

    @staticmethod
    def load_models():
        if os.path.exists(join(dirname(__file__), 'models')):
            models = []
            for move in range(32):
                filename = join(dirname(__file__), 'models', f'{move}_ridge.sav')
                models.append(joblib.load(filename))
            return models
        else:
            return None

    @staticmethod
    def reset_board():
        board = np.zeros((8, 8), dtype=np.int8)
        board[[3, 4], [4, 3]] = 1
        board[[4, 3], [4, 3]] = -1
        board = board.tolist()
        return board

    def find_pieces(self, board, player):
        pieces = []
        for xb in self.x_board:
            for yb in self.y_board:
                if board[xb][yb] == player:
                    pieces.append((xb, yb))
        return pieces

    def find_possible_moves(self, board, player):
        moves = set()
        for xb in self.x_board:
            for yb in self.y_board:
                if board[xb][yb] == player:
                    for xs in self.x_square:
                        for ys in self.y_square:
                            if board[xb + xs][yb + ys] == -player:
                                x_to_evaluate = xb + xs * 2
                                y_to_evaluate = yb + ys * 2
                                while 1 <= x_to_evaluate <= 6 and 1 <= y_to_evaluate <= 6:
                                    if board[x_to_evaluate][y_to_evaluate] == 0:
                                        moves.add((x_to_evaluate, y_to_evaluate))
                                        break
                                    elif board[x_to_evaluate][y_to_evaluate] == player:
                                        break
                                    else:
                                        x_to_evaluate += xs
                                        y_to_evaluate += ys
        return tuple(moves)

    def update_board(self, board, row, column, player):
        board = [z[:] for z in board]
        board[row][column] = player
        for xs in self.x_square:
            for ys in self.y_square:
                if board[row + xs][column + ys] == -player:
                    board_mask = [z[:] for z in board]
                    x_to_evaluate = row + xs
                    y_to_evaluate = column + ys
                    while 1 <= x_to_evaluate <= 6 and 1 <= y_to_evaluate <= 6:
                        if board_mask[x_to_evaluate][y_to_evaluate] == player:
                            board = board_mask
                            break
                        elif board_mask[x_to_evaluate][y_to_evaluate] == -player:
                            board_mask[x_to_evaluate][y_to_evaluate] = player
                            x_to_evaluate += xs
                            y_to_evaluate += ys
                        else:
                            break
        return board

    def next_layer(self, board, player, return_moves=True, recursive=False):
        """ board: np.array """
        moves = {}
        for xb in self.x_board:
            for yb in self.y_board:
                if board[xb, yb] == player:
                    for xs in self.x_square:
                        for ys in self.y_square:
                            if board[xb + xs, yb + ys] == -player:
                                x_to_evaluate = xb + xs * 2
                                y_to_evaluate = yb + ys * 2
                                while 1 <= x_to_evaluate <= 6 and 1 <= y_to_evaluate <= 6:
                                    if board[x_to_evaluate, y_to_evaluate] == 0:
                                        moves.setdefault((x_to_evaluate, y_to_evaluate), [])
                                        moves[(x_to_evaluate, y_to_evaluate)].append((xb, yb))
                                        break
                                    elif board[x_to_evaluate, y_to_evaluate] == player:
                                        break
                                    else:
                                        x_to_evaluate += xs
                                        y_to_evaluate += ys
        if len(moves) == 0 and not recursive:
            return self.next_layer(board, -player, False, True)
        elif len(moves) == 0 and recursive:
            return np.array([board])
        list_of_results = np.array([board] * len(moves))
        n = 0
        for (xmove, ymove), start_cells in moves.items():
            list_of_results[n, xmove, ymove] = player
            for xcell, ycell in start_cells:
                dirx = abs(xmove - xcell)
                diry = abs(ymove - ycell)
                signdirx = dirx and (xmove - xcell) // dirx or 0
                signdiry = diry and (ymove - ycell) // diry or 0
                for i in range(max(dirx, diry) - 1):
                    xcell += signdirx
                    ycell += signdiry
                    list_of_results[n, xcell, ycell] = player
            n += 1
        if return_moves:
            return list(moves.keys()), list_of_results
        else:
            return list_of_results

    def find_best_move_simple(self, board):
        moves, boards = self.next_layer(np.array(board), -1)
        move = moves[boards.sum(axis=2).sum(axis=1).argmin()]
        return move

    def find_best_move_model(self, board, move_number):
        scores = []
        moves, boards = self.next_layer(np.array(board), -1)
        if move_number < 30:
            boards_second_layer = [self.next_layer(np.array(board), 1, return_moves=False)[:, 1:7, 1:7] for board in boards]
            model = self.models[move_number + 2]
            for boards in boards_second_layer:
                scores.append(max([model.predict(board.reshape(1, -1))[0] for board in boards]))
        else:
            model = self.models[move_number + 1]
            scores.append([model.predict(board[1:7, 1:7].reshape(1, -1))[0] for board in boards])
        move = moves[scores.index(min(scores))]
        return move

    def test_vs_random_bot(self, _):
        board = self.reset_board()
        player = 1
        move_count = 0
        no_possible_moves = [False, False]
        while True:
            moves = self.find_possible_moves(board, player)
            if len(moves) == 0:  # if no possible moves: for both player end the game, else change player and continue
                no_possible_moves[max(0, player)] = True
                if all(no_possible_moves):
                    return np.sum(board)
                else:
                    player = -player
                    continue
            elif player == 1:
                movex, movey = choice(moves)
                no_possible_moves[1] = False
            else:
                if move_count == 31:
                    movex, movey = self.find_best_move_simple(board)
                else:
                    movex, movey = self.find_best_move_model(board, move_count)
                no_possible_moves[0] = False
            board = self.update_board(board, movex, movey, player)
            player = -player
            move_count += 1

    def multiprocessing_test_vs_random_bot(self, simulations, n_jobs=cpu_count() - 2):
        if not self.models:
            raise "There aren't any models!"
        tests = process_map(self.test_vs_random_bot, range(simulations), max_workers=n_jobs, chunksize=1,
                            total=simulations, unit='simulations', mininterval=1, smoothing=0.2, desc='Running simulations... ')
        return tests

    def random_simulations(self, simulations=10000, to_csv=False):
        progress_bar = tqdm(range(simulations), total=simulations, unit='simulation', mininterval=1, desc='Running random simulations... ')
        X = np.zeros((32, simulations, 36), dtype=np.int8)
        y = np.zeros(simulations, dtype=np.int8)
        for i in progress_bar:
            player = 1
            move_count = 0
            while True:
                moves = self.find_possible_moves(self.board, player)
                if len(moves) == 0:
                    self.no_possible_moves[max(0, player)] = True
                    if all(self.no_possible_moves):
                        y[i] = np.sum(self.board)
                        break
                    else:
                        player = -player
                        continue
                elif player == 1:
                    movex, movey = choice(moves)
                    self.no_possible_moves[1] = False
                else:
                    movex, movey = choice(moves)
                    self.no_possible_moves[0] = False
                self.board = self.update_board(self.board, movex, movey, player)
                player = -player
                X[move_count, i] = list(chain(*[i[1:7] for i in self.board[1:7]]))
                move_count += 1
            self.board = self.reset_board()
        if to_csv:
            os.mkdir('Data')
            np.savetxt(r'Data\y_sum.csv', y, fmt='%i', delimiter=",")
            for i in range(X.shape[0]):
                np.savetxt(rf'Data\{i}.csv', X[i, :, :], fmt='%i', delimiter=",")
        else:
            return X, y

    @staticmethod
    def train_models():
        os.mkdir('Models')
        y = np.genfromtxt(r'Data\y_sum.csv', delimiter=',', dtype=np.int8)
        progress_bar = tqdm(range(32), total=32, desc='Computing models for every moves... ', unit='moves')
        for move in progress_bar:
            X = np.genfromtxt(rf'Data\{move}.csv', delimiter=',', dtype=np.int8)
            X_non_zero_indices = np.where((np.count_nonzero(X, axis=1) > 0))[0]
            model = Ridge(alpha=1, max_iter=10000)
            model.fit(X[X_non_zero_indices], y[X_non_zero_indices])
            filename = rf'Models\{move}_ridge.sav'
            joblib.dump(model, filename)


if __name__ == '__main__':

    othello = Othello()
    othello.random_simulations(simulations=10000, to_csv=True)
    othello.train_models()

    results = othello.multiprocessing_test_vs_random_bot(simulations=10000)
    print(np.bincount(np.clip(results, -1, 1) + 1) / 10000)
