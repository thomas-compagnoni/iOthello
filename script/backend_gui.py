from os.path import join, dirname
import joblib
import numpy as np
import warnings

warnings.simplefilter('ignore', UserWarning)


class Othello:

    def __init__(self):
        self.board = np.zeros((6, 6), dtype=np.int8)
        self.board[[2, 3], [3, 2]] = +1
        self.board[[3, 2], [3, 2]] = -1

        self.x_board = (1, 2, 3, 4, 5, 6)
        self.y_board = (1, 2, 3, 4, 5, 6)

        self.x_square = (-1, 0, 1)
        self.y_square = (-1, 0, 1)

        self.models = self.load_models()

    @staticmethod
    def load_models():
        models = []
        for move in range(32):
            filename = join(dirname(__file__), 'models', f'{move}_ridge.sav')
            models.append(joblib.load(filename))
        return models

    def reset_board(self):
        self.board = np.zeros((6, 6), dtype=np.int8)
        self.board[[2, 3], [3, 2]] = 1
        self.board[[3, 2], [3, 2]] = -1

    def find_possible_moves(self, player):
        mask = np.zeros((8, 8))
        mask[1:7, 1:7] = self.board
        board = mask.tolist()
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
                                        moves.add((x_to_evaluate - 1, y_to_evaluate - 1))
                                        break
                                    elif board[x_to_evaluate][y_to_evaluate] == player:
                                        break
                                    else:
                                        x_to_evaluate += xs
                                        y_to_evaluate += ys
        return tuple(moves)

    def update_board(self, move, player):
        row = move[0] + 1
        column = move[1] + 1
        mask = np.zeros((8, 8))
        mask[1:7, 1:7] = self.board
        board = mask
        board[row, column] = player
        for xs in self.x_square:
            for ys in self.y_square:
                if board[row + xs, column + ys] == -player:
                    board_mask = board.copy()
                    x_to_evaluate = row + xs
                    y_to_evaluate = column + ys
                    while 1 <= x_to_evaluate <= 6 and 1 <= y_to_evaluate <= 6:
                        if board_mask[x_to_evaluate, y_to_evaluate] == player:
                            board = board_mask
                            break
                        elif board_mask[x_to_evaluate, y_to_evaluate] == -player:
                            board_mask[x_to_evaluate, y_to_evaluate] = player
                            x_to_evaluate += xs
                            y_to_evaluate += ys
                        else:
                            break
        self.board = board[1:7, 1:7]

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

    def find_best_move_simple(self):
        board = np.zeros((8, 8))
        board[1:7, 1:7] = self.board
        moves, boards = self.next_layer(np.array(board), -1)
        move = moves[boards.sum(axis=2).sum(axis=1).argmin()]
        return [x - 1 for x in move]

    def find_best_move_model(self, move_number):
        board = np.zeros((8, 8))
        board[1:7, 1:7] = self.board
        scores = []
        moves, boards = self.next_layer(np.array(board), -1)
        if move_number < 30:
            boards_second_layer = [self.next_layer(np.array(board), 1, return_moves=False)[:, 1:7, 1:7] for board in boards]
            model = self.models[move_number + 2]  # take the right model given our move number
            for boards in boards_second_layer:
                scores.append(max([model.predict(board.reshape(1, -1))[0] for board in boards]))
        else:
            model = self.models[move_number + 1]
            scores.append([model.predict(board[1:7, 1:7].reshape(1, -1))[0] for board in boards])
        move = moves[scores.index(min(scores))]
        return [x - 1 for x in move]
