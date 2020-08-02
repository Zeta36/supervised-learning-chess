import enum
import chess.pgn
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


class ChessEnv:
    def __init__(self):
        self.board = None
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False

    def reset(self):
        self.board = chess.Board()
        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = chess.Board(board)
        self.turn = self.board.fullmove_number
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def step(self, action):
        """
        :param int|None action, None is resign
        :return:
        """
        if action is None:
            return self.board, {}

        self.board.push_uci(action)

        self.turn += 1

        if self.board.is_game_over() or self.board.can_claim_threefold_repetition():
            self._game_over()

        return self.board, {}

    def _game_over(self):
        self.done = True
        if self.winner is None:
            result = self.board.result()
            if result == '1-0':
                self.winner = Winner.white
            elif result == '0-1':
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def _win_another_player(self):
        if self.board.turn == chess.BLACK:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def black_and_white_plane(self):
        board_state = self.replace_tags()
        board_white = [ord(val) if val.isupper() and val !=
                       "1" else 0 for val in board_state.split(" ")[0]]
        board_white = np.reshape(board_white, (8, 8))
        # Only black plane
        board_black = [ord(val) if val.islower() and val !=
                       "1" else 0 for val in board_state.split(" ")[0]]
        board_black = np.reshape(board_black, (8, 8))

        return board_white, board_black

    def replace_tags(self):
        board_san = self.board.fen()
        board_san = board_san.split(" ")[0]
        board_san = board_san.replace("2", "11")
        board_san = board_san.replace("3", "111")
        board_san = board_san.replace("4", "1111")
        board_san = board_san.replace("5", "11111")
        board_san = board_san.replace("6", "111111")
        board_san = board_san.replace("7", "1111111")
        board_san = board_san.replace("8", "11111111")

        return board_san.replace("/", "")

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()
