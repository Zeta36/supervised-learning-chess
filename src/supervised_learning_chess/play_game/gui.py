from logging import getLogger

import chess
from supervised_learning_chess.config import Config, PlayWithHumanConfig
from supervised_learning_chess.play_game.game_model import PlayWithHuman
from supervised_learning_chess.env.chess_env import ChessEnv
from random import random

logger = getLogger(__name__)


def start(config: Config):
    PlayWithHumanConfig().update_play_config(config.play)
    chess_model = PlayWithHuman(config)

    env = ChessEnv().reset()
    human_is_black = random() < 0.5
    chess_model.start_game(human_is_black)

    while not env.done:
        if (env.board.turn == chess.BLACK) == human_is_black:
            action = chess_model.move_by_human(env)
            print("You move to: " + action)
        else:
            action = chess_model.move_by_ai(env)
            print("AI moves to: " + action)
        board, _ = env.step(action)
        env.render()
        print("Board FEN = " + board.fen())

    print("\nEnd of the game.")
    print("Game result:")
    print(env.board.result())
