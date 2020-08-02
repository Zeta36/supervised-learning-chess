import os
from datetime import datetime
from logging import getLogger
from time import time
import chess
import re
from supervised_learning_chess.agent.player_chess import ChessPlayer
from supervised_learning_chess.config import Config
from supervised_learning_chess.env.chess_env import ChessEnv, Winner
from supervised_learning_chess.lib import tf_util
from supervised_learning_chess.lib.data_helper import get_game_data_filenames, write_game_data_to_file, find_pgn_files, read_game_data_from_file, \
    get_next_generation_model_dirs
from supervised_learning_chess.lib.model_helper import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed
import random
from time import sleep
import keras.backend as k
import numpy as np
from keras.optimizers import SGD

from supervised_learning_chess.agent.model_chess import ChessModel, objective_function_for_policy, \
    objective_function_for_value


logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.59)
    return SupervisedLearningWorker(config, env=ChessEnv()).start()


class SupervisedLearningWorker:
    def __init__(self, config: Config, env=None, model=None):
        """

        :param config:
        :param ChessEnv|None env:
        :param supervised_learning_chess.agent.model_chess.ChessModel|None model:
        """
        self.config = config
        self.model = model
        self.env = env     # type: ChessEnv
        self.black = None  # type: ChessPlayer
        self.white = None  # type: ChessPlayer
        self.buffer = []
        self.optimizer = OptimizeWorker(config)

    def start(self):
        if self.model is None:
            self.model = self.load_model()

        self.buffer = []
        idx = 1
        k = 0
        while True:
            start_time = time()
            _ = self.read_game(idx)
            end_time = time()
            logger.debug(
                f"Reading game {idx} time={end_time - start_time} sec")
            if (idx % self.config.play_data.nb_game_in_file) == 0:
                reload_best_model_weight_if_changed(self.model)
            idx += 1
            k += 1
            if k > 100:
                self.optimizer.training()
                k = 0

    def read_game(self, idx):
        self.env.reset()
        self.black = ChessPlayer(self.config, self.model)
        self.white = ChessPlayer(self.config, self.model)
        files = find_pgn_files(self.config.resource.play_data_dir)
        if len(files) > 0:
            random.shuffle(files)
            filename = files[0]
            pgn = open(filename, errors='ignore')
            size = os.path.getsize(filename)
            pos = random.randint(0, size)
            pgn.seek(pos)

            line = pgn.readline()
            offset = 0
            # Parse game headers.
            while line:
                if line.isspace() or line.startswith("%"):
                    line = pgn.readline()
                    continue

                # Read header tags.
                tag_match = TAG_REGEX.match(line)
                if tag_match:
                    offset = pgn.tell()
                    break

                line = pgn.readline()

            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            node = game
            result = game.headers["Result"]
            actions = []
            while not node.is_end():
                next_node = node.variation(0)
                actions.append(node.board().uci(next_node.move))
                node = next_node
            pgn.close()

            k = 0
            observation = self.env.observation
            while not self.env.done and k < len(actions):
                if self.env.board.turn == chess.BLACK:
                    action = self.black.sl_action(observation, actions[k])
                else:
                    action = self.white.sl_action(observation, actions[k])
                board, _ = self.env.step(action)
                observation = board.fen()
                k += 1

            self.env.done = True
            if result == '1-0':
                self.env.winner = Winner.white
            elif result == '0-1':
                self.env.winner = Winner.black
            else:
                self.env.winner = Winner.draw

            self.finish_game()
            self.save_play_data(write=idx %
                                self.config.play_data.nb_game_in_file == 0)
            self.remove_play_data()
        else:
            logger.debug(f"there is no pgn file in the dataset folder!")
        return self.env

    def save_play_data(self, write=True):
        data = self.black.moves + self.white.moves
        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(
            rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        write_game_data_to_file(path, self.buffer)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def finish_game(self):
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0

        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

    def load_model(self):
        from supervised_learning_chess.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: ChessModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None
        self.model = self.load_model()
        self.compile_model()
        self.total_steps = 0

    def training(self):
        min_data_size_to_learn = 1000
        self.load_play_data()

        if self.dataset_size < min_data_size_to_learn:
            logger.info(
                f"dataset_size={self.dataset_size} is less than {min_data_size_to_learn}")
            self.load_play_data()
            return
        logger.debug(
            f"total steps={self.total_steps}, dataset size={self.dataset_size}")

        self.update_learning_rate(self.total_steps)
        steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
        self.total_steps += steps
        save_as_best_model(self.model)

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def update_learning_rate(self, total_steps):
        if total_steps < 100000:
            lr = 1e-2
        elif total_steps < 500000:
            lr = 1e-3
        elif total_steps < 900000:
            lr = 1e-4
        else:
            lr = 2.5e-5
        k.set_value(self.optimizer.lr, lr)
        logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        z_ary = np.concatenate(z_ary_list)
        return state_ary, policy_ary, z_ary

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def load_model(self):
        from supervised_learning_chess.agent.model_chess import ChessModel
        model = ChessModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            model.build()
            save_as_best_model(model)
            logger.debug(f"loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError(f"Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug(f"loading latest model")
            config_path = os.path.join(
                latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(
                latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

    def load_play_data(self):
        filenames = get_game_data_filenames(self.config.resource)
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)
            updated = True

        if updated:
            logger.debug("updating training dataset")
            self.dataset = self.collect_all_loaded_data()

    def load_data_from_file(self, filename):
        try:
            logger.debug(f"loading data from {filename}")
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            logger.warning(str(e))

    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]

    @staticmethod
    def convert_to_training_data(data):
        """

        :param data: format is SelfPlayWorker.buffer
        :return:
        """
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            env = ChessEnv().update(state)

            black_ary, white_ary = env.black_and_white_plane()
            state = [black_ary, white_ary] if env.board.turn == chess.BLACK else [
                white_ary, black_ary]

            state_list.append(state)
            policy_list.append(policy)
            z_list.append(z)

        return np.array(state_list), np.array(policy_list), np.array(z_list)
