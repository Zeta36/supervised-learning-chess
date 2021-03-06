class EvaluateConfig:
    def __init__(self):
        self.play_config = PlayConfig()
        self.play_config.c_puct = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0


class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100
        self.max_file_num = 10


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 10
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 5
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.03
        self.change_tau_turn = 10
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 4
        self.prediction_worker_sleep_sec = 0.00001
        self.wait_for_expanding_sleep_sec = 0.000001


class TrainerConfig:
    def __init__(self):
        self.batch_size = 32  # 2048
        self.epoch_to_checkpoint = 1


class ModelConfig:
    cnn_filter_num = 16
    cnn_filter_size = 3
    res_layer_num = 1
    l2_reg = 1e-4
    value_fc_size = 16
