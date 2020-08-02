class EvaluateConfig:
    def __init__(self):
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 400
        self.play_config.thinking_loop = 1
        self.play_config.c_puct = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0


class PlayDataConfig:
    def __init__(self):
        self.nb_game_in_file = 100
        self.max_file_num = 200


class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 400
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 2
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.03
        self.change_tau_turn = 10
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 16
        self.prediction_worker_sleep_sec = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001


class TrainerConfig:
    def __init__(self):
        self.batch_size = 256
        self.epoch_to_checkpoint = 2


class ModelConfig:
    cnn_filter_num = 256
    cnn_filter_size = 3
    res_layer_num = 12
    l2_reg = 1e-4
    value_fc_size = 256
