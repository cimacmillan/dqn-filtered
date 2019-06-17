

class Experiment(object):
    def __init__(self, replay_memory_size, replay_initialisation_size, repeats, random_seed, action_buffer):
        self.replay_memory_size = replay_memory_size
        self.replay_initialisation_size = replay_initialisation_size
        self.repeats = repeats
        self.random_seed = random_seed
        self.action_buffer = action_buffer

    def get_string(self):
        return 'r:%s, f0 = %s, f1 = %s, un: %s' % (self.replay_memory_size, self.action_buffer.replay_filter.start_filter_value, self.action_buffer.replay_filter.end_filter_value, self.action_buffer.get_small_string())


    def get_manifest_string(self):
        return ("replay_size: %s " % self.replay_memory_size) +\
               ("replay_init: %s " % self.replay_initialisation_size) +\
               ("replay_filter_start: %s " % self.action_buffer.replay_filter.start_filter_value) +\
               ("replay_filter_end: %s " % self.action_buffer.replay_filter.end_filter_value) +\
               ("repeats: %s " % self.repeats) +\
               ("random_seed: %s " % self.random_seed) +\
               ("sampling: %s " % self.action_buffer.get_string())


