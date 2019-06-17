import numpy as np
from scipy.stats import norm

class ReplayFilter(object):

    def __init__(self, dimension, start_filter_value, end_filter_value, range, reward_clip, filter_function):
        self.start_filter_value = start_filter_value
        self.end_filter_value = end_filter_value
        self.filter_value = start_filter_value
        self.range = range
        self.rolling_mean = 0
        self.rolling_variance = 0
        self.rolling_n = 0
        self.discarded_frames = 0
        self.accepted_frames = 0
        self.deviation_array = []
        self.error_list = []
        self.count = 0
        self.modulo = 100
        self.reward_clip = reward_clip
        self.filter_function = filter_function


    def calculate_rmse(self, a, b):
        frame_a = a.ravel() / 255.0
        frame_b = b.ravel() / 255.0
        mse = ((frame_a - frame_b) ** 2).mean()
        rmse = np.sqrt(mse)
        return rmse


    def should_store_frame(self, last_frame, current_frame, reward):

        gradient = min(self.count / self.range, 1.0)
        self.filter_value = ((1 - gradient) * self.start_filter_value) + (gradient * self.end_filter_value)
        self.count = self.count + 1

        if abs(reward) > self.reward_clip:
            return True

        error = self.calculate_rmse(last_frame, current_frame)

        if self.count % self.modulo == 0:
            self.error_list.append(error)

        if self.rolling_n == 0:

            self.rolling_n = 1
            self.rolling_mean = error
            self.rolling_variance = 0

            return True

        else:

            should_store = self.filter_function(self.rolling_mean, self.rolling_variance, self.rolling_n, error, self.filter_value)

            if should_store:
                self.accepted_frames = self.accepted_frames + 1
            else:
                self.discarded_frames = self.discarded_frames + 1

            new_n = self.rolling_n + 1
            new_mean = self.rolling_mean + ((error - self.rolling_mean) / new_n)
            new_variance = self.rolling_variance + ((error - self.rolling_mean) * (error - new_mean))
            self.rolling_n = new_n
            self.rolling_mean = new_mean
            self.rolling_variance = new_variance

            return should_store >= self.filter_value
