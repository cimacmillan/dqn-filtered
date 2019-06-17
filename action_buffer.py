import numpy as np
import random

class ActionBuffer(object):

    def __init__(self, size, frame_history, replay_filter):
        self.size = size
        self.frame_history = frame_history
        self.replay_filter = replay_filter
        self.frame_buffer_index = 0
        self.action_buffer_utilisation = 0
        self.frame_buffer_utilisation = 0

        self.frame_buffer = None
        self.action_buffer = []
        self.action_target = ((1.0 - abs(self.replay_filter.filter_value)) * self.size) - (self.frame_history + 1)


    #Should store the frame in the frame buffer
    def store_frame(self, frame):
        frame = frame.transpose(2, 0, 1)
        if self.frame_buffer is None:
            self.frame_buffer = np.empty([self.size] + list(frame.shape), dtype=np.uint8)

        self.frame_buffer[self.frame_buffer_index] = frame
        self.frame_buffer_index = (self.frame_buffer_index + 1) % self.size
        self.frame_buffer_utilisation = min(self.size, self.frame_buffer_utilisation + 1)



    def get_observation_at_index(self, index):
        begin = index - self.frame_history
        end = index
        recent_frames = []

        for id in range(begin, end):
            index = id % self.frame_buffer_utilisation
            recent_frames.append(self.frame_buffer[index])

        return np.concatenate(recent_frames, 0)

    #Should return (4, 84, 84) array of the last 4 frames
    def get_recent_observation(self):
        return self.get_observation_at_index(self.frame_buffer_index)


    #Check the last 4 frames against the replay filter and possible add
    #Buffer knows the location of S, S+1 etcs
    def possibly_store_effect(self, action, reward, done):

        recent_obs = self.get_recent_observation()
        comparison_a = recent_obs[0]
        comparison_b = recent_obs[self.frame_history - 1]

        should_store_frame = self.replay_filter.should_store_frame(comparison_a, comparison_b, reward)

        if should_store_frame:
            self.store_effect(action, reward, done)

    def store_effect(self, action, reward, done):
        done = 1.0 if done else 0.0
        s_index = self.frame_buffer_index
        s1_index = (self.frame_buffer_index + 1) % self.size
        effect = (s_index, s1_index, action, reward, done)

        self.action_buffer_utilisation = self.action_buffer_utilisation + 1
        self.action_buffer.append(effect)
        self.check_should_remove_to_element()

    def check_should_remove_to_element(self):
        if self.action_buffer_utilisation > self.action_target:
            self.action_buffer.pop(0)
            self.action_buffer_utilisation = self.action_buffer_utilisation - 1


    #Returns a (batch_size, 4, 84, 84) S array,
    #(batch_size) array of actions picked
    #(batch_size) array of rewards given
    #(batch_size, 4, 84, 84) s+1 array
    #(bastch_size) array of done values (1. = done, 0. = running)
    def sample(self, batch_size):

        random_indices = np.random.randint(0, self.action_buffer_utilisation - 1, size=(batch_size))
        action_samples = [self.action_buffer[index] for index in random_indices]

        obs_1 = np.concatenate(
            [self.get_observation_at_index(action[0])[None] for action in action_samples], 0)
        obs_2 = np.concatenate(
            [self.get_observation_at_index(action[1])[None] for action in action_samples], 0)

        actions = np.array([action[2] for action in action_samples])
        rewards = np.array([action[3] for action in action_samples])
        done = np.array([action[4] for action in action_samples])

        return obs_1, actions, rewards, obs_2, done

    #Return 0 if buffer not big enough yet
    def can_sample(self, batch_size):
        return self.action_buffer_utilisation > (batch_size + 1)


    def get_string(self):
        return "Uniformly sampled"

    def get_small_string(self):
        return "True"