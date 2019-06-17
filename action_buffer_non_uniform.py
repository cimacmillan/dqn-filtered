import numpy as np
from heapq import merge

class ActionBufferNonUniform(object):

    def __init__(self, size, frame_history, replay_filter, function, function_name, mean_based):
        self.size = size
        self.frame_history = frame_history
        self.replay_filter = replay_filter
        self.frame_buffer_index = 0
        self.action_buffer_utilisation = 0
        self.frame_buffer_utilisation = 0

        self.frame_buffer = None
        self.action_buffer_chronological = []
        self.action_target = int(((1.0 - abs(self.replay_filter.filter_value)) * self.size) - (self.frame_history + 1))
        self.action_buffer_sorted = []
        self.action_buffer_to_sort = []

        self.function_name = function_name
        self.timestep = 0
        self.function = function
        self.mean_based = mean_based


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

        delta = self.replay_filter.calculate_rmse(comparison_a, comparison_b)

        should_store_frame = self.replay_filter.should_store_frame(comparison_a, comparison_b, reward)

        if should_store_frame:
            self.store_effect(action, reward, done, delta)

    def store_effect(self, action, reward, done, delta):
        done = 1.0 if done else 0.0
        s_index = self.frame_buffer_index
        s1_index = (self.frame_buffer_index + 1) % self.size

        if self.mean_based:
            delta = 1.0 / (abs(delta - self.replay_filter.rolling_mean) + 1)

        if abs(reward) > 0.1:
            delta = delta + 1

        effect = (s_index, s1_index, action, reward, done, delta, self.timestep)
        self.timestep += 1
        self.action_buffer_utilisation = self.action_buffer_utilisation + 1
        self.action_buffer_chronological.append(effect)
        self.action_buffer_to_sort.append(effect)


    def check_should_remove_to_element(self):
        if self.action_buffer_utilisation > self.action_target:
            amountToRemove = self.action_buffer_utilisation - self.action_target
            for i in range(amountToRemove):
                object = self.action_buffer_chronological[0]
                self.action_buffer_sorted.remove(object)
                self.action_buffer_chronological.pop(0)

            self.action_buffer_utilisation = self.action_target


    def merge_into(self, l, m):
        result = []
        i = j = 0
        total = len(l) + len(m)

        for k in range(total):

            if len(l) == i:
                result += m[j:]

                break
            elif len(m) == j:
                result += l[i:]
                break
            elif l[i][5] < m[j][5]:
                result.append(l[i])
                i += 1
            else:
                result.append(m[j])
                j += 1

        return result


    def sample(self, batch_size):

        self.action_buffer_sorted = self.merge_into(self.action_buffer_sorted, self.action_buffer_to_sort)
        self.action_buffer_to_sort.clear()

        self.check_should_remove_to_element()

        action_buffer_size = self.action_buffer_utilisation - 1

        random_floats = np.random.uniform(size=batch_size)
        action_samples = [self.action_buffer_sorted[int(self.function(x) * action_buffer_size)] for x in random_floats]

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
        return ("Non-Uniformly sampled " + self.function_name)

    def get_small_string(self):
        return "False"
