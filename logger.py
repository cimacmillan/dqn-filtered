
import numpy as np
import sys
import time
import os


def average_list(list):
    min_list = np.min([len(l) for l in list])
    shortened = [l[:min_list] for l in list]
    return np.average(np.array(shortened), axis=0)


# Injects average into first logger
def average_loggers(loggers):
    loggers[0].repeat_rewards = [logger.current_reward_at_tick for logger in loggers]

    sub_reward_results = [logger.current_reward_at_tick for logger in loggers]
    sub_delta_results = [logger.current_delta_at_tick for logger in loggers]
    sub_delta_variance_results = [logger.current_delta_variance_at_tick for logger in loggers]
    sub_discard_proportion_at_tick = [logger.discard_proportion_at_tick for logger in loggers]
    sub_accepted_frames_at_tick = [logger.accepted_frames_at_tick for logger in loggers]

    loggers[0].current_reward_at_tick = average_list(sub_reward_results)
    loggers[0].current_delta_at_tick = average_list(sub_delta_results)
    loggers[0].current_delta_variance_at_tick = average_list(sub_delta_variance_results)
    loggers[0].discard_proportion_at_tick = average_list(sub_discard_proportion_at_tick)
    loggers[0].accepted_frames_at_tick = average_list(sub_accepted_frames_at_tick)

    return loggers[0]

class Logger(object):

    def __init__(self, learning_starts, exploration, optimizer_spec, action_buffer):
        self.t = 0
        self.log_every_n_steps = 1000
        self.best_mean_episode_reward = None
        self.mean_episode_reward = None
        self.episode_rewards = []
        self.learning_starts = learning_starts
        self.exploration = exploration
        self.optimizer_spec = optimizer_spec
        self.action_buffer = action_buffer
        self.replay_filter = action_buffer.replay_filter
        self.current_reward_at_tick = []
        self.t_at_log_tick = []
        self.s_at_log_tick = []
        self.beginning_time = time.time()
        self.current_delta_at_tick = []
        self.current_delta_variance_at_tick = []
        self.discard_proportion_at_tick = []
        self.accepted_frames_at_tick = []
        self.action_performed_at_tick = []
        self.Q = None


    def tick(self):
        self.t += 1

    def should_print(self):
        return self.t % self.log_every_n_steps == 0 and self.t != 0 and self.t >= self.log_every_n_steps

    def log_episode_rewards(self, episode_rewards):
        self.episode_rewards = episode_rewards
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
            self.best_mean_episode_reward = self.mean_episode_reward if self.best_mean_episode_reward is None \
                else max(self.best_mean_episode_reward, self.mean_episode_reward)

    def log_console_progress(self):
        print("---------------------------------")
        print("Timestep %d" % (self.t,))
        print("learning started? %d" % (self.t > self.learning_starts))
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)
        print("episodes %d" % len(self.episode_rewards))
        print("exploration %f" % self.exploration.value(self.t))
        print("learning_rate %f" % self.optimizer_spec.kwargs['lr'])
        print("replay memory frame utilisiation %d / %d"
              % (self.action_buffer.frame_buffer_utilisation, self.action_buffer.size))
        print("replay memory action utilisiation %d / %d"
              % (self.action_buffer.action_buffer_utilisation, self.action_buffer.action_target))
        print("replay filter proportion: %f" % self.replay_filter.filter_value)

        print("action size: ", len(self.action_buffer.action_buffer))

        if self.replay_filter.rolling_n > 0:
            print("replay filter discarded proportion: %f"
                  % (self.replay_filter.discarded_frames / self.replay_filter.rolling_n))

        sys.stdout.flush()


    def log_simulation_progress(self):
        self.current_reward_at_tick.append(self.episode_rewards[-1])
        self.t_at_log_tick.append(self.t)
        self.s_at_log_tick.append(int(time.time() - self.beginning_time))
        self.current_delta_at_tick.append(self.replay_filter.rolling_mean)
        self.current_delta_variance_at_tick.append(np.sqrt(self.replay_filter.rolling_variance / self.replay_filter.rolling_n))
        self.discard_proportion_at_tick.append(self.replay_filter.discarded_frames / self.replay_filter.rolling_n)
        self.accepted_frames_at_tick.append(self.replay_filter.accepted_frames)


    def log_action(self, action):
        self.action_performed_at_tick.append(action)

    def save_q(self, Q):
        self.Q = Q


    # ### 4. Log progress
    # if t % SAVE_MODEL_EVERY_N_STEPS == 0:
    #     if not os.path.exists("models"):
    #         os.makedirs("models")
    #     add_str = ''
    #     model_save_path = "models/%s_%s_%d_%s.model" %(str(env_id), add_str, t, str(time.ctime()).replace(' ', '_'))
    #     torch.save(Q.state_dict(), model_save_path)

