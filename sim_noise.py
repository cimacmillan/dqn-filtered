import torch
from torch.autograd import Variable
import gym.spaces
from collections import namedtuple
from utils.schedules import *
from utils.gym_setup import *
from delta import *
from action_buffer import ActionBuffer
from logger import Logger
from filter_functions import *
from model import DQN
from plotting import *
import os

directory = "tmp/sim/"
SET_SEEDS = [88148636, 14188759, 56216764, 71451696, 36454472, 66961291, 47178561, 54902737, 83020963, 71898249]
# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def to_np(x):
    return x.data.cpu().numpy()


def request_action(observations, Q):
    obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
    q_value_all_actions = Q(Variable(obs, volatile=True)).cpu()
    action = ((q_value_all_actions).data.max(1)[1])[0]
    return action


def sim_dqn(env,
          q_func,
          dqn_path,
          action_write_dir,
          frame_history_len=4,
          experiment_length=100,
                 ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)
    in_channels = input_shape[2]
    num_actions = env.action_space.n

    # Agent objects
    Q = q_func(in_channels, num_actions).type(dtype)
    Q.load_state_dict(torch.load(dqn_path, map_location='cpu'))
    Q.eval()
    # action_buffer = ActionBufferNonUniform(replay_buffer_size, frame_history_len, replay_filter)
    target_policy_tick = 0
    last_obs = env.reset()

    EXPERIMENT_LENGTH = range(experiment_length)

    action_buffer=ActionBuffer(100000, frame_history_len, ReplayFilter((84, 84), 0, 0, 1, 1, keep))

    frame_feed_length = experiment_length
    frame_feed_end = 0.5
    frame_feed_start = 0.5
    episode_rewards = []
    episode_frame_feed = []
    episode_reward = 0
    episode_count = 0

    frame_select = 0

    for t in range(10000000):

        action_buffer.store_frame(last_obs)
        observations = action_buffer.get_recent_observation()

        action = request_action(observations, Q)


        # action = 2 right
        # action = 3 left
        # action = 4 right n fire
        # action = 5 left n fire
        # actions_performed.append(action)

        obs, reward, done, info = env.step(action)

        grad = t / experiment_length
        frame_feed = (frame_feed_end * grad) + (frame_feed_start * (1 - grad))
        # if np.random.uniform() < frame_feed:

        # env.render()

        # clipping the reward, noted in nature paper
        # reward = np.clip(reward, -1.0, 1.0)

        # action_buffer.possibly_store_effect(action, reward, done)

        # reset env if reached episode boundary
        if done:
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            obs = env.reset()
            if len(episode_rewards) > episode_count:
                episode_count = len(episode_rewards)
                print(episode_count)
                if episode_count >= experiment_length:
                    break

        # update last_obs
        frame_select += 1
        # if np.random.randint(2) == 0:
        last_obs = obs
    #

    write_array(action_write_dir, get_wrapper_by_name(env, "Monitor").get_episode_rewards())
    write_array(action_write_dir + str("_f"), episode_frame_feed)
    #





# task_id = 6
# model_path = "figures/00_mean_based_n_small_memory/e6r0/model.model"
# seed = SET_SEEDS[2]
# action_write_dir = "sim/action.txt"

def farm_reward(task_id, exp_id, model_path, action_write_dir):
    seed = SET_SEEDS[exp_id]
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[task_id]
    np.random.seed(seed)
    env = get_env(task, seed, directory)
    sim_dqn(env, DQN, model_path, action_write_dir)

print("FARMING 6 space_small_filtered")

# space_small_filtered
farm_reward(6, 0, "figures/00_mean_based_n_small_memory/e6r0/model.model", "sim/reward6_0.txt")
farm_reward(6, 1, "figures/00_mean_based_n_small_memory/e6r1/model.model", "sim/reward6_1.txt")
farm_reward(6, 2, "figures/00_mean_based_n_small_memory/e6r2/model.model", "sim/reward6_2.txt")
farm_reward(6, 3, "figures/00_mean_based_n_small_memory/e6r3/model.model", "sim/reward6_3.txt")
farm_reward(6, 4, "figures/00_mean_based_n_small_memory/e6r4/model.model", "sim/reward6_4.txt")

print("FARMING 8")
# space_small_unfiltered
farm_reward(6, 0, "figures/00_mean_based_n_small_memory/e8r0/model.model", "sim/reward8_0.txt")
farm_reward(6, 1, "figures/00_mean_based_n_small_memory/e8r1/model.model", "sim/reward8_1.txt")
farm_reward(6, 2, "figures/00_mean_based_n_small_memory/e8r2/model.model", "sim/reward8_2.txt")
farm_reward(6, 3, "figures/00_mean_based_n_small_memory/e8r3/model.model", "sim/reward8_3.txt")
farm_reward(6, 4, "figures/00_mean_based_n_small_memory/e8r4/model.model", "sim/reward8_4.txt")

print("FARMING 7")
# space_large_filtered
farm_reward(6, 0, "figures/00_mean_based_n_small_memory/e7r0/model.model", "sim/reward7_0.txt")
farm_reward(6, 1, "figures/00_mean_based_n_small_memory/e7r1/model.model", "sim/reward7_1.txt")
farm_reward(6, 2, "figures/00_mean_based_n_small_memory/e7r2/model.model", "sim/reward7_2.txt")
farm_reward(6, 3, "figures/00_mean_based_n_small_memory/e7r3/model.model", "sim/reward7_3.txt")
farm_reward(6, 4, "figures/00_mean_based_n_small_memory/e7r4/model.model", "sim/reward7_4.txt")

print("FARMING 9")
#
# space_large_unfiltered
farm_reward(6, 0, "figures/00_mean_based_n_small_memory/e9r0/model.model", "sim/reward9_0.txt")
farm_reward(6, 1, "figures/00_mean_based_n_small_memory/e9r1/model.model", "sim/reward9_1.txt")
farm_reward(6, 2, "figures/00_mean_based_n_small_memory/e9r2/model.model", "sim/reward9_2.txt")
farm_reward(6, 3, "figures/00_mean_based_n_small_memory/e9r3/model.model", "sim/reward9_3.txt")
farm_reward(6, 4, "figures/00_mean_based_n_small_memory/e9r4/model.model", "sim/reward9_4.txt")


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
