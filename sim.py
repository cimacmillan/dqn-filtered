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
          experiment_length=10001,
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
    last_obs = (env.reset(),)

    EXPERIMENT_LENGTH = range(experiment_length)

    action_buffer=ActionBuffer(100000, frame_history_len, ReplayFilter((84, 84), 0, 0, 1, 1, keep))

    actions_performed = []

    for t in EXPERIMENT_LENGTH:

        for last_ob in last_obs:
            action_buffer.store_frame(last_ob)

        observations = action_buffer.get_recent_observation()

        # action = request_action(observations, Q)
        # action = 2 right
        # action = 3 left
        # action = 4 right n fire
        # action = 5 left n fire
        actions_performed.append(action)

        obs = []
        reward = None
        done = False
        for repeat in range(1):
            step_obs, step_reward, step_done, info = env.step(action)
            obs.append(step_obs)
            reward = step_reward if reward is None else max(reward, step_reward)
            if step_done:
                done = True
                break

        env.render()

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        action_buffer.possibly_store_effect(action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = (env.reset(),)

        # update last_obs
        last_obs = obs

    write_array(action_write_dir, actions_performed)




# space_small_filtered = logger(
#                 ["00_mean_based_n_small_memory/e6r0/",
#                  "00_mean_based_n_small_memory/e6r1/",
#                  "00_mean_based_n_small_memory/e6r2/",
#                  "00_mean_based_n_small_memory/e6r3/",
#                  "00_mean_based_n_small_memory/e6r4/",
#                 ]
# );
#
# space_large_filtered = logger(
#                 ["00_mean_based_n_small_memory/e7r0/",
#                  "00_mean_based_n_small_memory/e7r1/",
#                  "00_mean_based_n_small_memory/e7r2/",
#                  "00_mean_based_n_small_memory/e7r3/",
#                  "00_mean_based_n_small_memory/e7r4/",
#                 ]
# );
#
# space_small_unfiltered = logger(
#                 ["00_mean_based_n_small_memory/e8r0/",
#                  "00_mean_based_n_small_memory/e8r1/",
#                  "00_mean_based_n_small_memory/e8r2/",
#                  "00_mean_based_n_small_memory/e8r3/",
#                  "00_mean_based_n_small_memory/e8r4/",
#                 ]
# );
#
# space_large_unfiltered = logger(
#                 ["00_mean_based_n_small_memory/e9r0/",
#                  "00_mean_based_n_small_memory/e9r1/",
#                  "00_mean_based_n_small_memory/e9r2/",
#                  "00_mean_based_n_small_memory/e9r3/",
#                  "00_mean_based_n_small_memory/e9r4/",
#                 ]
# );


# task_id = 6
# model_path = "figures/00_mean_based_n_small_memory/e6r0/model.model"
# seed = SET_SEEDS[2]
# action_write_dir = "sim/action.txt"

def farm_actions(task_id, exp_id, model_path, action_write_dir):
    seed = SET_SEEDS[exp_id]
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[task_id]
    env = get_env(task, seed, directory)
    sim_dqn(env, DQN, model_path, action_write_dir)

print("FARMING 6")

farm_actions(6, 0, "figures/00_mean_based_n_small_memory/e6r0/model.model", "sim/action6_0.txt")
farm_actions(6, 1, "figures/00_mean_based_n_small_memory/e6r1/model.model", "sim/action6_1.txt")
farm_actions(6, 2, "figures/00_mean_based_n_small_memory/e6r2/model.model", "sim/action6_2.txt")
farm_actions(6, 3, "figures/00_mean_based_n_small_memory/e6r3/model.model", "sim/action6_3.txt")
farm_actions(6, 4, "figures/00_mean_based_n_small_memory/e6r4/model.model", "sim/action6_4.txt")

print("FARMING 7")

farm_actions(6, 0, "figures/00_mean_based_n_small_memory/e7r0/model.model", "sim/action7_0.txt")
farm_actions(6, 1, "figures/00_mean_based_n_small_memory/e7r1/model.model", "sim/action7_1.txt")
farm_actions(6, 2, "figures/00_mean_based_n_small_memory/e7r2/model.model", "sim/action7_2.txt")
farm_actions(6, 3, "figures/00_mean_based_n_small_memory/e7r3/model.model", "sim/action7_3.txt")
farm_actions(6, 4, "figures/00_mean_based_n_small_memory/e7r4/model.model", "sim/action7_4.txt")

print("FARMING 8")

farm_actions(6, 0, "figures/00_mean_based_n_small_memory/e8r0/model.model", "sim/action8_0.txt")
farm_actions(6, 1, "figures/00_mean_based_n_small_memory/e8r1/model.model", "sim/action8_1.txt")
farm_actions(6, 2, "figures/00_mean_based_n_small_memory/e8r2/model.model", "sim/action8_2.txt")
farm_actions(6, 3, "figures/00_mean_based_n_small_memory/e8r3/model.model", "sim/action8_3.txt")
farm_actions(6, 4, "figures/00_mean_based_n_small_memory/e8r4/model.model", "sim/action8_4.txt")

print("FARMING 9")

farm_actions(6, 0, "figures/00_mean_based_n_small_memory/e9r0/model.model", "sim/action9_0.txt")
farm_actions(6, 1, "figures/00_mean_based_n_small_memory/e9r1/model.model", "sim/action9_1.txt")
farm_actions(6, 2, "figures/00_mean_based_n_small_memory/e9r2/model.model", "sim/action9_2.txt")
farm_actions(6, 3, "figures/00_mean_based_n_small_memory/e9r3/model.model", "sim/action9_3.txt")
farm_actions(6, 4, "figures/00_mean_based_n_small_memory/e9r4/model.model", "sim/action9_4.txt")


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
