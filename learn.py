"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

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
import os

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def to_np(x):
    return x.data.cpu().numpy()


def request_action(t, learning_starts, num_actions, exploration, observations, Q):
    # before learning starts, choose actions randomly
    action = None
    if t < learning_starts:
        action = np.random.randint(num_actions)
    else:
        # epsilon greedy exploration
        sample = random.random()
        threshold = exploration.value(t)
        if sample > threshold:
            obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
            q_value_all_actions = Q(Variable(obs, volatile=True)).cpu()
            action = ((q_value_all_actions).data.max(1)[1])[0]
        else:
            action = np.random.randint(num_actions)

    return action



def dqn_learning(env,
          q_func,
          optimizer_spec,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          batch_size=128,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          experiment_length=2000001,
          action_buffer = ActionBuffer(1000000, 4, ReplayFilter((84, 84), 0, 0, 1, 0.1, keep)),
                 ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    img_h, img_w, img_c = env.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)
    in_channels = input_shape[2]
    num_actions = env.action_space.n

    # Agent objects
    Q = q_func(in_channels, num_actions).type(dtype)
    Q_target = q_func(in_channels, num_actions).type(dtype)
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)
    # action_buffer = ActionBufferNonUniform(replay_buffer_size, frame_history_len, replay_filter)

    # Logging progress
    logger = Logger(learning_starts, exploration, optimizer_spec, action_buffer)

    target_policy_tick = 0
    last_obs = (env.reset(),)

    EXPERIMENT_LENGTH = range(experiment_length)

    for t in EXPERIMENT_LENGTH:
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        for last_ob in last_obs:
            action_buffer.store_frame(last_ob)

        observations = action_buffer.get_recent_observation()

        action = request_action(t, learning_starts, num_actions, exploration, observations, Q)

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

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        action_buffer.possibly_store_effect(action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = (env.reset(),)

        # update last_obs
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                action_buffer.can_sample(batch_size)):

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = action_buffer.sample(batch_size)

            obs_t = Variable(torch.from_numpy(obs_t)).type(dtype) / 255.0
            act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
            rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
            obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype) / 255.0
            done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            q_values = Q(obs_t)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            # get the Q values for best actions in obs_tp1
            # based off frozen Q network
            # max(Q(s', a', theta_i_frozen)) wrt a'
            q_tp1_values = Q_target(obs_tp1).detach()
            q_s_a_prime, a_prime = q_tp1_values.max(1)

            # if current state is end of episode, then there is no next Q value
            q_s_a_prime = (1 - done_mask) * q_s_a_prime

            # Compute Bellman error
            # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
            error = rew_t + gamma * q_s_a_prime - q_s_a

            # clip the error and flip
            clipped_error = -1.0 * error.clamp(-1, 1)

            # backwards pass
            optimizer.zero_grad()

            q_s_a.backward(clipped_error)

            # update
            optimizer.step()
            target_policy_tick += 1

            # update target Q network weights with current Q network weights
            if target_policy_tick % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())

        logger.tick()
        if logger.should_print():
            logger.log_episode_rewards(get_wrapper_by_name(env, "Monitor").get_episode_rewards())
            logger.log_console_progress()
            logger.log_simulation_progress()
            logger.log_action(action)

    logger.save_q(Q)

    return logger


