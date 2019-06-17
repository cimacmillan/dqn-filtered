import numpy as np

root = "/Users/callum/Workplace/workrepo/uniwork/diss/code/figures/"

def logger(x, cut):
    return readAndAverageAll([(root + y) for y in x], cut)


def readAndAverageAll(setOfDirectories, cut):
    loggers = [readDat(x, cut) for x in setOfDirectories]
    return average_loggers(loggers, cut)


def average_list(list):
    min_list = np.min([len(l) for l in list])
    shortened = [l[:min_list] for l in list]
    return np.average(np.array(shortened), axis=0)


# Injects average into first logger
def average_loggers(loggers, cut):

    loggers[0].repeat_rewards = [logger.current_reward_at_tick for logger in loggers]
    loggers[0].repeat_actions = [logger.action_lists for logger in loggers]

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

    def __init__(self):
        self.current_reward_at_tick = []
        self.current_delta_at_tick = []
        self.current_delta_variance_at_tick = []
        self.discard_proportion_at_tick = []
        self.accepted_frames_at_tick = []
        self.error_list = []
        self.s_at_log_tick = []
        self.t_at_log_tick = []
        self.repeat_rewards = []
        self.repeat_actions = []
        self.action_lists = []


import os
import re

# prog = re.compile("\D*(?P<action>\d+)\D*")
prog = re.compile("\D*tensor\((?P<action>\d+)\)*\D*")

def readIntArray(file):
    text_file = open(file, "r")
    lines = text_file.readlines()
    text_file.close()
    return [int(x) for x in lines]

def readFloatArray(file):
    text_file = open(file, "r")
    lines = text_file.readlines()
    text_file.close()
    return [float(x) for x in lines]

def readActionArray(file):
    text_file = open(file, "r")
    lines = text_file.readlines()
    text_file.close()
    actions = [];

    for line in lines:
        match = prog.match(line);
        if match is not None:
            actions.append(int(match.group("action")))

    return actions;

def expand_reward_array(rewards, length):
    reward_at_tick = []
    otherLength = len(rewards)
    for i in range(length):
        index = int((i / length) * otherLength)
        reward_at_tick.append(rewards[index])
    return reward_at_tick


def readDat(directory, cut):
    logger = Logger()
    logger.accepted_frames_at_tick = readFloatArray(directory + "accepted_frames_at_tick.txt")[:cut]
    logger.discard_proportion_at_tick = readFloatArray(directory + "discard_proportion_at_tick.txt")[:cut]
    logger.current_delta_variance_at_tick = readFloatArray(directory + "current_delta_deviation_at_tick.txt")[:cut]
    logger.current_delta_at_tick = readFloatArray(directory + "current_delta_at_tick.txt")[:cut]
#     logger.current_reward_at_tick = expand_reward_array(readFloatArray(directory + "episode_rewards.txt"), len(logger.accepted_frames_at_tick))
    logger.current_reward_at_tick = readFloatArray(directory + "current_reward_at_tick.txt")[:cut]

#     logger.error_list = readFloatArray(directory + "error_list.txt")
    logger.t_at_log_tick = readFloatArray(directory + "t_at_log_tick.txt")[:cut]
    logger.s_at_log_tick = readFloatArray(directory + "s_at_log_tick.txt")[:cut]
    logger.repeat_rewards = [logger.current_reward_at_tick]

    try:
        logger.action_lists = readActionArray(directory + "action_list.txt")
        logger.repeat_actions =  [logger.action_lists]
    # Store configuration file values
    except FileNotFoundError:
        logger.action_lists = []
        # Keep preset values

    return logger;
