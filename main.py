import torch
import torch.optim as optim
import argparse
import copy

from model import DQN
from learn import dqn_learning, OptimizerSpec
from utils.gym_setup import *
from utils.schedules import *
from plotting import *
from experiment import Experiment
from delta import *
from logger import *
from action_buffer_non_uniform import ActionBufferNonUniform
from action_buffer import ActionBuffer
from filter_functions import *
from sampling_functions import *
# Global Variables
#HLC Atari Default
DIMENSION = (84, 84)

# Nature settings
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 10000
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 1000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.0001
EXPLORATION_SCHEDULE = LinearSchedule(100000, 0.02)
DEFAULT_EXPERIMENT = Experiment(replay_memory_size=REPLAY_BUFFER_SIZE,
                                replay_initialisation_size=LEARNING_STARTS,
                                repeats=10,
                                random_seed=False,
                                action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep))
                                )



# Each set is given as an argument to console, this is for BCP
experiment_sets = (
    # (
    #      DEFAULT_EXPERIMENT,
    # ),

    (
        Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBufferNonUniform(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: diagonal(x), "Diag(0, 1)", True)),
    ),
    (
        Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBufferNonUniform(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: diagonalab(x), "Diag(0.5, 1)", True)),
    ),
    (
        Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBufferNonUniform(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: exp(x), "Exp", True)),
    ),

    (
        Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBufferNonUniform(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: diagonal(x), "Diag(0, 1)", False)),
    ),
    (
        Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBufferNonUniform(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: diagonalab(x), "Diag(0.5, 1)", False)),
    ),
    (
        Experiment(replay_memory_size=REPLAY_BUFFER_SIZE, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBufferNonUniform(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN, ReplayFilter(DIMENSION, 0.1, 0.1, 1, 0.1, keep), lambda x: exp(x), "Exp", False)),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                           ReplayFilter(DIMENSION, 0.02, 0.02, 1, 2,
                                                                                        smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.04, 0.04, 1, 2,
                                                                              smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.06, 0.06, 1, 2,
                                                                              smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.08, 0.08, 1, 2,
                                                                              smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.1, 0.1, 1, 2,
                                                                              smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.2, 0.2, 1, 2,
                                                                              smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.02, 0.02, 1, 2,
                                                                              most_common_keep_filter))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.04, 0.04, 1, 2,
                                                                              most_common_keep_filter))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.06, 0.06, 1, 2,
                                                                              most_common_keep_filter))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.08, 0.08, 1, 2,
                                                                              most_common_keep_filter))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.1, 0.1, 1, 2,
                                                                              most_common_keep_filter))),
    ),

    (
        Experiment(replay_memory_size=100000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0.2, 0.2, 1, 2,
                                                                              most_common_keep_filter))),
    ),



    (
        Experiment(replay_memory_size=10000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0, 0.1, 2000000, 0.1,
                                                                              smallest_remove))),
    ),

    (
        Experiment(replay_memory_size=1000000, replay_initialisation_size=LEARNING_STARTS, repeats=5,
                   random_seed=False, action_buffer=ActionBuffer(REPLAY_BUFFER_SIZE, FRAME_HISTORY_LEN,
                                                                 ReplayFilter(DIMENSION, 0, 0.1, 2000000, 0.1,
                                                                              smallest_remove))),
    ),

)
EXPERIMENT_LENGTH = 4000000
SET_SEEDS = [88148636, 14188759, 56216764, 71451696, 36454472, 66961291, 47178561, 54902737, 83020963, 71898249]
THREADS = 16


def atari_learn(task, env_id, num_timesteps, experiment_sets, experiment_id, rep_off=None, roffset=0):

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    if not os.path.exists("figures"):
        os.makedirs("figures")

    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE)
    )

    if roffset is None:
        roffset = 0

    print("Experiment ID: ", experiment_id, " Repeat: ", rep_off, " Offset: ", roffset);

    if rep_off is None:

        for experiment in experiment_sets[experiment_id]:

            for exp_id in range(roffset, experiment.repeats):

                seed = np.random.randint(0, 100000) if experiment.random_seed else SET_SEEDS[exp_id]
                np.random.seed(seed)

                print(seed)

                vid_dir_name = (task.env_id + str(experiment_id) + "-" + str(exp_id))

                env = get_env(task, seed, vid_dir_name)
                logger = dqn_learning(
                    env=env,
                    q_func=DQN,
                    optimizer_spec=optimizer,
                    exploration=EXPLORATION_SCHEDULE,
                    stopping_criterion=stopping_criterion,
                    batch_size=BATCH_SIZE,
                    gamma=GAMMA,
                    learning_starts=experiment.replay_initialisation_size,
                    learning_freq=LEARNING_FREQ,
                    frame_history_len=FRAME_HISTORY_LEN,
                    target_update_freq=TARGET_UPDATE_FREQ,
                    experiment_length=EXPERIMENT_LENGTH,
                    action_buffer=copy.deepcopy(experiment.action_buffer) #Copying so repeats to mess with filter
                )
                save_sub_result(logger, experiment_id, exp_id)


    else: # Special case for training a single experiment on BCP
        experiment = experiment_sets[experiment_id][0]
        seed = SET_SEEDS[rep_off]
        np.random.seed(seed)
        env = get_env(task, seed, task.env_id)
        logger = dqn_learning(
            env=env,
            q_func=DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=experiment.replay_initialisation_size,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            experiment_length=EXPERIMENT_LENGTH,
            action_buffer=copy.deepcopy(experiment.action_buffer)  # Copying so repeats to mess with filter
        )
        experiment_results = [logger]
        save_results(experiment_sets[experiment_id], experiment_results, experiment_id)

    # env.close()

    print("-----------------------------------")
    print("-Experiments Finished Successfully-")
    print("-----------------------------------")


def main():


    parser = argparse.ArgumentParser(description='RL agents for atari')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train an RL agent for atari games")
    train_parser.add_argument("--task-id", type=int, required=True, help="0 = BeamRider, 1 = Breakout, 2 = Enduro, 3 = Pong, 4 = Qbert, 5 = Seaquest, 6 = Spaceinvaders")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--exp", type=int, default=None, help="ID of experiment to be run")
    train_parser.add_argument("--rep", type=int, default=None, help="Repeat to be ran")
    train_parser.add_argument("--roffset", type=int, default=None, help="Repeat offset")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("CUDA Available, using GPU")
        torch.cuda.set_device(0)
        print("Device count - ", torch.cuda.device_count())
        for id in range(torch.cuda.device_count()):
            print("Device ", id, ": ", torch.cuda.get_device_name(id))
    else:
        print("CUDA UNAVAILABLE, using CPU")
        torch.set_num_threads(THREADS)

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    # 0 = BeamRider
    # 1 = Breakout
    # 2 = Enduro
    # 3 = Pong
    # 4 = Qbert
    # 5 = Seaquest
    # 6 = Spaceinvaders
    for i in benchmark.tasks:
        print(i)
    task = benchmark.tasks[args.task_id]

    # Run training
    atari_learn(task, task.env_id, num_timesteps=task.max_timesteps, experiment_sets=experiment_sets, experiment_id=args.exp, rep_off=args.rep, roffset=args.roffset)


if __name__ == '__main__':
    main()
