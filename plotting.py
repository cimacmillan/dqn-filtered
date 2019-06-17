
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import datetime
import os
import torch

def save_sub_result(logger, exp_number, rep_number):
    string_folder = "figures/"
    date_time = str(datetime.datetime.now())
    sub_directory = string_folder + date_time + "/e" + str(exp_number) + "r" + str(rep_number) + "/"

    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)

    write_array(sub_directory + "episode_rewards.txt", logger.episode_rewards)
    write_array(sub_directory + "current_reward_at_tick.txt", logger.current_reward_at_tick)

    write_array(sub_directory + "current_delta_at_tick.txt", logger.current_delta_at_tick)
    write_array(sub_directory + "current_delta_deviation_at_tick.txt",
                logger.current_delta_variance_at_tick)
    write_array(sub_directory + "discard_proportion_at_tick.txt",
                logger.discard_proportion_at_tick)
    write_array(sub_directory + "accepted_frames_at_tick.txt",
                logger.accepted_frames_at_tick)

    write_array(sub_directory + "t_at_log_tick.txt",
                logger.t_at_log_tick)
    write_array(sub_directory + "s_at_log_tick.txt",
                logger.s_at_log_tick)
    write_array(sub_directory + "error_list.txt",
                logger.replay_filter.error_list)

    write_array(sub_directory + "action_list.txt",
                logger.action_performed_at_tick)

    torch.save(logger.Q.state_dict(), sub_directory + "model.model")


def save_results(experiments, experiment_results, set_number):

    string_folder = "figures/"
    date_time = str(datetime.datetime.now())
    directory = string_folder + date_time + "/" + str(set_number) + "/"
    print("Saving Results to: ", directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

    write_manifest(directory + "manifest.txt", experiments)
    output_tables(directory + "data/", experiments, experiment_results)
    plot_all_results(directory + "plots/", experiments, experiment_results)


def output_tables(directory, experiments, experiment_results):

    for experiment_id in range(len(experiments)):
        sub_directory = directory + ("%d/" % experiment_id)
        if not os.path.exists(sub_directory):
            os.makedirs(sub_directory)

        write_array(sub_directory + "episode_rewards.txt", experiment_results[experiment_id].episode_rewards)
        write_array(sub_directory + "current_reward_at_tick.txt", experiment_results[experiment_id].current_reward_at_tick)
        write_array(sub_directory + "current_delta_at_tick.txt", experiment_results[experiment_id].current_delta_at_tick)
        write_array(sub_directory + "current_delta_deviation_at_tick.txt", experiment_results[experiment_id].current_delta_variance_at_tick)
        write_array(sub_directory + "discard_proportion_at_tick.txt",
                    experiment_results[experiment_id].discard_proportion_at_tick)
        write_array(sub_directory + "accepted_frames_at_tick.txt",
                    experiment_results[experiment_id].accepted_frames_at_tick)

        write_array(sub_directory + "t_at_log_tick.txt",
                    experiment_results[experiment_id].t_at_log_tick)
        write_array(sub_directory + "s_at_log_tick.txt",
                    experiment_results[experiment_id].s_at_log_tick)
        write_array(sub_directory + "error_list.txt",
                    experiment_results[experiment_id].replay_filter.error_list)

        torch.save(experiment_results[experiment_id].Q.state_dict(), sub_directory + "model.model")

        write_array(sub_directory + "action_list.txt",
                    experiment_results[experiment_id].action_performed_at_tick)


def write_manifest(file, experiments):
    with open(file, 'w') as f:
        for experiment in experiments:
            f.write("%s\n" % experiment.get_manifest_string())


def write_array(file, array):
    with open(file, 'w') as f:
        for item in array:
            f.write("%s\n" % str(item))


def plot_all_results(directory, experiments, experiment_results):

    if not os.path.exists(directory):
        os.makedirs(directory)

    experiment_rewards = [i.current_reward_at_tick for i in experiment_results]
    experiment_ticks = [[x / 1000 for x in i.t_at_log_tick] for i in experiment_results]
    experiment_seconds = [i.s_at_log_tick for i in experiment_results]
    experiment_mem = [[y / 1000 for y in i.accepted_frames_at_tick] for i in experiment_results]
    replay_memory_utilisation = [0 for i in experiment_results]

    experiment_delta = [i.current_delta_at_tick for i in experiment_results]
    experiment_delta_var = [i.current_delta_variance_at_tick for i in experiment_results]
    experiment_errors = [i.replay_filter.error_list for i in experiment_results]
    experiment_standard_error = [[(exp[x] / np.sqrt(x + 1)) for x in range(len(exp))] for exp in experiment_delta_var]
    experiment_utilisation = [i.discard_proportion_at_tick for i in experiment_results]

    #Detailed Graphs
    plot_experiment_results(directory + ("performance_tick.png"),
                            experiment_rewards, experiments, experiment_ticks, replay_memory_utilisation, "Timestep (10e3)")
    plot_experiment_results(directory + ("performance_seconds.png"),
                            experiment_rewards, experiments, experiment_seconds, replay_memory_utilisation, "Time (s)")
    #Smoothed Graphs
    plot_smooth_experiment_results(directory + ("performance_tick_smoothed.png"),
                                   experiment_rewards, experiments, experiment_ticks, replay_memory_utilisation, "Timestep (10e3)")
    plot_smooth_experiment_results(directory + ("performance_seconds_smoothed.png"),
                                   experiment_rewards, experiments, experiment_seconds, replay_memory_utilisation, "Time (s)")

    #Delta plots
    plot_delta_mean(directory + ("delta_mean.png"), experiments,
                             experiment_delta, experiment_ticks)

    plot_delta_variance(directory + ("delta_deviation.png"), experiments,
                        experiment_delta_var, experiment_ticks)

    plot_delta_standard_error(directory + ("delta_standard error.png"), experiments,
                              experiment_standard_error, experiment_ticks)

    plot_delta_histogram(directory + ("error_histogram.png"), experiments, experiment_errors)

    plot_memory_utilisation(directory + ("experiences_discarded.png"), experiments, experiment_utilisation, experiment_ticks)

    plot_smooth_experiment_results(directory + ("against_memory.png"),
                                   experiment_rewards, experiments, experiment_mem, replay_memory_utilisation, "Experiences Saved (10e3)")


def plot_delta_histogram(string_save_location, experiments, experiment_errors):

    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.hist(experiment_errors[experiment], normed=True, bins=50, color = plot_colour,
                 label=experiments[experiment].get_string(), alpha=.5)


    ax.set_title('Delta Histogram')
    ax.legend(loc='upper left')
    ax.set_xlabel('Delta')
    ax.set_ylabel('Probability')
    fig.tight_layout()
    fig.savefig(string_save_location)


def plot_delta_mean(string_save_location, experiments, delta_means, x_ticks):

    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], delta_means[experiment],
                label=experiments[experiment].get_string(),
                color=plot_colour)

    ax.set_title('Delta Mean')
    ax.legend(loc='upper left')
    ax.set_ylabel('Mean')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(string_save_location)


def plot_delta_variance(string_save_location, experiments, delta_variance, x_ticks):
    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], delta_variance[experiment],
                label=experiments[experiment].get_string(),
                color=plot_colour)

    ax.set_title('Delta Deviation')
    ax.legend(loc='upper left')
    ax.set_ylabel('Deviation')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(string_save_location)


def plot_delta_standard_error(string_save_location, experiments, delta_standard_error, x_ticks):
    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], delta_standard_error[experiment],
                label=experiments[experiment].get_string(),
                color=plot_colour)

    ax.set_title('Delta Standard Error')
    ax.legend(loc='upper left')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(string_save_location)


def plot_memory_utilisation(string_save_location, experiments, proportion, x_ticks):
    fig, ax = plt.subplots(figsize=(5, 3))
    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], proportion[experiment],
                label=experiments[experiment].get_string(),
                color=plot_colour)

    ax.set_title('Experiences discarded')
    ax.legend(loc='upper left')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(string_save_location)


def plot_experiment_results(string_save_location, experiment_rewards, experiments, x_ticks, replay_memory_utilisation, lower_label):

    fig, ax = plt.subplots(figsize=(5, 3))


    for experiment in range(len(experiments)):

        plot_colour = next(ax._get_lines.prop_cycler)['color']

        means = []
        upper_vars = []
        lower_vars = []
        mean_length = 50

        for ep in range(len(experiment_rewards[experiment])):
            mean_sample = experiment_rewards[experiment][ep - mean_length:ep + mean_length]
            mean_value = np.mean(mean_sample)
            variance_value = np.sqrt(np.var(mean_sample)) / 2.0
            means.append(mean_value)
            upper_vars.append(mean_value + variance_value)
            lower_vars.append(mean_value - variance_value)

        ax.plot(x_ticks[experiment], means,
                label=experiments[experiment].get_string(),
                color=plot_colour)
        ax.fill_between(x_ticks[experiment], upper_vars, lower_vars,
                        alpha=0.5,color=plot_colour)

        # plt.axvline(x=x_ticks[experiment][replay_memory_utilisation[experiment]], color=plot_colour, linestyle='--')

    # ax.stackplot(yrs, rng + rnd, labels=[labels, labels, labels])
    ax.set_title('DQN Experiments')
    ax.legend(loc='upper left')
    ax.set_ylabel('Reward')
    ax.set_xlabel(lower_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    fig.savefig(string_save_location)


def plot_smooth_experiment_results(string_save_location, experiment_rewards, experiments, x_ticks, replay_memory_utilisation, lower_label):

    fig, ax = plt.subplots(figsize=(5, 3))


    for experiment in range(len(experiments)):

        plot_colour = next(ax._get_lines.prop_cycler)['color']

        means = []
        upper_vars = []
        lower_vars = []
        mean_length = 200

        for ep in range(len(experiment_rewards[experiment])):
            mean_sample = experiment_rewards[experiment][ep - mean_length:ep + mean_length]
            mean_value = np.mean(mean_sample)
            variance_value = np.sqrt(np.var(mean_sample)) / 2.0
            means.append(mean_value)
            upper_vars.append(mean_value + variance_value)
            lower_vars.append(mean_value - variance_value)

        ax.plot(x_ticks[experiment], means,
                label=experiments[experiment].get_string(),
                color=plot_colour)


    # ax.stackplot(yrs, rng + rnd, labels=[labels, labels, labels])
    ax.set_title('DQN Experiments')
    ax.legend(loc='upper left')
    ax.set_ylabel('Reward')
    ax.set_xlabel(lower_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    fig.savefig(string_save_location)
