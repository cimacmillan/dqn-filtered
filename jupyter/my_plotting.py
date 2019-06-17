import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import os

size = (10, 3)


def mean_avg(ys):
    result = []
    for i in range(len(ys)):
        sample = ys[max(0, i - 100):(i + 1)]
        mean = np.array(sample).mean();
        result.append(mean)

    return result

def plot_avg_bar_charts(loggers, labels, xlabel, ylabel, multiplication, x_labels):
    fig, ax = plt.subplots(figsize=size)
#     ax.axhline(0, color="gray")
#     ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
#         transform=ax.get_yaxis_transform())

    baseline_rewards = np.array([i for i in loggers[0][0].repeat_rewards])
    baseline_means = baseline_rewards.mean(axis = 0)


    index = np.arange(len(loggers[0][1:]))
    bar_width = (1 / (len(loggers))) * 0.8

    minimum = 0
    maximum = 0


    for ids in range(len(loggers)):
        ex = loggers[ids]
        ys = []
        for i in ex[1:]:
            experiment_rewards = np.array([x for x in i.repeat_rewards])
            experiment_means = experiment_rewards.mean(axis = 0)
            diff = (experiment_means - baseline_means)
            sum_diff = diff.mean()
            ys.append((sum_diff * multiplication))


        ax.bar(index + (ids * bar_width), ys, bar_width, label=labels[ids + 1]);
        minimum = min(min(ys), minimum)
        maximum = max(max(ys), maximum)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(index + ((bar_width / 2) * (len(loggers) - 1)))
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.axhline(0, color="gray")
    ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
        transform=ax.get_yaxis_transform())

    diff = (maximum - minimum);
    minimum = minimum - diff
    maximum = maximum + diff

    ax.set_ylim([minimum,maximum])


def plot_avg_diff_per_exp_over_time(loggers, labels, xlabel, ylabel, multiplication):
    fig, ax = plt.subplots(figsize=size)
#     ax.axhline(0, color="gray")
#     ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
#         transform=ax.get_yaxis_transform())

    baseline_rewards = np.array([i for i in loggers[0][0].repeat_rewards])
    baseline_means = baseline_rewards.mean(axis = 0)

    for ids in range(len(loggers)):
        ex = loggers[ids]

        ys = []
        for i in ex[1:]:
            experiment_rewards = np.array([x for x in i.repeat_rewards])
            experiment_means = experiment_rewards.mean(axis = 0)
            diff = mean_avg(experiment_means - baseline_means)
            ys.append(diff);


        x_ticks = np.array(loggers[0][0].t_at_log_tick) / 1000
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        for y in range(len(ys)):
            plot_colour = next(ax._get_lines.prop_cycler)['color']
            ax.plot(x_ticks, ys[y], label=labels[y + 1], color=plot_colour)
#         ax.bar(index + (ids * bar_width), ys, bar_width, label=labels[ids + 1]);

    ax.axhline(0, color="gray")
    ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
        transform=ax.get_yaxis_transform())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

def plt_repeat_diff_over_time(loggers, labels, xlabel, ylabel, multiplication):
    fig, ax = plt.subplots(figsize=size)

    baseline_rewards = np.array([i for i in loggers[0].repeat_rewards])
    baseline_means = baseline_rewards.mean(axis = 0)
    for i in range(len(loggers[1:])):
        experiment_rewards = np.array([x for x in loggers[i + 1].repeat_rewards])
        experiment_diffs = (np.array([x - baseline_means for x in experiment_rewards]))

        plot_colour = next(ax._get_lines.prop_cycler)['color']
#         experiment_means = experiment_rewards.mean(axis = 0)
#         diff = (experiment_means - baseline_means)
        x_ticks = np.array(loggers[0].t_at_log_tick) / 1000

        for rep in range(len(experiment_diffs)):
            ys = mean_avg(experiment_diffs[rep])
            if rep == 0:
                ax.plot(x_ticks, ys, color=plot_colour, label=labels[i + 1])
            else:
                ax.plot(x_ticks, ys, color=plot_colour)


#         sum_diff = (diff.sum() / len(i.repeat_rewards)) / (np.array(i.accepted_frames_at_tick).mean())
#         ys.append((sum_diff * multiplication))


    ax.axhline(0, color="gray")
    ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
    transform=ax.get_yaxis_transform())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()



def plot_avg_diff_over_time(loggers, labels, xlabel, ylabel, multiplication):
    fig, ax = plt.subplots(figsize=size)
#     ax.axhline(0, color="gray")
#     ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
#         transform=ax.get_yaxis_transform())

    baseline_rewards = np.array([i for i in loggers[0][0].repeat_rewards])
    baseline_means = baseline_rewards.mean(axis = 0)

    for ids in range(len(loggers)):
        ex = loggers[ids]

        ys = []
        for i in ex[1:]:
            experiment_rewards = np.array([x for x in i.repeat_rewards])
            experiment_means = experiment_rewards.mean(axis = 0)
            diff = (experiment_means - baseline_means)
            ys.append(diff);


        mean_differences = mean_avg(np.array(ys).mean(axis=0))
        x_ticks = np.array(loggers[0][0].t_at_log_tick) / 1000
        ax.plot(x_ticks, mean_differences, label=labels[ids + 1])
#         ax.bar(index + (ids * bar_width), ys, bar_width, label=labels[ids + 1]);

    ax.axhline(0, color="gray")
    ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
        transform=ax.get_yaxis_transform())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def plt_repeat_bar_charts(loggers, labels, xlabel, ylabel, multiplication):
    fig, ax = plt.subplots(figsize=size)
    ax.axhline(0, color="gray")
    ax.text(1.02, 0, "Baseline", va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
        transform=ax.get_yaxis_transform())

    index = np.arange(len(loggers[0].repeat_rewards))
    bar_width = 0.35

    minimum = 0
    maximum = 0

    baseline_rewards = np.array([i for i in loggers[0].repeat_rewards])
    baseline_means = baseline_rewards.mean(axis = 0)
    for i in range(len(loggers[1:])):
        experiment_rewards = np.array([x for x in loggers[i + 1].repeat_rewards])
        experiment_diffs = [x - baseline_means for x in experiment_rewards]
        experiment_means = [x.mean() for x in experiment_diffs]

        minimum = min(minimum, min(experiment_means))
        maximum = max(maximum, max(experiment_means))

        ax.bar(index + bar_width * i, experiment_means, bar_width, label=labels[i + 1]);

#         sum_diff = (diff.sum() / len(i.repeat_rewards)) / (np.array(i.accepted_frames_at_tick).mean())
#         ys.append((sum_diff * multiplication))


    bar_labels = [("r(" + str(x) + ")") for x in range(5)]
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(bar_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    diff = (maximum - minimum);
    minimum = minimum - diff / 2
    maximum = maximum + diff / 2
    ax.set_ylim([minimum,maximum])


#         overall_score += sum_diff

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def plot_all_results(directory, experiments, experiment_results, labels):

    if not os.path.exists(directory):
        os.makedirs(directory)


    experiment_rewards = [i.current_reward_at_tick for i in experiment_results]
    experiment_ticks = [[x / 1000 for x in i.t_at_log_tick] for i in experiment_results]
    experiment_seconds = [i.s_at_log_tick for i in experiment_results]
#     experiment_ticks = [[x for x in range(len(i.current_reward_at_tick))] for i in experiment_results]
#     experiment_seconds = [[x for x in range(len(i.current_reward_at_tick))] for i in experiment_results]
    experiment_mem = [[y / 1000 for y in i.accepted_frames_at_tick] for i in experiment_results]
    replay_memory_utilisation = [0 for i in experiment_results]

    experiment_delta = [i.current_delta_at_tick for i in experiment_results]
    experiment_delta_var = [i.current_delta_variance_at_tick for i in experiment_results]
    experiment_errors = [i.error_list for i in experiment_results]
#     experiment_errors = [0 for i in experiment_results]
    experiment_standard_error = [[(exp[x] / np.sqrt(x + 1)) for x in range(len(exp))] for exp in experiment_delta_var]
    experiment_utilisation = [i.discard_proportion_at_tick for i in experiment_results]

    repeat_rewards = [[x for x in i.repeat_rewards] for i in experiment_results]

#     #Detailed Graphs
#     plot_experiment_results(directory + ("performance_tick.png"),
#                             experiment_rewards, experiments, experiment_ticks, replay_memory_utilisation, "Timestep (10e3)", labels)
#     plot_experiment_results(directory + ("performance_seconds.png"),
#                             experiment_rewards, experiments, experiment_seconds, replay_memory_utilisation, "Time (s)", labels)
#     #Smoothed Graphs
#     plot_smooth_experiment_results(directory + ("performance_tick_smoothed.png"),
#                                    experiment_rewards, experiments, experiment_ticks, replay_memory_utilisation, "Timestep (10e3)", labels)
#     plot_smooth_experiment_results(directory + ("performance_seconds_smoothed.png"),
#                                    experiment_rewards, experiments, experiment_seconds, replay_memory_utilisation, "Time (s)", labels)

#     #Delta plots
    plot_delta_mean(directory + ("delta_mean.png"), experiments,
                             experiment_delta, experiment_ticks, labels)

#     plot_delta_variance(directory + ("delta_deviation.png"), experiments,
#                         experiment_delta_var, experiment_ticks, labels)

#     plot_delta_standard_error(directory + ("delta_standard error.png"), experiments,
#                               experiment_standard_error, experiment_ticks, labels)

#     plot_delta_histogram(directory + ("error_histogram.png"), experiments, experiment_errors, labels)

    plot_memory_utilisation(directory + ("experiences_discarded.png"), experiments, experiment_utilisation, experiment_ticks, labels)

#     plot_smooth_experiment_results(directory + ("against_memory.png"),
#                                    experiment_rewards, experiments, experiment_mem, replay_memory_utilisation, "Experiences Saved (10e3)", labels)



    plot_smooth_var_experiment_results(directory + ("performance_seconds_smoothed_var.png"),
                                   experiment_rewards, experiments, experiment_ticks, replay_memory_utilisation, "Timestep (10e3)", labels, repeat_rewards)

    # plot_smooth_var_experiment_results(directory + ("performance_seconds_smoothed_var.png"),
    #                            experiment_rewards, experiments, experiment_mem, replay_memory_utilisation, "Experiences Saved (10e3)", labels, repeat_rewards)

    # plot_smooth_var_experiment_results(directory + ("performance_seconds_smoothed_var.png"),
    #                            experiment_rewards, experiments, experiment_seconds, replay_memory_utilisation, "Time (s)", labels, repeat_rewards)


    # plot_smooth_exp_results_per_experiment(directory + ("performance_seconds_smoothed_var.png"),
    #                            experiment_rewards, experiments, experiment_mem, replay_memory_utilisation, "Experiences Saved (10e3)", labels, repeat_rewards)
    #


    plot_smooth_exp_results_per_experiment(directory + ("performance_seconds_smoothed_var.png"),
                                   experiment_rewards, experiments, experiment_ticks, replay_memory_utilisation, "Timestep (10e3)", labels, repeat_rewards)


    if experiment_results[0].repeat_actions is not None:
        action_lists = [i.repeat_actions for i in experiment_results]
        # plot_action_histogram(labels, action_lists)
        # plot_action_time_graph(labels, action_lists)

def plot_delta_histogram(string_save_location, experiments, experiment_errors, labels):

    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.hist(experiment_errors[experiment], normed=True, bins=50, color = plot_colour,
                 label=labels[experiment], alpha=.5)


    ax.set_title('Delta Histogram')
    ax.legend(loc='upper left')
    ax.set_xlabel('Delta')
    ax.set_ylabel('Probability')
    fig.tight_layout()
#     fig.savefig(string_save_location)


def plot_delta_mean(string_save_location, experiments, delta_means, x_ticks, labels):

    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], delta_means[experiment],
                label=labels[experiment],
                color=plot_colour)

    ax.set_title('Delta Mean')
    ax.legend(loc='upper left')
    ax.set_ylabel('Mean')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
#     fig.savefig(string_save_location)


def plot_delta_variance(string_save_location, experiments, delta_variance, x_ticks, labels):
    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], delta_variance[experiment],
                label=labels[experiment],
                color=plot_colour)

    ax.set_title('Delta Deviation')
    ax.legend(loc='upper left')
    ax.set_ylabel('Deviation')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
#     fig.savefig(string_save_location)


def plot_delta_standard_error(string_save_location, experiments, delta_standard_error, x_ticks, labels):
    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], delta_standard_error[experiment],
                label=labels[experiment],
                color=plot_colour)

    ax.set_title('Delta Standard Error')
    ax.legend(loc='upper left')
    ax.set_ylabel('Variance')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
#     fig.savefig(string_save_location)


def plot_memory_utilisation(string_save_location, experiments, proportion, x_ticks, labels):
    fig, ax = plt.subplots(figsize=(5, 3))
    for experiment in range(len(experiments)):
        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.plot(x_ticks[experiment], proportion[experiment],
                label=labels[experiment],
                color=plot_colour)

    ax.set_title('Experiences discarded')
    ax.legend(loc='upper left')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Timestep (10e3)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
#     fig.savefig(string_save_location)


def plot_experiment_results(string_save_location, experiment_rewards, experiments, x_ticks, replay_memory_utilisation, lower_label, labels):

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
                label=labels[experiment],
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

#     fig.savefig(string_save_location)


def plot_smooth_experiment_results(string_save_location, experiment_rewards, experiments, x_ticks, replay_memory_utilisation, lower_label, labels):

    fig, ax = plt.subplots(figsize=(10, 6))


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
                label=labels[experiment],
                color=plot_colour)


    # ax.stackplot(yrs, rng + rnd, labels=[labels, labels, labels])
    ax.set_title('DQN Experiments')
    ax.legend(loc='upper left')
    ax.set_ylabel('Reward')
    ax.set_xlabel(lower_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()




def plot_action_histogram(labels, experiment_actions):
    fig, ax = plt.subplots(figsize=(5, 3))

    for experiment in range(len(experiment_actions)):

        flattened_list = [y for x in experiment_actions[experiment][-10000:] for y in x]

        plot_colour = next(ax._get_lines.prop_cycler)['color']
        ax.hist(flattened_list, normed=True, bins=6, color = plot_colour,
                 label=labels[experiment], alpha=.5)


    ax.set_title('Action Histogram')
    ax.legend(loc='upper left')
    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    fig.tight_layout()

def percentage_of_occurence(x, xs):
    x_occur = [action for action in xs if action == x ]
    return len(x_occur) / len(xs)

def plot_action_time_graph(labels, experiment_actions):


    for exp_id in range(len(experiment_actions)):

        exp = experiment_actions[exp_id]
        label = labels[exp_id]

        resolution = 1000

        fig, ax = plt.subplots(figsize=(5, 3))

        ax.set_title('Action-Time for ' + label)
        ax.legend(loc='upper left')
        ax.set_xlabel('Timestep (10e3)')
        ax.set_ylabel('Action Distribution')
        fig.tight_layout

        minimum_action = 1000;
        maximum_action = 0;

        for experiment in experiment_actions:
            for repeat in experiment:

                if len(repeat) == 0:
                    return;

                minimum = np.min(repeat);
                maximum = np.max(repeat);
                if minimum < minimum_action:
                    minimum_action = minimum
                if maximum > maximum_action:
                    maximum_action = maximum

        experiment = exp[0]

    #     experiment = experiment_actions[0][0]
        x_steps = int(len(experiment) / resolution)
        x_ticks = [x for x in range(x_steps)]

        experiment_fill_list = []

        for x in range(x_steps):
            action_sub_list = experiment[x : (x + resolution)]

            occurences = [percentage_of_occurence(i, action_sub_list) for i in range(minimum_action, maximum_action + 1)]
            occurences_fill = []
            occurence_track = 0;
            for occurence in occurences:
                occurences_fill.append((occurence_track, occurence_track + occurence))
                occurence_track += occurence

            experiment_fill_list.append(occurences_fill);

        fill_between = []

        for i in range(minimum_action, maximum_action + 1):
            top_y = []
            bottom_y = []
            for x in range(x_steps):
                top_y.append(experiment_fill_list[x][i][0])
                bottom_y.append(experiment_fill_list[x][i][1])

            ax.fill_between(x_ticks, top_y, bottom_y)


#     x_s = [x for x in range(x_steps)]
#     ys = [x*x for x in x_s]

#     ax.plot(x_s, ys);

def best_mean(ys, window_length):
    result = [];
    best_mean = -100;
    for i in range(len(ys)):
        before_index = max(0, i - window_length)
        mean = np.mean(np.array(ys[before_index:(i + 1)]))
        if(mean > best_mean):
            result.append(mean);
            best_mean = mean;
        else:
            result.append(best_mean);

    return result;


def plot_smooth_var_experiment_results(string_save_location, experiment_rewards, experiments, x_ticks, replay_memory_utilisation, lower_label, labels, repeat_rewards):

    fig, ax = plt.subplots(figsize=size)

    repeats = np.array(repeat_rewards)
    for experiment in range(len(experiments)):

        plot_colour = next(ax._get_lines.prop_cycler)['color']

        repeat_set = repeats[experiment]

        # savitzy_transformed = [savitzky_golay(np.array(best_mean(ys, 100)), window_size=1001, order=3) for ys in repeat_set]
        # savitzy_transformed = [mean_avg(ys) for ys in repeat_set]
        savitzy_transformed = [mean_avg(ys) for ys in repeat_set]
#         savitzy_transformed = [best_mean(ys, 100) for ys in repeat_set];

        means = []
        upper_vars = []
        lower_vars = []
        savitzy_length = len(savitzy_transformed)


        for t in range(len(savitzy_transformed[0])):


            values_at_point = [savitzy_transformed[array_index][t] for array_index in range(savitzy_length)]
            mean = np.mean(values_at_point)
            variance = np.sqrt(np.var(values_at_point))
            means.append(mean)
            upper_vars.append(mean + variance)
            lower_vars.append(mean - variance)

        ax.plot(x_ticks[experiment], means,
                label=labels[experiment],
                color=plot_colour)
        ax.fill_between(x_ticks[experiment], upper_vars, lower_vars,
                        alpha=0.1,color=plot_colour)

    # ax.stackplot(yrs, rng + rnd, labels=[labels, labels, labels])
    ax.set_title('Deep-Q Agent Learning Efficiency')
    ax.legend(loc='lower right')
    ax.set_ylabel('Reward')
    ax.set_xlabel(lower_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    fig.savefig("per_experiment.png")


def plot_smooth_exp_results_per_experiment(string_save_location, experiment_rewards, experiments, x_ticks, replay_memory_utilisation, lower_label, labels, repeat_rewards):

    fig, ax = plt.subplots(figsize=size)

    repeats = np.array(repeat_rewards)
    for experiment in range(len(experiments)):

        plot_colour = next(ax._get_lines.prop_cycler)['color']

        repeat_set = repeats[experiment]

        # savitzy_transformed = [savitzky_golay(np.array(best_mean(ys, 100)), window_size=1001, order=3) for ys in repeat_set]
        savitzy_transformed = [mean_avg(ys) for ys in repeat_set]
        # savitzy_transformed = repeat_set
#         savitzy_transformed = [best_mean(ys, 100) for ys in repeat_set];

        means = []
        upper_vars = []
        lower_vars = []
        savitzy_length = len(savitzy_transformed)

        for y_tick in range(len(savitzy_transformed)):
            if y_tick == 0:
                ax.plot(x_ticks[experiment], savitzy_transformed[y_tick],
                        color=plot_colour, label=labels[experiment])
            else:
                ax.plot(x_ticks[experiment], savitzy_transformed[y_tick],
                        color=plot_colour)

#         for t in range(len(savitzy_transformed[0])):


#             values_at_point = [savitzy_transformed[array_index][t] for array_index in range(savitzy_length)]
#             mean = np.mean(values_at_point)
#             variance = np.sqrt(np.var(values_at_point))
#             means.append(mean)
#             upper_vars.append(mean + variance)
#             lower_vars.append(mean - variance)

#         ax.plot(x_ticks[experiment], means,
#                 label=labels[experiment],
#                 color=plot_colour)
#         ax.fill_between(x_ticks[experiment], upper_vars, lower_vars,
#                         alpha=0.2,color=plot_colour)

    # ax.stackplot(yrs, rng + rnd, labels=[labels, labels, labels])
    ax.set_title('Deep-Q Agent Learning Efficiency')
    ax.legend(loc='lower right')
    ax.set_ylabel('Reward')
    ax.set_xlabel(lower_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    fig.savefig("per_repeat.png")
