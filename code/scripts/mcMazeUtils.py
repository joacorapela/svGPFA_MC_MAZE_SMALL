
import numpy as np

def getSpikesTimes(mat):
    n_trials = mat["data"]["binned_response"][0,0].shape[0]
    n_units = mat["data"]["binned_response"][0,0][0,0].shape[1]
    spikes_times = [[None for n in range(n_units)] for r in range(n_trials)]
    Fs = 1e3

    for r in range(n_trials):
        for n in range(n_units):
            spikes_times[r][n] = \
                np.where(mat["data"]["binned_response"][0,0][r,0][:,n])[0]/Fs

    return n_trials, n_units, spikes_times

def subset_trials_data(selected_trials, trials, spikes_times,
                           trials_start_times, trials_end_times):
    indices = np.nonzero(np.in1d(trials, selected_trials))[0]
    spikes_times_subset = [spikes_times[i] for i in indices]
    trials_start_times_subset = [trials_start_times[i] for i in indices]
    trials_end_times_subset = [trials_end_times[i] for i in indices]

    return spikes_times_subset, trials_start_times_subset, trials_end_times_subset

def subset_clusters_data(selected_clusters, clusters,
                             spikes_times):
    indices = np.nonzero(np.in1d(clusters, selected_clusters))[0]
    n_trials = len(spikes_times)
    spikes_times_subset = [[spikes_times[r][i] for i in indices]
                           for r in range(n_trials)]

    return spikes_times_subset

def lists_are_equal(l1, l2):
    flag = all(x == y for x, y in zip(l1, l2))
    return flag

def buildMarkedEventsInfo(events_times, events_colors, events_markers):
    n_events = len(events_times)
    n_trials = len(events_times[0])
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for r in range(n_trials):
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(n_events):
            trial_marked_events_times.append(events_times[i][r])
            trial_marked_events_colors.append(events_colors[i])
            trial_marked_events_markers.append(events_markers[i])
        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers

def get_categories_color_patterns():
    color_patterns=["rgba(0,0,139,{:f})",       # darkblue
                    "rgba(0,0,255,{:f})",       # blue
                    "rgba(173,216,230,{:f})",   # lightblue
                    "rgba(139,0,0,{:f})",       # darkred
                    "rgba(255,0,0,{:f})",       # red
                    "rgba(2,48,32,{:f})",       # darkgreen
                    "rgba(144,238,144,{:f})",   # lightgreen
                    "rgba(255,255,0,{:f})",     # yellow
                    "rgba(255,255,224,{:f})"]   # lightyellow
    return color_patterns

def get_trials_colors_patterns(trials_categories, color_patterns=None):
    if color_patterns is None:
        color_patterns = get_categories_color_patterns()
    trials_color_patterns = [color_patterns[i] for i in trials_categories]
    return trials_color_patterns

def get_trials_categories(trials_df,
                          locations=[[-100, 35],
                                     [-92, 81],
                                     [-77, 82],
                                     [117, 15],
                                     [104, 80],
                                     [125, -64],
                                     [133, -81],
                                     [-91, -70],
                                     [-118, -83]],
                         ):
    n_trials = trials_df.shape[0]
    trials_categories = [None for r in range(n_trials)]
    n_locations = len(locations)

    for r in range(n_trials):
        target_pos = trials_df.iloc[r]["target_pos"]
        active_target = trials_df.iloc[r]["active_target"]
        for i in range(n_locations):
            if lists_are_equal(target_pos[active_target].tolist(), locations[i]):
                trials_categories[r] = i
                continue
        if trials_categories[r] is None:
            raise ValueError(f"Could not find trial category for trial {r}")

    return trials_categories

def getLDAsamples(times, latents_means, start_time, duration,
                  trials_categories):
    # times \in n_trials \times n_quad
    # latents_means \in n_trials \times n_quad \times n_latents
    # answer: list of length n_categories
    # answer[i] \in (n_selected_times * n_trials_in_category_i) \times n_latents
    n_trials = times.shape[0]
    n_categories = len(set(trials_categories))
    answer = [None] * n_categories
    for r in range(n_trials):
        time_indices = np.where(np.logical_and(
            start_time <= times[r, :], times[r, :] < start_time + duration))[0]
        if answer[trials_categories[r]] is None:
            answer[trials_categories[r]] = latents_means[r, time_indices, :].T
        else:
            answer[trials_categories[r]] = np.hstack(
                (answer[trials_categories[r]], latents_means[r, time_indices, :].T))
    return answer
