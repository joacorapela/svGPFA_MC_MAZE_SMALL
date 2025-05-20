
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

def subset_trials_ids_data(selected_trials_ids, trials_ids, spikes_times,
                           trials_start_times, trials_end_times):
    indices = np.nonzero(np.in1d(trials_ids, selected_trials_ids))[0]
    spikes_times_subset = [spikes_times[i] for i in indices]
    trials_start_times_subset = [trials_start_times[i] for i in indices]
    trials_end_times_subset = [trials_end_times[i] for i in indices]

    return spikes_times_subset, trials_start_times_subset, trials_end_times_subset

def subset_clusters_ids_data(selected_clusters_ids, clusters_ids,
                             spikes_times):
    indices = np.nonzero(np.in1d(clusters_ids, selected_clusters_ids))[0]
    n_trials = len(spikes_times)
    spikes_times_subset = [[spikes_times[r][i] for i in indices]
                           for r in range(n_trials)]

    return spikes_times_subset

