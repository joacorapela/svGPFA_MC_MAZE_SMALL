
import sys
import argparse
import configparser
import pickle
import numpy as np

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import svGPFA.utils.miscUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
                        # default="sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedSpikesTimes.{:s}")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath = args.filepath
    epoch_event_name = args.epoch_event_name
    epoched_spikes_times_filename_pattern = args.epoched_spikes_times_filename_pattern

    #%%
    # Download data
    # ^^^^^^^^^^^^^
    # with DandiAPIClient() as client:
    #     asset = client.get_dandiset(dandiset_ID, "draft").get_asset_by_path(filepath)
    #     s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

    # io = NWBHDF5IO(s3_path, mode="r", driver="ros3")

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        trials_df = nwbfile.intervals["trials"].to_dataframe()


    # n_neurons
    n_neurons = units_df.shape[0]

    # continuous spikes times
    continuous_spikes_times = [None for n in range(n_neurons)]
    units_ids = [None for n in range(n_neurons)]
    units_locs = [None for n in range(n_neurons)]
    for n in range(n_neurons):
        units_ids[n] = units_df.index[n]
        units_locs[n] = units_df.iloc[n].electrodes['location']
        continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

    # n_trials
    n_trials = trials_df.shape[0]

    #%%
    # Epoch spikes times
    # ^^^^^^^^^^^^^^^^^^
    epochs_times = [None for r in range(n_trials)]
    trials_start_times = [None for r in range(n_trials)]
    trials_end_times = [None for r in range(n_trials)]
    spikes_times = [[None for n in range(n_neurons)] for r in range(n_trials)]
    for r in range(n_trials):
        epoch_start_time = trials_df.iloc[r]["start_time"]
        epoch_end_time = trials_df.iloc[r]["stop_time"]
        epoch_time = trials_df.iloc[r][epoch_event_name]
        epochs_times[r] = epoch_time
        trials_start_times[r] = epoch_start_time - epoch_time
        trials_end_times[r] = epoch_end_time - epoch_time
        for n in range(n_neurons):
            spikes_times[r][n] = (continuous_spikes_times[n][
                np.logical_and(epoch_start_time <= continuous_spikes_times[n],
                               continuous_spikes_times[n] <= epoch_end_time)] -
                epoch_time)

    results_to_save = {"neurons": units_ids,
                       "region": units_locs,
                       "epochs_times": epochs_times,
                       "spikes_times": spikes_times,
                       "trials_start_times": trials_start_times,
                       "trials_end_times": trials_end_times,
                      }
    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format(
            dandiset_ID, epoch_event_name, "pickle")
    with open(epoched_spikes_times_filename, "wb") as f:
        pickle.dump(results_to_save, f)
    print("Saved results to {:s}".format(
        epoched_spikes_times_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
