
import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import svGPFA.utils.miscUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000128")
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--epoched_hand_pos_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedHandPos.{:s}")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath = args.filepath
    epoch_event_name = args.epoch_event_name
    epoched_hand_pos_filename_pattern = args.epoched_hand_pos_filename_pattern

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        hand_pos_df = pd.DataFrame(nwbfile.processing["behavior"]["hand_pos"].data[:], columns=["x", "y"])
        hand_pos_df.index = nwbfile.processing["behavior"]["hand_pos"].timestamps[:]
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    # n_trials
    n_trials = trials_df.shape[0]

    #%%
    # Epoch hand positions
    # ^^^^^^^^^^^^^^^^^^^^
    epochs_times = [None for r in range(n_trials)]
    hand_pos_list = [None for r in range(n_trials)]
    for r in range(n_trials):
        epoch_start_time = trials_df.iloc[r]["start_time"]
        epoch_end_time = trials_df.iloc[r]["stop_time"]
        epoch_time = trials_df.iloc[r][epoch_event_name]
        epochs_times[r] = epoch_time

        trial_indices = np.where(np.logical_and(
            epoch_start_time <= hand_pos_df.index,
            hand_pos_df.index < epoch_end_time,
        ))[0]

        times = hand_pos_df.index[trial_indices] - epoch_time
        x = hand_pos_df.iloc[trial_indices]["x"]
        y = hand_pos_df.iloc[trial_indices]["y"]
        hand_pos_list[r] = pd.DataFrame(index=times, data=zip(x, y),
                                        columns=["x", "y"])
    results_to_save = {
                       "hand_pos_list": hand_pos_list,
                       "epochs_times": epochs_times,
                      }
    epoched_hand_pos_filename = \
        epoched_hand_pos_filename_pattern.format(
            dandiset_ID, epoch_event_name, "pickle")
    with open(epoched_hand_pos_filename, "wb") as f:
        pickle.dump(results_to_save, f)
    print("Saved results to {:s}".format(
        epoched_hand_pos_filename))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
