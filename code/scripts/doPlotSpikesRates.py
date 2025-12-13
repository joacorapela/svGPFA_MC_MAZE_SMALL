
import sys
import os
import numpy as np
import scipy.io
import argparse
import pickle

import gcnu_common.utils
import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedSpikesTimes.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help=("figure filename pattern"),
                        type=str,
                        default=("../../figures/spikes_rates_{:s}.{:s}"))
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    epoch_event_name = args.epoch_event_name
    epoched_spikes_times_filename_pattern = args.epoched_spikes_times_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format(
            dandiset_ID, epoch_event_name, "pickle")
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]

    trials_durations = [trials_end_times[i] - trials_start_times[i]
                        for i in range(len(trials_end_times))]
    spikes_rates = svGPFA.utils.miscUtils.computeSpikeRates(
        trials_durations=trials_durations, spikes_times=spikes_times)

    n_trials = len(spikes_times)
    n_clusters = len(spikes_times[0])
    trials_ids = np.arange(n_trials)
    clusters_ids = np.arange(n_clusters)

    fig = svGPFA.plot.plotUtilsPlotly.getPlotSpikesRatesAllTrialsAllClusters(
        spikes_rates=spikes_rates, trials_ids=trials_ids,
        clusters_ids=clusters_ids)

    descriptor, _ = os.path.splitext(os.path.basename(
        epoched_spikes_times_filename))

    fig.write_image(fig_filename_pattern.format(descriptor, "png"))
    fig.write_html(fig_filename_pattern.format(descriptor, "html"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
