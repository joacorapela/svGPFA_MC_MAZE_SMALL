
import sys
import argparse
import pickle
import numpy as np
from pynwb import NWBHDF5IO
import plotly.graph_objs as go
import svGPFA.plot.plotUtilsPlotly
import utils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath_pattern", help="dandi filepath pattern", type=str,
                        default="../../data/{:s}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--cluster", help="cluster to plot", type=int,
                        default=10)
    parser.add_argument("--trials_to_plot", help="trials to plot", type=str,
                        default="[]")
    parser.add_argument("--gf_std_secs", help="gaussian filter std (sec)",
                        type=float, default=0.1)
    parser.add_argument("--bin_size_secs", help="bin size (secs)",
                        type=float, default=0.01)
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedSpikesTimes.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/smoothedSpikes_00000000_dandisetID{:s}_epochedEvent{:s}_binned{:.02f}sec_gfSTD{:f}_cluster{:03d}.{:s}")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath_pattern = args.filepath_pattern
    epoch_event_name = args.epoch_event_name
    cluster = args.cluster
    if len(args.trials_to_plot)>2:
        trials_to_plot = [int(str) for str in args.trials_to_plot[1:-1].split(",")]
    else:
        trials_to_plot = []
    gf_std_secs = args.gf_std_secs
    bin_size_secs = args.bin_size_secs
    epoched_spikes_times_filename_pattern = \
        args.epoched_spikes_times_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format(
            dandiset_ID, epoch_event_name, "pickle")

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]
    n_trials = len(spikes_times)

    filepath = filepath_pattern.format(dandiset_ID)
    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    trials_colors_patterns = utils.get_trials_colors_patterns(trials_df=trials_df)
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    if len(trials_to_plot) == 0:
        trials_to_plot = np.arange(n_trials)
    title = "Cluster {:d}".format(cluster)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotSmoothedSpikesUnequalLengthTrials(
        spikes_times=spikes_times,
        gf_std_secs=gf_std_secs,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        bin_size_secs=bin_size_secs,
        cluster=cluster,
        trials_to_plot=trials_to_plot,
        title=title,
        trials_colors=trials_colors,
    )
    png_fig_filename = fig_filename_pattern.format(dandiset_ID,
                                                   epoch_event_name,
                                                   bin_size_secs,
                                                   gf_std_secs,
                                                   cluster,
                                                   "png")
    html_fig_filename = fig_filename_pattern.format(dandiset_ID,
                                                    epoch_event_name,
                                                    bin_size_secs,
                                                    gf_std_secs,
                                                    cluster,
                                                    "html")
    fig.write_image(png_fig_filename)
    fig.write_html(html_fig_filename)
    # fig.show()

    # breakpoint()

if __name__=="__main__":
    main(sys.argv)
