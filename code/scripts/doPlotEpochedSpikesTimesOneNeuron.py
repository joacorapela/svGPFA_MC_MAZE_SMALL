
import sys
import argparse
import pickle
import numpy as np

import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", help="cluster to plot",
                        type=int, default=0)
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--behavioral_events_names", type=str,
                        help="behavioral events names to plot",
                        default="[trials_start_times,stimOn_times,goCue_times,response_times,feedback_times,stimOff_times,trials_end_times]")
    parser.add_argument("--colors_event_name", type=str,
                        help="events names used to color spikes", default="choice")
    parser.add_argument("--align_event_name",
                        help="name of event used for alignment",
                        type=str,
                        default="response_times")
    parser.add_argument("--events_names",
                        help="names of marked events (e.g., stimOn_times, goCue_times, response_times, feedback_times, stimOff_times)",
                        type=str,
                        default="[stimOn_times,response_times,stimOff_times]")
    parser.add_argument("--events_colors",
                        help="colors for marked events (e.g., stimOn_times, goCue_times, response_times, feedback_times, stimOff_times)",
                        type=str, default="[magenta,green,black]")
    parser.add_argument("--events_markers",
                        help="markers for marked events (e.g., stimOn_times, goCue_times, response_times, feedback_times, stimOff_times)",
                        type=str, default="[circle,circle,circle]")
    parser.add_argument("--xmin", type=float, help="mininum x-axis value",
                        default=-3)
    parser.add_argument("--xmax", type=float, help="maximum x-axis value",
                        default=3)
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedSpikesTimes.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help=("figure filename pattern"),
                        type=str,
                        default="../../figures/00000000_dandisetID{:s}_epochedEvent{:s}_epochedSpikesTimes_cluster{:02d}.{:s}")
    args = parser.parse_args()

    cluster = args.cluster
    dandiset_ID = args.dandiset_ID
    epoch_event_name = args.epoch_event_name
    events_names = [str for str in args.events_names[1:-1].split(",")]
    events_colors = [str for str in args.events_colors[1:-1].split(",")]
    events_markers = [str for str in args.events_markers[1:-1].split(",")]
    xmin = args.xmin
    xmax = args.xmax
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

    n_trials = len(spikes_times)
    trials_ids = np.arange(n_trials)
    n_clusters = len(spikes_times[0])


    # colors_event = trials_info[colors_event_name]

    # cluster_id_index = np.nonzero(np.array(clusters_ids)==cluster_id)[0].item()
    # loc_acronym_for_cluster_id = locs_for_clusters_ids[cluster_id_index]

    # colors_event = trials_info[colors_event_name]
    # trials_colors = ["red" if an_event==1 else "blue" for an_event in colors_event]

    # events_names.append("trials_start_times")
    # events_names.append("trials_end_times")
    # events_times = []
    # for event_name in events_names:
    #     events_times.append([trials_info[event_name][trial_id]
    #                          for trial_id in trials_ids])
    # events_colors.extend(["gray", "brown"])
    # events_markers.extend(["circle", "circle"])

    # marked_events_times, marked_events_colors, marked_events_markers = \
    #     iblUtils.buildMarkedEventsInfo(
    #         events_times=events_times,
    #         events_colors=events_colors,
    #         events_markers=events_markers,
    #     )

    # align_event_times = [trials_info[align_event_name][trial_id]
    #                      for trial_id in trials_ids]

    # sorting_label = sorting_event_name if sorting_event_name is not None else "None"
    title=f"Cluster: {cluster}, Epoched by: {epoch_event_name}"
    fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneCluster(
        spikes_times=spikes_times,
        # sorting_times=sorting_times,
        cluster=cluster,
        title=title,
        trials_ids=trials_ids,
        # marked_events_times=marked_events_times,
        # marked_events_colors=marked_events_colors,
        # marked_events_markers=marked_events_markers,
        # align_event_times=align_event_times,
        # trials_colors=trials_colors,
    )
    if xmin is None:
        xmin = np.min(trials_start_times)
    if xmax is None:
        xmax = np.max(trials_end_times)
    fig.update_xaxes(range=[xmin, xmax])

    fig.write_image(fig_filename_pattern.format(
        dandiset_ID, epoch_event_name, cluster, "png"))
    fig.write_html(fig_filename_pattern.format(
        dandiset_ID, epoch_event_name, cluster, "html"))

    # breakpoint()

if __name__=="__main__":
    main(sys.argv)
