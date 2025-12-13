

import sys
import warnings
import configparser
import numpy as np
import jax.numpy as jnp
import pickle
import argparse
from pynwb import NWBHDF5IO

import gcnu_common.stats.pointProcesses.tests
import svGPFA.utils.miscUtils
import svGPFA.utils.statsUtils
import svGPFA.plot.plotUtilsPlotly
import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--filepath_pattern", help="dandi filepath pattern", type=str,
                        default="../../data/{:s}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--cluster", help="cluster to plot",
                        type=int, default=10)
    parser.add_argument("--bin_size_secs", help="bin size (secs)",
                        type=float, default=0.01)
    parser.add_argument("--choices_colors_patterns",
                        help="color patterns for choice left and choice right",
                        type=str,
                        default="[rgba(0,0,255,{:f});rgba(255,0,0,{:f})]")
    parser.add_argument("--align_event_name",
                        help="name of event used for alignment",
                        type=str,
                        default="move_onset_time")
    parser.add_argument("--events_names",
                        help="names of marked events (e.g., start_time, target_on_time, go_cue_time, move_onset_time, stop_time)",
                        type=str,
                        default="[start_time,target_on_time,go_cue_time,move_onset_time,stop_time]")
    parser.add_argument("--events_colors",
                        help="colors for marked events (e.g., start_time, target_on_time, go_cue_time, move_onset_time, stop_time)",
                        type=str, default="[black,cyan,magenta,orange,pink]")
    parser.add_argument("--events_markers",
                        help="markers for marked events (e.g., start_time, target_on_time, go_cue_time, move_onset_time, stop_time)",
                        type=str, default="[circle,circle,circle,circle,circle]")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern", type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern", type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--fig_filename_pattern_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/{:08d}_preIntensity_cluster{:03d}_allTrials.{{:s}}")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    filepath_pattern = args.filepath_pattern
    cluster = args.cluster
    bin_size_secs = args.bin_size_secs
    choices_colors_patterns = [str for str in args.choices_colors_patterns[1:-1].split(";")]
    align_event_name = args.align_event_name
    events_names = [str for str in args.events_names[1:-1].split(",")]
    events_colors = [str for str in args.events_colors[1:-1].split(",")]
    events_markers = [str for str in args.events_markers[1:-1].split(",")]
    model_save_filename_pattern = args.model_save_filename_pattern
    metadata_filename_pattern = args.metadata_filename_pattern
    fig_filename_pattern_pattern = args.fig_filename_pattern_pattern

    model_save_filename = model_save_filename_pattern.format(estResNumber)
    metadata_filename = metadata_filename_pattern.format(estResNumber)
    fig_filename_pattern = fig_filename_pattern_pattern.format(estResNumber, cluster)

    metaData = configparser.ConfigParser()
    metaData.read(metadata_filename)
    dandiset_ID = metaData["data_params"]["dandiset_ID"]
    epoched_spikes_times_filename = metaData["data_params"]["epoched_spikes_times_filename"]

    filepath = filepath_pattern.format(dandiset_ID)
    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    trials_colors_patterns = utils.get_trials_colors_patterns(trials_df=trials_df)
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]

    with open(model_save_filename, "rb") as f:
        est_results = pickle.load(f)
    if "estimation_params" not in est_results.keys():
        est_results["estimation_params"] = est_results.pop("estimaton_params")
    # trials_ids = est_results["trials_ids"].tolist()
    trials_ids = est_results["trials"].tolist()
    kernels_types = est_results["kernels_types"]
    # clusters = est_results["clusters_ids"]
    clusters = est_results["clusters"]
    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    lowerBoundHist = est_results["lower_bound_hist"]
    elapsedTimeHist = est_results["elapsed_time_hist"]
    estimated_params = est_results["estimated_params"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    assert(len(clusters) == C.shape[0])

    # We should use cluster_index as index to estimated arrays but cluster as
    # index to the original spikes
#     where_res = np.where(clusters==cluster)
#     if len(where_res[0]) == 0:
#         raise ValueError(
#             f"Cluster {cluster} not found in valid clusters: {clusters}")
#     cluster_index = where_res[0][0]
    cluster_index = clusters.index(cluster)

#     n_trials = len(spikes_times)
#     clusters_ids_str = " ".join(str(i) for i in clusters_ids)
#     if len(cluster_index) == 0:
#         raise ValueError("Cluster id {:d} is not valid. Valid cluster id are ".format(
#             cluster_index.item()) + clusters_ids_str)
# 
#     trials_labels = np.array([str(i) for i in trials_ids])

#     n_trials = len(spikes_times)

#     trials_choices = [trials_info["choice"][trial_id]
#                       for trial_id in trials_ids]
#     trials_rewarded = [trials_info["feedbackType"][trial_id]
#                        for trial_id in trials_ids]
#     trials_contrast = [trials_info["contrastRight"][trial_id]
#                        if not np.isnan(trials_info["contrastRight"][trial_id])
#                        else trials_info["contrastLeft"][trial_id]
#                        for trial_id in trials_ids]
#     trials_colors_patterns = [choices_colors_patterns[0]
#                               if trials_choices[r] == -1
#                               else choices_colors_patterns[1]
#                               for r in range(n_trials)]
#     trials_colors = [trial_color_pattern.format(1.0)
#                      for trial_color_pattern in trials_colors_patterns]
#     trials_annotations = {"choice": trials_choices,
#                           "rewarded": trials_rewarded,
#                           "contrast": trials_contrast,
#                           "choice_prev": np.insert(trials_choices[:-1], 0,
#                                                    np.NAN),
#                           "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
#                                                      np.NAN)}
# 
    events_times = []
    for event_name in events_names:
        events_times.append([trials_df.iloc[trial_id][event_name]
                             for trial_id in trials_ids])

    marked_events_times, marked_events_colors, marked_events_markers = \
        utils.buildMarkedEventsInfo(events_times=events_times,
                                    events_colors=events_colors,
                                    events_markers=events_markers)

    align_event_times = [trials_df.iloc[trial_id][align_event_name]
                         for trial_id in trials_ids]
    times = jnp.asarray(leg_quad_points)

    # plot preIntensity
    h_means, h_vars = svGPFA.utils.statsUtils.computePreIntensity(
        C=C, d=d, vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    title = "Cluster {:d}".format(cluster)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotPreIntensityAcrossTrials(
        times=times,
        preIntensityMeans=h_means[:, cluster_index, :],
        preIntensitySTDs=np.sqrt(h_vars[:, cluster_index, :]),
        trials_ids=trials_ids,
        align_event_times=align_event_times,
        events_names=events_names,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors_patterns=trials_colors_patterns,
        title=title)
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
