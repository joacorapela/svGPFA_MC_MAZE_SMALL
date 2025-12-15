

import sys
import configparser
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import argparse
from pynwb import NWBHDF5IO
import plotly.graph_objects as go

import gcnu_common.numerical_methods
import gcnu_common.stats.discrimination
import svGPFA.utils.statsUtils
import svGPFA.plot.plotUtilsPlotly
import mcMazeUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--lda_start_time",
                        help="start time (in sec) for LDA claculation", type=float,
                        default=0.0)
    parser.add_argument("--lda_duration",
                        help="segment duration (in sec) for LDA calculation",
                        type=int, default=0.20)
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
    parser.add_argument("--model_metadata_filename_pattern",
                        help="filename pattern of the model metadata",
                        type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_save_filename_pattern",
                        help="filename pattern where the model was saved",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--histProj_fig_filename_pattern",
                        help=("filename pattern where to save the histogram "
                              "of projections figure"),
                        type=str,
                        default="../../figures/{:08d}_histProjectionsLatentsOntoDiscriminatoryDir{:02d}From{:.02f}Duration{:.03f}.{:s}")
    parser.add_argument("--latents_fig_filename_pattern",
                        help=("filename pattern where to save the latents "
                              "figure"),
                        type=str,
                        default="../../figures/{:08d}_ldaLatent{{:03d}}_mean.{{:s}}")
    args = parser.parse_args()


    estResNumber = args.estResNumber
    filepath = args.filepath
    lda_start_time = args.lda_start_time
    lda_duration = args.lda_duration
    align_event_name = args.align_event_name
    events_names = [str for str in args.events_names[1:-1].split(",")]
    events_colors = [str for str in args.events_colors[1:-1].split(",")]
    events_markers = [str for str in args.events_markers[1:-1].split(",")]
    model_metadata_filename = args.model_metadata_filename_pattern.format(estResNumber)
    model_save_filename = args.model_save_filename_pattern.format(estResNumber)
    histProj_fig_filename_pattern = args.histProj_fig_filename_pattern
    latents_fig_filename_pattern = args.latents_fig_filename_pattern.format(estResNumber)

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    trials_categories = mcMazeUtils.get_trials_categories(trials_df=trials_df)

    trials_colors_patterns = mcMazeUtils.get_trials_colors_patterns(
        trials_categories=trials_categories)
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    metadata = configparser.ConfigParser()
    metadata.read(model_metadata_filename)
    epoched_spikes_times_filename = metadata["data_params"]["epoched_spikes_times_filename"]

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]

    with open(model_save_filename, "rb") as f:
        est_results = pickle.load(f)
    trials_ids = est_results["trials"].tolist()
    kernels_types = est_results["kernels_types"]
    clusters = est_results["clusters"]
    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    estimated_params = est_results["estimated_params"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    events_times = []
    for event_name in events_names:
        events_times.append([trials_df.iloc[trial_id][event_name]
                             for trial_id in trials_ids])

    marked_events_times, marked_events_colors, marked_events_markers = \
        mcMazeUtils.buildMarkedEventsInfo(events_times=events_times,
                                    events_colors=events_colors,
                                    events_markers=events_markers)

    align_event_times = [trials_df.iloc[trial_id][align_event_name]
                         for trial_id in trials_ids]
    times = jnp.asarray(leg_quad_points)

    l_means, l_vars = svGPFA.utils.statsUtils.computeLatents(
        vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    l_means = np.transpose(jnp.asarray(l_means), (1, 2, 0))
    l_vars = np.transpose(jnp.asarray(l_vars), (1, 2, 0))
    estimatedC = jnp.asarray(C)

    # X: list of length n_categories
    # X[i] \in n_latents \times (n_selected_times * n_trials_in_category_i)
    X = mcMazeUtils.getLDAsamples(times=times, latents_means=l_means,
                                  start_time=lda_start_time,
                                  duration=lda_duration,
                                  trials_categories=trials_categories,
                                 )

    # discr_dirs \in n_latents \times n_components
    discr_dirs, discr_evals = gcnu_common.stats.discrimination.discriminativeLDA(X)
    discr_dirs = gcnu_common.numerical_methods.gram_schmidt(discr_dirs)

    # lda_l_means \in  n_trials \times n_times \times n_components
    lda_l_means = l_means @ discr_dirs

    n_categories = len(X)
    n_components = n_categories - 1

    # plot latents
    for latent_to_plot in range(n_components):
        fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentMeanAcrossTrials(
            times=times,
            latentsMeans=lda_l_means,
            latentToPlot=latent_to_plot,
            trials_ids=trials_ids,
            align_event_times=align_event_times,
            events_names=events_names,
            marked_events_times=marked_events_times,
            marked_events_colors=marked_events_colors,
            marked_events_markers=marked_events_markers,
            trials_colors_patterns=trials_colors_patterns,
            xlabel="Time (sec)")
        fig.write_image(latents_fig_filename_pattern.format(latent_to_plot, "png"))
        fig.write_html(latents_fig_filename_pattern.format(latent_to_plot, "html"))

    # calculate projection of latents into discr_dirs
    projs = [None] * n_categories
    for i in range(n_categories):
        # projs[i] \in (n_times_points[r] * n_latents) \times n_components
        projs[i] = X[i].T @ discr_dirs

    # plot histograms of projections of latents into discr_dirs
    categories_color_patternss = mcMazeUtils.get_categories_color_patterns()
    for j in range(n_components):
        fig = go.Figure()
        for i in range(n_categories):
            trace = go.Histogram(x=projs[i][:, j], name=f"Target {i+1}",
                                 marker_color=categories_color_patternss[i].format(1.0))
            fig.add_trace(trace)
        fig.update_xaxes(title="Projection")
        fig.update_layout(title=(f"Discriminatory direction {j} (eigenvalue: {discr_evals[j]})"))

        fig.write_html(histProj_fig_filename_pattern.format(
            estResNumber, j, lda_start_time, lda_duration, "html"))
        fig.write_image(histProj_fig_filename_pattern.format(
            estResNumber, j, lda_start_time, lda_duration, "png"))

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
