

import sys
import warnings
import configparser
import numpy as np
import jax.numpy as jnp
import pickle
import argparse
from pynwb import NWBHDF5IO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.graph_objects as go

import gcnu_common.numerical_methods
import svGPFA.utils.miscUtils
import svGPFA.utils.statsUtils
import svGPFA.plot.plotUtilsPlotly
import utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--latent_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--latents_to_3D_plot", help="latents to plot in 3D plot",
                        type=str, default="[0,1,2]")
    parser.add_argument("--latents_to_2D_plot", help="latents to plot in 2D plot",
                        type=str, default="[0,1]")
#     parser.add_argument("--cluster", help="cluster to plot",
#                         type=int, default=10)
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
    args = parser.parse_args()

    estResNumber = args.estResNumber
    filepath = args.filepath
    latent_to_plot = args.latent_to_plot
    latents_to_3D_plot = [int(str) for str in args.latents_to_3D_plot[1:-1].split(",")]
    latents_to_2D_plot = [int(str) for str in args.latents_to_2D_plot[1:-1].split(",")]
#     cluster = args.cluster
    align_event_name = args.align_event_name
    events_names = [str for str in args.events_names[1:-1].split(",")]
    events_colors = [str for str in args.events_colors[1:-1].split(",")]
    events_markers = [str for str in args.events_markers[1:-1].split(",")]

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    trials_categories = utils.get_trials_categories(trials_df=trials_df)

    trials_colors_patterns = utils.get_trials_colors_patterns(
        trials_categories=trials_categories)
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    modelSaveFilename = "../../results/{:08d}_estimatedModel.pickle".format(estResNumber)
    metaDataFilename = "../../results/{:08d}_estimation_metaData.ini".format(estResNumber)
    latentsFigFilenamePattern = "../../figures/{:08d}_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatentsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    latents_to_3D_plot_str = "".join(str(i)+"_" for i in latents_to_3D_plot)
    latents_to_2D_plot_str = "".join(str(i)+"_" for i in latents_to_2D_plot)
    orthonormalizedLatents3DFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latents{:s}.{{:s}}".format(estResNumber, latents_to_3D_plot_str)
    orthonormalizedLatents2DFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latents{:s}.{{:s}}".format(estResNumber, latents_to_2D_plot_str)

    metaData = configparser.ConfigParser()
    metaData.read(metaDataFilename)
    epoched_spikes_times_filename = metaData["data_params"]["epoched_spikes_times_filename"]

    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]

    with open(modelSaveFilename, "rb") as f:
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
    estimated_params = est_results["estimated_params"]

    vMean = estimated_params["variational_mean"]
    vChol = estimated_params["variational_chol_vecs"]
    C = estimated_params["C"]
    d = estimated_params["d"]
    kernels_params = estimated_params["kernels_params"]
    ind_points_locs = estimated_params["ind_points_locs"]

    # index to the original spikes
#     cluster_index = np.where(clusters==cluster)[0][0]

#     n_trials = len(spikes_times)
#     clusters_ids_str = " ".join(str(i) for i in clusters_ids)
#     if len(cluster_index) == 0:
#         raise ValueError("Cluster id {:d} is not valid. Valid cluster id are ".format(
#             cluster_index.item()) + clusters_ids_str)
# 
#     trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
#         start_times=trials_start_times,
#         end_times=trials_end_times,
#         n_steps=n_time_steps_CIF)
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

    l_means, l_vars = svGPFA.utils.statsUtils.computeLatents(
        vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    l_means = np.transpose(jnp.asarray(l_means), (1, 2, 0))
    l_vars = np.transpose(jnp.asarray(l_vars), (1, 2, 0))
    estimatedC = jnp.asarray(C)

    lda_start_time = 0.0
    lda_duration = 0.20
    latent_to_plot = 1
    histProj_fig_filename_pattern = \
            "../../figures/{:08d}_histProjectionsLatentsOntoDiscriminatoryDir{:02d}From{:.02f}Duration{:.03f}_scikitLearn.{:s}"
    latents_fig_filename_pattern = "../../figures/{:08d}_ldaLatent{:03d}_scikitLearn.{{:s}}".format(estResNumber, latent_to_plot)

    # X: list of length n_categories
    # X[i] \in n_latents \times (n_selected_times * n_trials_in_category_i)
    X = utils.getLDAsamples(times=times, latents_means=l_means,
                            start_time=lda_start_time,
                            duration=lda_duration,
                            trials_categories=trials_categories,
                           )
    # Xs: list of length n_categories
    Xs = [X[i].T for i in range(len(X))]

    y = [] # Labels
    for i in range(len(Xs)):
        y.extend([i] * Xs[i].shape[0])

    # Combine and ensure final data arrays are NumPy arrays
    Xs = np.vstack(Xs)
    ys = np.array(y)

    print(f"Dataset Shape: X={Xs.shape}, y={ys.shape}")
    print("-" * 30)

    n_categories = len(set(trials_categories))

    # n_components should be smaller than n_categories
    n_components = n_categories - 1

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(Xs, ys)

    discr_dirs = lda.scalings_
    discr_evals = lda.explained_variance_ratio_

    discr_dirs = gcnu_common.numerical_methods.gram_schmidt(discr_dirs)
    # lda_l_means \in  n_trials \times n_times \times n_components
    # lda_l_vars  \in  n_trials \times n_times \times n_components
    lda_l_means = l_means @ discr_dirs
    lda_l_vars =  l_vars @ discr_dirs**2

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
        times=times,
        latentsMeans=lda_l_means,
        latentsSTDs=np.sqrt(lda_l_vars),
        latentToPlot=latent_to_plot,
        trials_ids=trials_ids,
        align_event_times=align_event_times,
        events_names=events_names,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors_patterns=trials_colors_patterns,
        xlabel="Time (sec)")
    fig.write_image(latents_fig_filename_pattern.format("png"))
    fig.write_html(latents_fig_filename_pattern.format("html"))

    n_categories = len(X)
    projs = [None] * n_categories
    categories_color_patternss = utils.get_categories_color_patterns()
    for i in range(n_categories):
        # projs[i] \in (n_times_points[r] * n_latents) \times n_components
        projs[i] = X[i].T @ discr_dirs

    for j in range(n_components):
        fig = go.Figure()
        for i in range(n_categories):
            trace = go.Histogram(x=projs[i][:, j], name=f"Category {i+1}",
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
