

import sys
import warnings
import numpy as np
import jax.numpy as jnp
import pickle
import argparse
from pynwb import NWBHDF5IO

# import gcnu_common.stats.pointProcesses.tests
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
    parser.add_argument("--cluster_id_to_plot", help="cluster id to plot",
                        type=int, default=0)
    parser.add_argument("--trial_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test", type=int, default=10)
    parser.add_argument("--n_time_steps_CIF",
                        help="number of time steps to plot for CIF",
                        type=int, default=1000)
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
    args = parser.parse_args()

    estResNumber = args.estResNumber
    filepath = args.filepath
    latent_to_plot = args.latent_to_plot
    latents_to_3D_plot = [int(str) for str in args.latents_to_3D_plot[1:-1].split(",")]
    latents_to_2D_plot = [int(str) for str in args.latents_to_2D_plot[1:-1].split(",")]
    cluster_id_to_plot = args.cluster_id_to_plot
    trial_to_plot = args.trial_to_plot
    ksTestGamma = args.ksTestGamma
    choices_colors_patterns = [str for str in args.choices_colors_patterns[1:-1].split(";")]
    align_event_name = args.align_event_name
    events_names = [str for str in args.events_names[1:-1].split(",")]
    events_colors = [str for str in args.events_colors[1:-1].split(",")]
    events_markers = [str for str in args.events_markers[1:-1].split(",")]
    n_time_steps_CIF = args.n_time_steps_CIF

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    trials_colors_patterns = utils.get_trials_colors_patterns(trials_df=trials_df)
    trials_colors = [trial_color_pattern.format(1.0)
                     for trial_color_pattern in trials_colors_patterns]

    modelSaveFilename = "../../results/{:08d}_estimatedModel.pickle".format(estResNumber)
    lowerBoundHistVsIterNoFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVsIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "../../figures/{:08d}_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatentsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    latents_to_3D_plot_str = "".join(str(i)+"_" for i in latents_to_3D_plot)
    latents_to_2D_plot_str = "".join(str(i)+"_" for i in latents_to_2D_plot)
    orthonormalizedLatents3DFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latents{:s}.{{:s}}".format(estResNumber, latents_to_3D_plot_str)
    orthonormalizedLatents2DFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latents{:s}.{{:s}}".format(estResNumber, latents_to_2D_plot_str)
    preIntensityFigFilenamePattern = "../../figures/{:08d}_preIntensity_clusterID_{:d}.{{:s}}".format(estResNumber, cluster_id_to_plot)
    orthonormalizedEmbeddingParamsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_preIntensity_params.{{:s}}".format(estResNumber)
#     CIFFigFilenamePattern = "../../figures/{:08d}_CIF_trial{:03d}_cluster_id_{:03d}.{{:s}}".format(estResNumber, trial_to_plot, cluster_id_to_plot)
#     CIFimageFigFilenamePattern = "../../figures/{:08d}_CIFimage_cluster_id_{:03d}_sortedBy{:s}.{{:s}}".format(estResNumber, cluster_id_to_plot, column_name)
    CIFsOneNeuronAllTrialsFigFilenamePattern = "../../figures/{:08d}_intensityFunctionOneNeuronAllTrials_cluster_id_{:03d}.{{:s}}".format(estResNumber, cluster_id_to_plot)
    ksTestTimeRescalingNumericalCorrectionFigFilenamePattern = "../../figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_cluster_id_{:d}.{{:s}}".format(estResNumber, trial_to_plot, cluster_id_to_plot)
    rocFigFilenamePattern = "../../figures/{:08d}_predictive_analysis_trial{:03d}_cluster_id_{:d}.{{:s}}".format(estResNumber, trial_to_plot, cluster_id_to_plot)
    kernelsParamsFigFilenamePattern = "../../figures/{:08d}_kernels_params.{{:s}}".format(estResNumber)

    with open(modelSaveFilename, "rb") as f:
        est_results = pickle.load(f)
    if "estimation_params" not in est_results.keys():
        est_results["estimation_params"] = est_results.pop("estimaton_params")
    trials_ids = est_results["trials_ids"].tolist()
    kernels_types = est_results["kernels_types"]
    clusters_ids = est_results["clusters_ids"][0]
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

#     n_trials = len(spikes_times)
    cluster_id_to_plot_index = np.where(clusters_ids==cluster_id_to_plot)[0][0]
#     clusters_ids_str = " ".join(str(i) for i in clusters_ids)
#     if len(cluster_id_to_plot_index) == 0:
#         raise ValueError("Cluster id {:d} is not valid. Valid cluster id are ".format(
#             cluster_id_to_plot.item()) + clusters_ids_str)
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

    # plot lower bound history
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    # plot estimated latent across trials
    l_means, l_vars = svGPFA.utils.statsUtils.computeLatents(
        vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    l_means = np.transpose(jnp.asarray(l_means), (1, 2, 0))
    l_vars = np.transpose(jnp.asarray(l_vars), (1, 2, 0))
#     testMuK, testVarK = model.predictLatents(times=trials_times)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
        times=times,
        latentsMeans=l_means,
        latentsSTDs=np.sqrt(l_vars),
        latentToPlot=latent_to_plot,
        trials_ids=trials_ids,
        trials_colors_patterns=trials_colors_patterns,
        xlabel="Time (sec)")
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))

    # plot orthonormalized estimated latent across trials
#     testMuK, _ = model.predictLatents(times=trials_times)
#     test_mu_k_np = [testMuK[r].detach().numpy() for r in range(len(testMuK))]
    estimatedC, estimatedD = jnp.asarray(C), jnp.asarray(d)
#     estimatedC_np = estimatedC.detach().numpy()

    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
        trials_times=times, latentsMeans=l_means,
        C=estimatedC, trials_ids=trials_ids,
        latentToPlot=latent_to_plot,
        align_event_times=align_event_times,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors=trials_colors,
#         trials_annotations=trials_annotations,
        xlabel="Time (sec)")
    fig.write_image(orthonormalizedLatentsFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatentsFigFilenamePattern.format("html"))

    """
    fig = svGPFA.plot.plotUtilsPlotly.get2DPlotOrthonormalizedLatentsAcrossTrials(
        trials_times=times, latentsMeans=l_means,
        C=estimatedC, trials_ids=trials_ids,
        latentsToPlot=latents_to_2D_plot,
        align_event_times=align_event_times,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors=trials_colors,
        # trials_annotations=trials_annotations,
    )
    fig.write_image(orthonormalizedLatents2DFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatents2DFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
        trials_times=times, latentsMeans=l_means,
        C=estimatedC, trials_ids=trials_ids,
        latentsToPlot=latents_to_3D_plot,
        align_event_times=align_event_times,
        marked_events_times=marked_events_times,
        marked_events_colors=marked_events_colors,
        marked_events_markers=marked_events_markers,
        trials_colors=trials_colors,
        # trials_annotations=trials_annotations,
    )
    fig.write_image(orthonormalizedLatents3DFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatents3DFigFilenamePattern.format("html"))

    # plot preIntensity
    h_means, h_vars = svGPFA.utils.statsUtils.computePreIntensity(
        C=C, d=d, vMean=vMean, vChol=vChol, kernels_params=kernels_params,
        ind_points_locs=ind_points_locs, kernels_types=kernels_types,
        leg_quad_points=leg_quad_points, reg_param=reg_param)

    title = "Neuron {:d}".format(cluster_id_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotPreIntensityAcrossTrials(
        times=times,
        preIntensityMeans=h_means[:, cluster_id_to_plot_index, :],
        preIntensitySTDs=np.sqrt(h_vars[:, cluster_id_to_plot_index, :]),
        trials_colors_patterns=trials_colors_patterns,
        title=title)
    fig.write_image(preIntensityFigFilenamePattern.format("png"))
    fig.write_html(preIntensityFigFilenamePattern.format("html"))
    """

#     # calculate expected IF values (for KS test and IF plots)
#     with torch.no_grad():
#         cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
#     cif_values_GOF = cif_values[trial_to_plot][cluster_id_to_plot_index]
# 
#     # CIF
# 
#     # CIFs one neuron all trials
#     title = f"Cluster ID: {clusters_ids[cluster_id_to_plot_index]}, Region: {locs_for_clusters_ids[cluster_id_to_plot]}"
#     fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
#         trials_times=trials_times,
#         cif_values=cif_values,
#         neuron_index=cluster_id_to_plot_index,
#         spikes_times=spikes_times,
#         trials_ids=trials_ids,
#         align_event_times=align_event_times,
#         marked_events_times=marked_events_times,
#         marked_events_colors=marked_events_colors,
#         marked_events_markers=marked_events_markers,
#         trials_annotations=trials_annotations,
#         trials_colors=trials_colors,
#         title=title,
#     )
#     fig.write_image(CIFsOneNeuronAllTrialsFigFilenamePattern.format("png"))
#     fig.write_html(CIFsOneNeuronAllTrialsFigFilenamePattern.format("html"))
# 
#     trial_times_GOF = trials_times[trial_to_plot, :, 0]
#     spikes_times_GOF = spikes_times[trial_to_plot][cluster_id_to_plot_index]
#     title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
#         trial_to_plot, cluster_id_to_plot, len(spikes_times_GOF))
# 
#     # plot KS test time rescaling (numerical correction)
#     if len(spikes_times_GOF) > 0:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF, cif_times=trial_times_GOF, cif_values=cif_values_GOF, gamma=ksTestGamma)
#         fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
#     fig.write_image(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("png"))
#     fig.write_html(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("html"))
# 
#     # ROC predictive analysis
#     fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
#         spikes_times=spikes_times_GOF,
#         cif_times=trial_times_GOF,
#         cif_values=cif_values_GOF)
#     fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
#         fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
#     fig.write_image(rocFigFilenamePattern.format("png"))
#     fig.write_html(rocFigFilenamePattern.format("html"))
# 
#     # plot orthonormalized preIntensity parameters
#     hovertemplate = "value: %{y}<br>" + \
#                     "neuron index: %{x}<br>" + \
#                     "%{text}"
#     text = [f"cluster_id: {cluster_id}" for cluster_id in clusters_ids]
#     estimatedC, estimatedD = model.getSVEmbeddingParams()
#     fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedEmbeddingParams(
#         C=estimatedC.numpy(), d=estimatedD.numpy(),
#         hovertemplate=hovertemplate, text=text)
#     fig.write_image(
#         orthonormalizedEmbeddingParamsFigFilenamePattern.format("png"))
#     fig.write_html(
#         orthonormalizedEmbeddingParamsFigFilenamePattern.format("html"))

    # plot kernel parameters
    # kernelsParams = model.getKernelsParams()
    # kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
    """
    fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
        kernelsTypes=kernels_types, kernelsParams=kernels_params)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))
    """

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
