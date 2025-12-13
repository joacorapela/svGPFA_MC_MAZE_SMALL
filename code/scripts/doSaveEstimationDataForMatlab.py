import sys
import os
import time
import random
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
import configparser
import warnings

import gcnu_common.utils.neural_data_analysis
import gcnu_common.utils.config_dict
import svGPFA.stats.em
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import mcMazeUtils

jax.config.update("jax_enable_x64", True)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number", type=int)
    parser.add_argument("--n_time_steps_CIF",
                        help="number of time steps to plot for CIF",
                        type=int, default=1000)
    parser.add_argument("--emMaxIter", help="EM maximum number of iterations",
                        type=int, default=500)
    parser.add_argument("--eStepMaxIter",
                        help="E step maximum number of iterations",
                        type=int, default=20)
    parser.add_argument("--mStepEmbeddingMaxIter",
                        help="M step embedding maximum number of iterations",
                        type=int, default=20)
    parser.add_argument("--mStepKernelsMaxIter",
                        help="M step kernels maximum number of iterations",
                        type=int, default=20)
    parser.add_argument("--mStepIndPointsLocsMaxIter",
                        help="M step inducing points locations maximum number of iterations",
                        type=int, default=20)
    parser.add_argument("--metadata_filename_pattern",
                        help="metadata filename pattern", type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--estimation_data_for_matlab_filename_pattern",
                        help="estimation data for matlab filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimationDataForMatlab.mat")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_res_number = args.est_res_number
    n_time_steps_CIF = args.n_time_steps_CIF
    emMaxIter = args.emMaxIter
    eStepMaxIter = args.eStepMaxIter
    mStepEmbeddingMaxIter = args.mStepEmbeddingMaxIter
    mStepKernelsMaxIter = args.mStepKernelsMaxIter
    mStepIndPointsLocsMaxIter = args.mStepIndPointsLocsMaxIter
    metadata_filename_pattern = args.metadata_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern
    estimation_data_for_matlab_filename_pattern = args.estimation_data_for_matlab_filename_pattern

    metadata_filename = metadata_filename_pattern.format(est_res_number)
    metadata = configparser.ConfigParser()
    metadata.read(metadata_filename)
    epoched_spikes_times_filename = metadata["data_params"]["epoched_spikes_times_filename"]
    min_cluster_trials_avg_firing_rate = float(metadata["data_params"]["min_cluster_trials_avg_firing_rate"])
    selected_clusters = [int(aStr) for aStr in metadata["data_params"]["clusters"][1:-1].split(",")]
    selected_trials = [int(aStr) for aStr in metadata["data_params"]["trials"][1:-1].split(",")]

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    with open(model_save_filename, "rb") as f:
        est_results = pickle.load(f)

    # begin recovering spikes_times
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials = jnp.arange(n_trials)

    # subset selected_clusters
    clusters = np.arange(n_neurons)
    spikes_times = mcMazeUtils.subset_clusters_data(
        selected_clusters=selected_clusters,
        clusters=clusters,
        spikes_times=spikes_times,
    )

    # subset selected_trials
    spikes_times, trials_start_times, trials_end_times = \
        mcMazeUtils.subset_trials_data(
            selected_trials=selected_trials,
            trials=trials,
            spikes_times=spikes_times,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    clusters_indices = jnp.arange(n_neurons).tolist()

    trials_durations = [trials_end_times[i] - trials_start_times[i]
                        for i in range(n_trials)]
    spikes_times, clusters_indices = \
        gcnu_common.utils.neural_data_analysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
            spikes_times=spikes_times, clusters_indices=clusters_indices,
            trials_durations=trials_durations,
            min_cluster_trials_avg_firing_rate=min_cluster_trials_avg_firing_rate)
    # done recovering spikes_times

    kernels_params0 = est_results["estimation_params"]["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]

    leg_quad_points = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_points"]
    leg_quad_weights = est_results["estimation_params"]["ell_calculation_params"]["leg_quad_weights"]
    reg_param = est_results["estimation_params"]["optim_params"]["prior_cov_reg_param"]
    qMu0 = est_results["estimation_params"]["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["mean"].squeeze()
    chol_vecs = est_results["estimation_params"]["initial_params"]["posterior_on_latents"]["posterior_on_ind_points"]["cholVecs"]
    C0 = est_results["estimation_params"]["initial_params"]["embedding"]["C0"]
    d0 = est_results["estimation_params"]["initial_params"]["embedding"]["d0"]
    Z0 = est_results["estimation_params"]["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["inducing_points_locs0"]
    trials_lengths = [trials_end_times[i] - trials_start_times[i]
                      for i in range(len(trials_end_times))]
    kernels_types = est_results["kernels_types"]

    # begin generating qSVec0, qSDiag0
    key = jax.random.key(42)
    little_noise_std = 1e-3
    little_noise = jax.random.normal(key,
                                     shape=chol_vecs.shape) * little_noise_std
    qSVec0, qSDiag0 = svGPFA.utils.miscUtils.getQSVecsAndQSDiagsFromQSCholVecs(
        qsCholVecs=chol_vecs+little_noise)
    # done generating qSVec0, qSDiag0

    latentsTrialsTimes = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=trials_start_times,
        end_times=trials_end_times,
        n_steps=n_time_steps_CIF)

    estimation_data_for_matlab_filename = \
        estimation_data_for_matlab_filename_pattern.format(est_res_number)

    svGPFA.utils.miscUtils.saveDataForMatlabEstimations(
        qMu=qMu0, qSVec=qSVec0, qSDiag=qSDiag0,
        C=C0, d=d0,
        indPointsLocs=Z0,
        legQuadPoints=leg_quad_points,
        legQuadWeights=leg_quad_weights,
        kernelsTypes=kernels_types,
        kernelsParams=kernels_params0,
        spikesTimes=spikes_times,
        indPointsLocsKMSRegEpsilon=reg_param,
        trialsLengths=trials_lengths,
        latentsTrialsTimes=latentsTrialsTimes,
        emMaxIter=emMaxIter,
        eStepMaxIter=eStepMaxIter,
        mStepEmbeddingMaxIter=mStepEmbeddingMaxIter,
        mStepKernelsMaxIter=mStepKernelsMaxIter,
        mStepIndPointsMaxIter=mStepIndPointsLocsMaxIter,
        saveFilename=estimation_data_for_matlab_filename)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
