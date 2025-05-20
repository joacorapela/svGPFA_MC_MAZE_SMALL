import sys
import os.path
import random
import torch
import pickle
import argparse
import configparser
import cProfile
import numpy as np

import gcnu_common.utils.neural_data_analysis
import gcnu_common.utils.config_dict
import svGPFA.stats.modelFactory
import svGPFA.stats.em
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils

import mcMazeUtils

# import svGPFA.utils.my_globals

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=8)
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--n_latents", help="number of latent processes",
                        type=int, default=5)
    parser.add_argument("--n_threads", help="number of threads for PyTorch",
                        type=int, default=6)
    parser.add_argument("--common_n_ind_points",
                        help="common number of inducing points",
                        type=int, default=8)
    parser.add_argument("--profile",
                        help="use this option if you want to profile svGPFA.maximize()",
                        action="store_true")
    parser.add_argument("--epoched_spikes_times_filename_pattern",
                        help="epoched spikes times filename pattern",
                        type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedSpikesTimes.{:s}")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../init/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estim_res_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--profiling_info_filename_pattern",
                        help="profiling information filename pattern",
                        type=str,
                        default="../../results/{:08d}_profiling_info.txt")
    parser.add_argument("--trials_ids_filename", help="trials ids filename",
                        type=str, default="../../init/trialsIDs_0_99.csv")
    parser.add_argument("--clusters_ids_filename", help="clusters ids filename",
                        type=str, default="../../init/clustersIDs_0_141.csv")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    est_init_number = args.est_init_number
    dandiset_ID = args.dandiset_ID
    epoch_event_name = args.epoch_event_name
    n_latents = args.n_latents
    n_threads = args.n_threads
    common_n_ind_points = args.common_n_ind_points
    profile = args.profile
    epoched_spikes_times_filename_pattern = args.epoched_spikes_times_filename_pattern
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    estim_res_metadata_filename_pattern = \
        args.estim_res_metadata_filename_pattern
    profiling_info_filename_pattern = args.profiling_info_filename_pattern
    trials_ids_filename = args.trials_ids_filename
    clusters_ids_filename = args.clusters_ids_filename
    model_save_filename_pattern = args.model_save_filename_pattern

    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    # fixing a bug in PyTorch
    # https://github.com/pytorch/pytorch/issues/90760
    torch.set_num_threads(torch.get_num_threads())

    # max_trial_duration = float(est_init_config["data_params"]["max_trial_duration"])
    min_neuron_trials_avg_firing_rate = float(est_init_config["data_params"]["min_neuron_trials_avg_firing_rate"])

    # get spike_times
    epoched_spikes_times_filename = \
        epoched_spikes_times_filename_pattern.format(dandiset_ID,
                                                     epoch_event_name,
                                                     "pickle")
    with open(epoched_spikes_times_filename, "rb") as f:
        load_res = pickle.load(f)
    spikes_times = load_res["spikes_times"]
    trials_start_times = load_res["trials_start_times"]
    trials_end_times = load_res["trials_end_times"]

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_ids = np.arange(n_trials)

    # subset selected_clusters_ids
    selected_clusters_ids = np.genfromtxt(clusters_ids_filename,
                                          dtype=np.uint64)
    clusters_ids = np.arange(n_neurons)
    spikes_times = mcMazeUtils.subset_clusters_ids_data(
        selected_clusters_ids=selected_clusters_ids,
        clusters_ids=clusters_ids,
        spikes_times=spikes_times,
    )

    # subset selected_trials_ids
    selected_trials_ids = np.genfromtxt(trials_ids_filename, dtype=np.uint64)
    spikes_times, trials_start_times, trials_end_times = \
            mcMazeUtils.subset_trials_ids_data(
                selected_trials_ids=selected_trials_ids,
                trials_ids=trials_ids,
                spikes_times=spikes_times,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)

    # breakpoint()

    n_trials = len(spikes_times)
    # trials_indices = np.arange(n_trials)
    n_neurons = len(spikes_times[0])
    neurons_indices = np.arange(n_neurons)

    trials_durations = [trials_end_times[i] - trials_start_times[i]
                        for i in range(n_trials)]
    spikes_times, neurons_indices = \
        gcnu_common.utils.neural_data_analysis.removeUnitsWithLessTrialAveragedFiringRateThanThr(
            spikes_times=spikes_times, neurons_indices=neurons_indices,
            trials_durations = trials_durations,
            min_neuron_trials_avg_firing_rate=min_neuron_trials_avg_firing_rate)
    clusters_ids = [clusters_ids[i] for i in neurons_indices]

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # breakpoint()

    #    build dynamic parameter specifications
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params_spec = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #   build config file parameters specification
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params_spec = \
        svGPFA.utils.initUtils.getParamsDictFromStringsDict(
            n_latents=n_latents, n_trials=n_trials,
            strings_dict=strings_dict, args_info=args_info)
    #    build default parameter specificiations
    default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents,
        common_n_ind_points=common_n_ind_points)
    #    finally, get the parameters from the dynamic,
    #    configuration file and default parameter specifications
    params, kernels_types, = \
        svGPFA.utils.initUtils.getParamsAndKernelsTypes(
            n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
            trials_start_times=trials_start_times,
            trials_end_times=trials_end_times,
            dynamic_params_spec=dynamic_params_spec,
            config_file_params_spec=config_file_params_spec)
            # config_file_params_spec=config_file_params_spec,
            # default_params_spec=default_params_spec)

    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estim_res_metadata_filename = \
            estim_res_metadata_filename_pattern.format(estResNumber)
        if not os.path.exists(estim_res_metadata_filename):
            estPrefixUsed = False
    modelSaveFilename = model_save_filename_pattern.format(estResNumber)
    if profile:
        profiling_info_filename_pattern = \
            profiling_info_filename_pattern.format(estResNumber)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # create model
    kernelMatrixInvMethod = svGPFA.stats.modelFactory.kernelMatrixInvChol
    indPointsCovRep = svGPFA.stats.modelFactory.indPointsCovChol
    model = svGPFA.stats.modelFactory.ModelFactory.buildModelPyTorch(
        conditionalDist=svGPFA.stats.modelFactory.PointProcess,
        linkFunction=svGPFA.stats.modelFactory.ExponentialLink,
        preIntensityType=svGPFA.stats.modelFactory.LinearPreIntensity,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    spikes_times = [[torch.tensor(spikes_times[r][n]) for n in range(n_neurons)]
                    for r in range(n_trials)]
    model.setParamsAndData(
        measurements=spikes_times,
        initial_params=params["initial_params"],
        eLLCalculationParams=params["ell_calculation_params"],
        priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

    # save estimated values
    estim_res_config = configparser.ConfigParser()
    estim_res_config["data_params"] = {
        "n_threads": n_threads,
        "trials_ids": selected_trials_ids,
        "neurons_indices": neurons_indices,
        "clusters_ids": selected_clusters_ids,
        "nLatents": n_latents,
        "common_n_ind_points": common_n_ind_points,
        # "max_trial_duration": max_trial_duration,
        "min_neuron_trials_avg_firing_rate": min_neuron_trials_avg_firing_rate,
        "epoched_spikes_times_filename": epoched_spikes_times_filename,
    }
    # estim_res_config["optim_params"] = params["optim_params"]
    estim_res_config["estimation_params"] = {"est_init_number":
                                             est_init_number}
    with open(estim_res_metadata_filename, "w") as f:
        estim_res_config.write(f)
    print(f"Saved {estim_res_metadata_filename}")

    # maximize lower bound
    def getSVPosteriorOnIndPointsParams(model, get_mean=True, latent=0, trial=0):
        params = model.getSVPosteriorOnIndPointsParams()
        base_index = 0
        if not get_mean:
            base_index = len(params)/2 - 1
        answer = params[base_index][trial, :, 0]
        return answer

    def getKernelsParams(model):
        params = model.getKernelsParams()
        return params

    # maximize lower bound
    em = svGPFA.stats.em.EM_PyTorch()
    if profile:
        pr = cProfile.Profile()
        pr.enable()

#     svGPFA.utils.my_globals.raise_exception = True

    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        em.maximizeInSteps(model=model, optim_params=params["optim_params"],
                      method=params["optim_params"]["optim_method"],
                      # getIterationModelParamsFn=getSVPosteriorOnIndPointsParams,
                      getIterationModelParamsFn=getKernelsParams,
                      printIterationModelParams=True)

    if profile:
        pr.disable()
        pr.dump_stats(filename=profiling_info_filename)

    resultsToSave = {
                     "trials_start_times": trials_start_times,
                     "trials_end_times": trials_end_times,
                     "lowerBoundHist": lowerBoundHist,
                     "elapsedTimeHist": elapsedTimeHist,
                     "terminationInfo": terminationInfo,
                     "iterationModelParams": iterationsModelParams,
                     "spikes_times": spikes_times,
                     "trials_ids": selected_trials_ids,
                     "clusters_ids": selected_clusters_ids,
                     "model": model,
                    }
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
        print("Saved results to {:s}".format(modelSaveFilename))

    print(f"Elapsed time {elapsedTimeHist[-1]}")

    # breakpoint()


if __name__ == "__main__":
    main(sys.argv)
