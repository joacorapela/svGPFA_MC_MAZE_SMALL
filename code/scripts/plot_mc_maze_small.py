
"""
Recordings from the primary motor and dorsal premotor cortex striatum of a monkey performing a delayed-reach task
=================================================================================================================

In this notebook we download publically available data from Dandi, epoch it, run svGPFA and plot its results.

"""

#%%
# Setup environment
# -----------------

#%%
# Import required packages
# ^^^^^^^^^^^^^^^^^^^^^^^^

import sys
import warnings
import pickle
import time
import configparser
import numpy as np
import pandas as pd
import torch

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import gcnu_common.utils.neural_data_analysis
import gcnu_common.stats.pointProcesses.tests
import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import svGPFA.plot.plotUtilsPlotly

#%%
# Get model
# ^^^^^^^^^
em_max_iter_dyn = 200
model_save_filename = \
    f"../../../../svGPFA/examples/results/jenkins_small_model_emMaxIter{em_max_iter_dyn}.pickle"

#%%
# Set figures filenames
# ^^^^^^^^^^^^^^^^^^^^^
estResNumber = 1
latent_to_plot = 0
latents_to_3D_plot = [0, 1, 2]
neuron_to_plot = 0
trial_to_plot = 0

ksTestTimeRescalingNumericalCorrectionFigFilenamePattern = "../../figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron_{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
rocFigFilenamePattern = "../../figures/{:08d}_predictive_analysis_trial{:03d}_neuron_{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
lowerBoundHistVsIterNoFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVSIterNo.{{:s}}".format(estResNumber)
lowerBoundHistVsElapsedTimeFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
latentsFigFilenamePattern = "../../figures/{:08d}_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
orthonormalizedLatentsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
latents_to_3D_plot_str = "".join(str(i)+"_" for i in latents_to_3D_plot)
orthonormalizedLatents3DFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latents{:s}.{{:s}}".format(estResNumber, latents_to_3D_plot_str)
embeddingsFigFilenamePattern = "../../figures/{:08d}_embedding_clusterID_{:d}.{{:s}}".format(estResNumber, neuron_to_plot)
CIFsOneNeuronAllTrialsFigFilenamePattern = "../../figures/{:08d}_intensityFunctionOneNeuronAllTrials_neuron_{:03d}.{{:s}}".format(estResNumber, neuron_to_plot)
orthonormalizedEmbeddingParamsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_embedding_params.{{:s}}".format(estResNumber)
kernelsParamsFigFilenamePattern = "../../figures/{:08d}_kernels_params.{{:s}}".format(estResNumber)

#%%
# Download data
# ^^^^^^^^^^^^^
dandiset_id = "000140"
filepath = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
epoch_event_name = "move_onset_time"
with DandiAPIClient() as client:
	asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
	s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

io = NWBHDF5IO(s3_path, mode="r", driver="ros3")
nwbfile = io.read()
units = nwbfile.units
units_df = units.to_dataframe()

# n_neurons
n_neurons = units_df.shape[0]

# continuous spikes times
continuous_spikes_times = [None for r in range(n_neurons)]
for n in range(n_neurons):
    continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

# trials
trials_df = nwbfile.intervals["trials"].to_dataframe()

# n_trials
n_trials = trials_df.shape[0]

#%%
# Epoch spikes times
# ^^^^^^^^^^^^^^^^^^
trials_start_times = [None for r in range(n_trials)]
trials_end_times = [None for r in range(n_trials)]
spikes_times = [[None for n in range(n_neurons)] for r in range(n_trials)]
for n in range(n_neurons):
    for r in range(n_trials):
        epoch_start_time = trials_df.iloc[r]["start_time"]
        epoch_end_time = trials_df.iloc[r]["stop_time"]
        epoch_time = trials_df.iloc[r][epoch_event_name]
        spikes_times[r][n] = (continuous_spikes_times[n][
            np.logical_and(epoch_start_time <= continuous_spikes_times[n],
                           continuous_spikes_times[n] <= epoch_end_time)] -
            epoch_time)
        trials_start_times[r] = epoch_start_time - epoch_time
        trials_end_times[r] = epoch_end_time - epoch_time

#%%
# Load model
# ----------
with open(model_save_filename, "rb") as f:
   estResults = pickle.load(f)
lowerBoundHist = estResults["lowerBoundHist"]
elapsedTimeHist = estResults["elapsedTimeHist"]
model = estResults["model"]

#%%
# Goodness-of-fit analysis
# ------------------------

#%%
# Set goodness-of-fit variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ks_test_gamma = 10
trial_for_gof = 0
neuron_for_gof = 0
n_time_steps_IF = 100

trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_IF)

#%%
# Calculate expected intensity function values (for KS test and IF plots)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with torch.no_grad():
    cif_values = model.computeExpectedPosteriorCIFs(times=trials_times)
cif_values_GOF = cif_values[trial_for_gof][neuron_for_gof]

#%%
# KS time-rescaling GOF test
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
trial_times_GOF = trials_times[trial_for_gof, :, 0]
spikes_times_GOF = spikes_times[trial_for_gof][neuron_for_gof]
if len(spikes_times_GOF) == 0:
    raise ValueError("No spikes found for goodness-of-fit analysis")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
        gcnu_common.stats.pointProcesses.tests.\
        KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF,
            cif_times=trial_times_GOF, cif_values=cif_values_GOF,
            gamma=ks_test_gamma)

title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(
    trial_for_gof, neuron_for_gof, len(spikes_times_GOF))
fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig.write_image(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("png"))
fig.write_html(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("html"))

#%%
# ROC predictive analysis
# ^^^^^^^^^^^^^^^^^^^^^^^
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fpr, tpr, roc_auc = svGPFA.utils.miscUtils.computeSpikeClassificationROC(
        spikes_times=spikes_times_GOF,
        cif_times=trial_times_GOF,
        cif_values=cif_values_GOF)
fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(
    fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
fig.write_image(rocFigFilenamePattern.format("png"))
fig.write_html(rocFigFilenamePattern.format("html"))

#%%
# Plotting
# --------

#%%
# Imports for plotting
# ^^^^^^^^^^^^^^^^^^^^

import numpy as np
import pandas as pd

#%%
# Set plotting variables
# ^^^^^^^^^^^^^^^^^^^^^^

trials_ids = np.arange(n_trials)
neurons_ids = np.arange(n_neurons)
choices_colors_patterns = ["rgba(0,0,255,{:f})", "rgba(255,0,0,{:f})"]
align_event_name = "response_times"
events_names = ["target_on_time", "go_cue_time", "move_onset_time"]
events_colors = ["magenta", "green", "black"]
events_markers = ["circle", "circle", "circle"]

#%%
# Plot lower bound history
# ^^^^^^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
    elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

#%%
# Plot estimated latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, testVarK = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
    times=trials_times.numpy(),
    latentsMeans=testMuK,
    latentsSTDs=torch.sqrt(testVarK),
    trials_ids=trials_ids,
    latentToPlot=latent_to_plot,
    xlabel="Time (msec)")
fig.write_image(latentsFigFilenamePattern.format("png"))
fig.write_html(latentsFigFilenamePattern.format("html"))

#%%
# Plot orthonormalized estimated latent across trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
testMuK, _ = model.predictLatents(times=trials_times)
test_mu_k_np = [testMuK[r].detach().numpy() for r in range(len(testMuK))]
estimatedC, estimatedD = model.getSVEmbeddingParams()
estimatedC_np = estimatedC.detach().numpy()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
    trials_times=trials_times, latentsMeans=test_mu_k_np, latentToPlot=latent_to_plot,
    C=estimatedC_np, trials_ids=trials_ids, xlabel="Time (msec)")
fig.write_image(orthonormalizedLatentsFigFilenamePattern.format("png"))
fig.write_html(orthonormalizedLatentsFigFilenamePattern.format("html"))

#%%
# Plot 3D scatter plot of orthonormalized latents
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
    trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
    C=estimatedC_np, trials_ids=trials_ids,
    latentsToPlot=latents_to_3D_plot)
fig.write_image(orthonormalizedLatents3DFigFilenamePattern.format("png"))
fig.write_html(orthonormalizedLatents3DFigFilenamePattern.format("html"))

#%%
# Plot embedding
# ^^^^^^^^^^^^^^
embeddingMeans, embeddingVars = model.predictEmbedding(times=trials_times)
embeddingMeans = embeddingMeans.detach().numpy()
embeddingVars = embeddingVars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
    times=trials_times.numpy(),
    embeddingsMeans=embeddingMeans[:, :, neuron_to_plot],
    embeddingsSTDs=np.sqrt(embeddingVars[:, :, neuron_to_plot]),
    title=title)
fig.write_image(embeddingsFigFilenamePattern.format("png"))
fig.write_html(embeddingsFigFilenamePattern.format("html"))

#%%
# Plot intensity functions for one neuron and all trials
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
title = f"Neuron: {neuron_to_plot}"
fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
    trials_times=trials_times,
    cif_values=cif_values,
    neuron_index=neuron_to_plot,
    spikes_times=spikes_times,
    trials_ids=trials_ids,
    title=title)
fig.write_image(CIFsOneNeuronAllTrialsFigFilenamePattern.format("png"))
fig.write_html(CIFsOneNeuronAllTrialsFigFilenamePattern.format("html"))

#%%
# Plot orthonormalized embedding parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
hovertemplate = "value: %{y}<br>" + \
                "neuron index: %{x}<br>" + \
                "%{text}"
text = [f"neuron: {neuron}" for neuron in neurons_ids]
estimatedC, estimatedD = model.getSVEmbeddingParams()
fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedEmbeddingParams(
    C=estimatedC.numpy(), d=estimatedD.numpy(),
    hovertemplate=hovertemplate, text=text)
fig.write_image(
    orthonormalizedEmbeddingParamsFigFilenamePattern.format("png"))
fig.write_html(
    orthonormalizedEmbeddingParamsFigFilenamePattern.format("html"))

#%%
# Plot kernel parameters
# ^^^^^^^^^^^^^^^^^^^^^^
kernelsParams = model.getKernelsParams()
kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
    kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
fig.write_image(
    kernelsParamsFigFilenamePattern.format("png"))
fig.write_html(
    kernelsParamsFigFilenamePattern.format("html"))

#%%
# .. raw:: html
#
#    <h3><font color="red">To run the Python script or Jupyter notebook below,
#    please download them to the <i>examples/sphinx_gallery</i> folder of the
#    repository and execute them from there.</font></h3>

# sphinx_gallery_thumbnail_path = '_static/ibl_logo.png'
