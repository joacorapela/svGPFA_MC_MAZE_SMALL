
import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pynwb import NWBHDF5IO

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000140")
    parser.add_argument("--filepath_pattern", help="dandi filepath pattern", type=str,
                        default="../../data/{:s}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--epoched_hand_pos_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedHandPos.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/00000000_dandisetID{:s}_uniqueActiveTargets.{:s}")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath_pattern = args.filepath_pattern
    epoch_event_name = args.epoch_event_name
    epoched_hand_pos_filename_pattern = args.epoched_hand_pos_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    filepath = filepath_pattern.format(dandiset_ID)
    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    marker_symbol_active = "cross"
    marker_symbol_inactive = "circle"
    fig = go.Figure()
    # hand_pos_list = load_res["hand_pos_list"]
    # for r, hand_pos in enumerate(hand_pos_list):
    active_target_pos = []
    for r in range(trials_df.shape[0]):
        target_pos = trials_df.iloc[r]["target_pos"]
        num_targets = trials_df.iloc[r]["num_targets"]
        active_target = trials_df.iloc[r]["active_target"]
        for i in range(num_targets):
            if i == active_target:
                active_target_pos.append(target_pos[i].tolist())

    unique_active_target_pos = np.unique(np.array(active_target_pos), axis=0)

    for i in range(unique_active_target_pos.shape[0]):
        trace = go.Scatter(x=[unique_active_target_pos[i, 0]],
                           y=[unique_active_target_pos[i, 1]],
                           mode="markers",
                           name="[{:d}, {:d}]".format(unique_active_target_pos[i, 0], unique_active_target_pos[i, 1]),
                          )
        fig.add_trace(trace)

    fig.update_xaxes(title="x")
    fig.update_yaxes(title="y")

    html_fig_filename = fig_filename_pattern.format(dandiset_ID, "html")
    fig.write_html(html_fig_filename)
    png_fig_filename = fig_filename_pattern.format(dandiset_ID, "png")
    fig.write_image(png_fig_filename)

    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
