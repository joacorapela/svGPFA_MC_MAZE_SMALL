
import sys
import argparse
import configparser
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pynwb import NWBHDF5IO

import utils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--dandiset_ID", help="dandiset ID", type=str,
                        default="000128")
    parser.add_argument("--filepath", help="dandi filepath", type=str,
                        default="../../data/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb")
    parser.add_argument("--epoch_event_name", help="epoch event name",
                        type=str, default="move_onset_time")
    parser.add_argument("--epoched_hand_pos_filename_pattern",
                        help="epched spikes times filename pattern", type=str,
                        default="../../results/00000000_dandisetID{:s}_epochedEvent{:s}_epochedHandPos.{:s}")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern", type=str,
                        default="../../figures/00000000_dandisetID{:s}_epochedEvent{:s}_epochedHandPos.{:s}")
    args = parser.parse_args()

    dandiset_ID = args.dandiset_ID
    filepath = args.filepath
    epoch_event_name = args.epoch_event_name
    epoched_hand_pos_filename_pattern = args.epoched_hand_pos_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    epoched_hand_pos_filename = \
        epoched_hand_pos_filename_pattern.format(
            dandiset_ID, epoch_event_name, "pickle")
    with open(epoched_hand_pos_filename, "rb") as f:
        load_res = pickle.load(f)

    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        trials_df = nwbfile.intervals["trials"].to_dataframe()

    marker_symbol_active = "cross"
    marker_symbol_inactive = "circle"
    marker_color_inactive = "gray"

    trace_color_patterns = utils.get_trials_color_patterns(trials_df=trials_df)

    fig = go.Figure()
    hand_pos_list = load_res["hand_pos_list"]
    for r, hand_pos in enumerate(hand_pos_list):
        target_pos = trials_df.iloc[r]["target_pos"]
        trial_version = trials_df.iloc[r]["trial_version"]
        num_targets = trials_df.iloc[r]["num_targets"]
        trace_color = trace_color_patterns[r].format(1.0)
        active_target = trials_df.iloc[r]["active_target"]

        trace = go.Scatter(x=hand_pos["x"], y=hand_pos["y"],
                           line_color=trace_color,
                           text=hand_pos.index,
                           hovertemplate=f"<b>trial</b>: {r:02d}<br>" +
                                         f"<b>target_pos</b>: [{target_pos[active_target][0]},{target_pos[active_target][1]}]<br>" +
                                         f"<b>trial_version</b>: {trial_version}<br>" +
                                          "<b>time</b>: %{text:.4f}",
                           name=f"trial={r:02d}",
                           legendgroup=f"trial={r:02d}",
                          )
        fig.add_trace(trace)

        for i in range(num_targets):
            if i == active_target:
                marker_symbol = marker_symbol_active
                target_status = "active"
            else:
                marker_symbol = marker_symbol_inactive
                target_status = "inactive"
            trace = go.Scatter(x=[target_pos[i, 0]], y=[target_pos[i, 1]],
                               mode="markers",
                               marker_symbol=marker_symbol,
                               marker_color=trace_color,
                               legendgroup=f"trial={r:02d}",
                               hovertemplate=f"<b>trial</b>:{r:02d}<br>"+
                                             f"<b>target_pos</b>: [{target_pos[i,0]},{target_pos[i,1]}]<br>"+
                                             f"<b>target_status</b>:{target_status}<br>",
                               name=f"trial={r:02d}",
                               showlegend=False,
                              )
            fig.add_trace(trace)

    fig.update_xaxes(title="x")
    fig.update_yaxes(title="y")

    html_fig_filename = fig_filename_pattern.format(dandiset_ID,
                                                    epoch_event_name, "html")
    fig.write_html(html_fig_filename)
    png_fig_filename = fig_filename_pattern.format(dandiset_ID,
                                                   epoch_event_name, "png")
    fig.write_image(png_fig_filename)
    breakpoint()


if __name__ == "__main__":
    main(sys.argv)
