
def lists_are_equal(l1, l2):
    flag = all(x == y for x, y in zip(l1, l2))
    return flag

def get_trials_colors_patterns(trials_df,
                               locations=[[-100, 35],
                                          [-92, 81],
                                          [-77, 82],
                                          [117, 15],
                                          [104, 80],
                                          [125, -64],
                                          [133, -81],
                                          [-91, -70],
                                          [-118, -83]],
                               color_patterns=["rgba(0,0,139,{:f})",       # darkblue
                                               "rgba(0,0,255,{:f})",       # blue
                                               "rgba(173,216,230,{:f})",   # lightblue
                                               "rgba(139,0,0,{:f})",       # darkred
                                               "rgba(255,0,0,{:f})",       # red
                                               "rgba(2,48,32,{:f})",       # darkgreen
                                               "rgba(144,238,144,{:f})",   # lightgreen
                                               "rgba(255,255,0,{:f})",     # yellow
                                               "rgba(255,255,224,{:f})"]): # lightyellow
    n_trials = trials_df.shape[0]
    trials_color_patterns = [None for r in range(n_trials)]
    n_locations = len(locations)

    for r in range(n_trials):
        target_pos = trials_df.iloc[r]["target_pos"]
        active_target = trials_df.iloc[r]["active_target"]
        for i in range(n_locations):
            if lists_are_equal(target_pos[active_target].tolist(), locations[i]):
                trials_color_patterns[r] = color_patterns[i]
                continue
        if trials_color_patterns[r] is None:
            raise ValueError(f"Could not find color pattern for trial {r}")

    return trials_color_patterns


def buildMarkedEventsInfo(events_times, events_colors, events_markers):
    n_events = len(events_times)
    n_trials = len(events_times[0])
    marked_events_times = [None for r in range(n_trials)]
    marked_events_colors = [None for r in range(n_trials)]
    marked_events_markers = [None for r in range(n_trials)]
    for r in range(n_trials):
        trial_marked_events_times = []
        trial_marked_events_colors = []
        trial_marked_events_markers = []
        for i in range(n_events):
            trial_marked_events_times.append(events_times[i][r])
            trial_marked_events_colors.append(events_colors[i])
            trial_marked_events_markers.append(events_markers[i])
        marked_events_times[r] = trial_marked_events_times
        marked_events_colors[r] = trial_marked_events_colors
        marked_events_markers[r] = trial_marked_events_markers
    return marked_events_times, marked_events_colors, marked_events_markers
