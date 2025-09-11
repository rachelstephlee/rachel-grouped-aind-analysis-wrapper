"""
Tools for plotting FIP data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_dynamic_foraging_data_utils import alignment as an
from aind_dynamic_foraging_data_utils import nwb_utils as nu

from aind_dynamic_foraging_basic_analysis.plot.style import FIP_COLORS, STYLE


def plot_fip_psth_compare_alignments(  # NOQA C901
    nwb,
    alignments,
    channel,
    tw=[-4, 4],
    ax=None,
    fig=None,
    censor=True,
    extra_colors={},
    data_column="data",
    error_type="sem",
):
    """
    Compare the same FIP channel aligned to multiple event types
    nwb, nwb object for the session, or a list of nwbs
    alignments, with one session alignments can be either a list of
        event types in df_events, or a dictionary whose keys are
        event types and values are a list of timepoints. With multiple
        sessions, alignments can be either a list of event types in df_events,
        or a list of dictionaries whose keys are event types and values are a
        list of timepoints.
    channel, (str) the name of the FIP channel
    tw, time window for the PSTH
    censor, censor important timepoints before and after aligned timepoints
    extra_colors (dict), a dictionary of extra colors.
        keys should be alignments, or colors are random
    data_column (string), name of data column in nwb.df_fip
    error_type, (string), either "sem" or "sem_over_sessions" to define
        the error bar for the PSTH

    EXAMPLE
    *******************
    plot_fip_psth_compare_alignments(nwb,['left_reward_delivery_time',
        'right_reward_delivery_time'],'G_1_preprocessed')
    """
    if error_type not in ["sem", "sem_over_sessions"]:
        raise Exception("unknown error type")

    nwb_list = nwb if isinstance(nwb, list) else [nwb]
    if len(nwb_list) == 1 and error_type == "sem_over_sessions":
        raise Exception("Cannot have sem_over_sessions with one session")

    for nwb_i in nwb_list:
        if not hasattr(nwb_i, "df_fip"):
            print("You need to compute the df_fip first")
            print("running `nwb.df_fip = create_df_fip(nwb,tidy=True)`")
            nwb_i.df_fip = nu.create_df_fip(nwb_i, tidy=True)
        if not hasattr(nwb_i, "df_events"):
            print("You need to compute the df_events first")
            print("run `nwb.df_events = create_df_events(nwb)`")
            nwb_i.df_events = nu.create_df_events(nwb_i)
        if channel not in nwb_i.df_fip["event"].values:
            print("channel {} not in df_fip".format(channel))

    # if single nwb - can pass list, or dictionary
    # if list of nwbs - can pass a single list, or list of dictionaries
    if len(nwb_list) == 1 and not (isinstance(alignments, list) or isinstance(alignments, dict)):
        raise Exception("Must pass alignments as a list of events, or a dictionary of times")
    elif len(nwb_list) > 1 and (not isinstance(alignments, list)):
        raise Exception(
            "Must pass alignments as a list of events, or a list of dictionaries of times"
        )

    # If we are given a list of dictionaries, ensure all dictionaries have the same keys
    if (
        len(nwb_list) > 1
        and isinstance(alignments, list)
        and all(isinstance(item, dict) for item in alignments)
    ):
        keys = set()
        for d in alignments:
            keys.update(list(d.keys()))
        for index, d in enumerate(alignments):
            missing = keys - set(d.keys())
            if len(missing) > 0:
                raise Exception(
                    "{} Missing alignment key: {}".format(nwb_list[index].session_id, list(missing))
                )

    if isinstance(alignments, dict):
        # We have a single NWB, given a dictionary of alignments, make it a list and we are done
        align_list = [alignments]
    elif isinstance(alignments, list) and all(isinstance(item, dict) for item in alignments):
        align_list = alignments
    elif isinstance(alignments, list):
        align_list = []
        for i, nwb_i in enumerate(nwb_list):
            align_dict = {}
            for a in alignments:
                if a not in nwb_i.df_events["event"].values:
                    print("{} not found in the events table: {}".format(a, nwb_i.session_id))
                    return
                else:
                    align_dict[a] = nwb_i.df_events.query("event == @a")["timestamps"].values
            align_list.append(align_dict)
    else:
        print(
            "alignments must be either a list of events in nwb.df_events, "
            + "or, for a single session, a dictionary where each key is an event type, "
            + "and the value is a list of timepoints. If multiple sessions are given, "
            + "you may pass a list of dictionaries"
        )
        return

    # Compute censor times
    censor_times_list = []
    for i, nwb_i in enumerate(nwb_list):
        censor_times = []
        for key in align_list[i]:
            censor_times.append(align_list[i][key])
        censor_times = np.sort(np.concatenate(censor_times))
        censor_times_list.append(censor_times)

    # Create figure if not supplied
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Get colors
    colors = {**FIP_COLORS, **extra_colors}

    align_label = "Time (s)"
    for alignment in align_list[0]:
        this_align = [x[alignment] for x in align_list]
        etr = fip_psth_multiple_inner_compute(
            nwb_list, this_align, channel, True, tw, censor, censor_times_list, data_column
        )
        fip_psth_inner_plot(ax, etr, colors.get(alignment, ""), alignment, data_column, error_type)

    plt.legend()
    ax.set_xlabel(align_label, fontsize=STYLE["axis_fontsize"])
    if data_column == "data":
        ylabel = "df/f"
    elif data_column == "data_z":
        ylabel = "z-scored df/f"
    else:
        # Default to df/f
        ylabel = "df/f"
    ax.set_ylabel(ylabel, fontsize=STYLE["axis_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(tw)
    ax.axvline(0, color="k", alpha=0.2)
    ax.tick_params(axis="both", labelsize=STYLE["axis_ticks_fontsize"])
    if len(nwb_list) == 1:
        ax.set_title(nwb_list[0].session_id, fontsize=STYLE["axis_fontsize"])
    else:
        ax.set_title("{} sessions".format(len(nwb_list)), fontsize=STYLE["axis_fontsize"])
    plt.tight_layout()
    return fig, ax


def plot_fip_psth_compare_channels(  # NOQA C901
    nwb,
    align,
    tw=[-4, 4],
    ax=None,
    fig=None,
    channels=[
        "G_1_preprocessed",
        "G_2_preprocessed",
        "R_1_preprocessed",
        "R_2_preprocessed",
        "Iso_1_preprocessed",
        "Iso_2_preprocessed",
    ],
    censor=True,
    data_column="data",
    error_type="sem",
):
    """
    nwb, the nwb object for the session of interest, or a list of nwb objects
    align should either be a string of the name of an event type in nwb.df_events,
        or a list of timepoints. if nwb is a list, then align should be a list containing
        lists of timepoints for each session.
    channels should be a list of channel names (strings)
    censor, censor important timepoints before and after aligned timepoints
    data_column (string), name of data column in nwb.df_fip
    error_type, (string), either "sem" or "sem_over_sessions" to define
        the error bar for the PSTH

    EXAMPLE
    ********************
    plot_fip_psth(nwb, 'goCue_start_time')
    plot_fip_psth(nwb_list, 'goCue_start_time')
    plot_fip_psth(nwb_list, [session_1_timepoints, session_2_timepoints, ... ])
    """

    if error_type not in ["sem", "sem_over_sessions"]:
        raise Exception("Unknown error type")

    # Check if nwb is a list, otherwise put it in a list to check
    nwb_list = nwb if isinstance(nwb, list) else [nwb]

    if len(nwb_list) == 1 and error_type == "sem_over_sessions":
        raise Exception("Cannot have sem_over_sessions with one session")
    if isinstance(nwb, list) and isinstance(align, list) and (len(nwb) != len(align)):
        raise Exception("NWB list and align list must match")
    if (
        isinstance(nwb, list)
        and isinstance(align, list)
        and not all(isinstance(item, list) or isinstance(item, np.ndarray) for item in align)
    ):
        raise Exception("When using multiple sessions, align must be a list of lists")
    if isinstance(nwb, list) and isinstance(align, str):
        align = [align] * len(nwb)
    if not isinstance(nwb, list):
        align = [align]

    # First check that each session has an events table and fip table
    for nwb_i in nwb_list:
        if not hasattr(nwb_i, "df_fip"):
            print("You need to compute the df_fip first")
            print("running `nwb.df_fip = create_df_fip(nwb,tidy=True)`")
            nwb_i.df_fip = nu.create_df_fip(nwb_i, tidy=True)
        if not hasattr(nwb_i, "df_events"):
            print("You need to compute the df_events first")
            print("run `nwb.df_events = create_df_events(nwb)`")
            nwb_i.df_events = nu.create_df_events(nwb_i)

        # Add warning if channels are missing
        missing_channels = [c for c in channels if c not in nwb_i.df_fip["event"].values]
        if len(missing_channels) > 0:
            print("{} missing channel {}".format(nwb_i.session_id, missing_channels))

    align_timepoints_list = []
    # Generate the alignment timepoints for each session
    for i, nwb_i in enumerate(nwb_list):
        align_i = align[i]
        if isinstance(align_i, str):
            if align_i not in nwb_i.df_events["event"].values:
                print("{} not found in the events table, {}".format(align_i, nwb_i.session_id))
                return

            align_timepoints_list.append(
                nwb_i.df_events.query("event == @align")["timestamps"].values
            )
            align_label = "Time from {} (s)".format(align_i)
        else:
            align_timepoints_list.append(align_i)
            align_label = "Time (s)"

    # Make figure if not supplied
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Iterate through channels and plot
    colors = [FIP_COLORS.get(c, "") for c in channels]
    for dex, c in enumerate(channels):
        include = [c in nwb.df_fip["event"].values for nwb in nwb_list]
        etr = fip_psth_multiple_inner_compute(
            [x for dex, x in enumerate(nwb_list) if include[dex]],
            [x for dex, x in enumerate(align_timepoints_list) if include[dex]],
            c,
            True,
            tw,
            censor,
            data_column=data_column,
        )
        fip_psth_inner_plot(ax, etr, colors[dex], c, data_column, error_type)

    plt.legend()
    ax.set_xlabel(align_label, fontsize=STYLE["axis_fontsize"])
    if data_column == "data":
        ylabel = "df/f"
    elif data_column == "data_z":
        ylabel = "z-scored df/f"
    else:
        ylabel = "df/f"
    ax.set_ylabel(ylabel, fontsize=STYLE["axis_fontsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(tw)
    ax.axvline(0, color="k", alpha=0.2)
    ax.tick_params(axis="both", labelsize=STYLE["axis_ticks_fontsize"])
    if len(nwb_list) == 1:
        ax.set_title(nwb_list[0].session_id)
    else:
        ax.set_title("{} sessions".format(len(nwb_list)))
    plt.tight_layout()
    return fig, ax


def fip_psth_inner_plot(ax, etr, color, label, data_column, error_type="sem"):
    """
    helper function that plots an event triggered response
    ax, the pyplot axis to plot on
    etr, the dataframe that contains the event triggered response
    color, the line color to plot
    label, the label for the etr
    data_column (string), name of data_column
    error_type, (string), the error bar type to plot, must be a column in etr
    """
    if color == "":
        cmap = plt.get_cmap("tab20")
        color = cmap(np.random.randint(20))
    ax.fill_between(
        etr.index,
        etr[data_column] - etr[error_type],
        etr[data_column] + etr[error_type],
        color=color,
        alpha=0.2,
    )
    ax.plot(etr.index, etr[data_column], color=color, label=label)


def fip_psth_multiple_inner_compute(
    nwb_list,
    align_timepoints_list,
    channel,
    average,
    tw=[-1, 1],
    censor=True,
    censor_times=None,
    data_column="data",
):
    """
    Wrapper function for fip_psth_inner_compute that takes a list of NWB files
    nwb_list, a list of nwb sessions
    align_timepoints_list, a list of alignments for each session
    censor_times, can be None, or a list of timepoints for each session
    """
    # Check that len(nwb_list) = len(align_timepoints_list) = len(censor_times)
    if len(nwb_list) != len(align_timepoints_list):
        raise Exception("length of nwb list and alignments list must match")
    if censor and censor_times is None:
        censor_times = [None] * len(nwb_list)
    if censor and (len(nwb_list) != len(censor_times)):
        raise Exception("length of nwb list and censor times must match")

    etr_list = []
    # Iterate through list of sessions, computing the etr for each
    for i, nwb_i in enumerate(nwb_list):
        etr_i = fip_psth_inner_compute(
            nwb_i,
            align_timepoints_list[i],
            channel,
            average=False,
            tw=tw,
            censor=censor,
            censor_times=censor_times[i],
            data_column=data_column,
        )
        etr_i["ses_idx"] = nwb_i.session_id
        etr_list.append(etr_i)

    # Concat etrs from each session into one dataframe
    etr_all = pd.concat(etr_list, axis=0).reset_index(drop=True)

    if average:
        # Average within each ses_idx for each time point
        mean_per_ses = etr_all.groupby(["ses_idx", "time"])[data_column].mean().unstack("ses_idx")
        # Grand mean: average across ses_idx for each time point
        grand_mean = mean_per_ses.mean(axis=1)
        # SEM over ses_idx for each time point
        grand_sem = mean_per_ses.sem(axis=1)
        # Combine into a DataFrame
        result = grand_mean.to_frame(name=data_column)
        result["sem_over_sessions"] = grand_sem

        # Compute SEM collapsing over sessions
        result["sem"] = etr_all.groupby("time")[data_column].sem()

        return result
    else:
        return etr_all


def fip_psth_inner_compute(
    nwb,
    align_timepoints,
    channel,
    average,
    tw=[-1, 1],
    censor=True,
    censor_times=None,
    data_column="data",
):
    """
    helper function that computes the event triggered response
    nwb, nwb object for the session of interest, should have df_fip attribute
    align_timepoints, an iterable list of the timepoints to compute the ETR aligned to
    channel, what channel in the df_fip dataframe to use
    average(bool), whether to return the average, or all individual traces
    tw, time window before and after each event
    censor, censor important timepoints before and after aligned timepoints
    censor_times, timepoints to censor
    data_column (string), name of data column in nwb.df_fip

    """

    data = nwb.df_fip.query("event == @channel")
    etr = an.event_triggered_response(
        data,
        "timestamps",
        data_column,
        align_timepoints,
        t_start=tw[0],
        t_end=tw[1],
        output_sampling_rate=40,
        censor=censor,
        censor_times=censor_times,
    )

    if average:
        mean = etr.groupby("time").mean()
        sem = etr.groupby("time").sem()
        mean["sem"] = sem[data_column]
        return mean
    return etr


def plot_histogram(nwb, preprocessed=True, edge_percentile=2, data_column="data"):
    """
    Generates a histogram of values of each FIP channel
    preprocessed (Bool), if True, uses the preprocessed channel
    edge_percentile (float), displays only the (2, 100-2) percentiles of the data
    data_column (string), name of data column in nwb.df_fip

    EXAMPLE
    ***********************
    plot_histogram(nwb)
    """
    if not hasattr(nwb, "df_fip"):
        print("You need to compute the df_fip first")
        print("running `nwb.df_fip = create_df_fip(nwb,tidy=True)`")
        nwb.df_fip = nu.create_df_fip(nwb, tidy=True)
        return

    fig, ax = plt.subplots(3, 2, sharex=True)
    channels = ["G", "R", "Iso"]
    mins = []
    maxs = []
    for i, c in enumerate(channels):
        for j, count in enumerate(["1", "2"]):
            if preprocessed:
                dex = c + "_" + count + "_preprocessed"
            else:
                dex = c + "_" + count
            df = nwb.df_fip.query("event == @dex")
            ax[i, j].hist(df[data_column], bins=1000, color=FIP_COLORS.get(dex, "k"))
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            if preprocessed:
                ax[i, j].set_xlabel("df/f")
            else:
                ax[i, j].set_xlabel("f")
            ax[i, j].set_ylabel("count")
            ax[i, j].set_title(dex)
            mins.append(np.percentile(df[data_column].values, edge_percentile))
            maxs.append(np.percentile(df[data_column].values, 100 - edge_percentile))
    ax[0, 0].set_xlim(np.min(mins), np.max(maxs))
    fig.suptitle(nwb.session_id)
    plt.tight_layout()
