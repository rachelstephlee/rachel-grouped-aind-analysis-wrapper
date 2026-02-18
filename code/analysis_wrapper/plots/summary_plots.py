import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from aind_dynamic_foraging_basic_analysis.plot import plot_fip as pf
from aind_dynamic_foraging_basic_analysis.plot import plot_foraging_session as pb
from aind_dynamic_foraging_basic_analysis.plot import plot_session_scroller as pss
import rachel_analysis_utils.analysis_utils as analysis_utils

import matplotlib.gridspec as gridspec


import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import seaborn as sns




N_COLS_PER_ROW = 5
FONTSIZE = 14
mpl.rcParams.update({
    "font.size": FONTSIZE,
    "legend.fontsize": FONTSIZE+ 5,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "figure.titlesize": FONTSIZE
})



def plot_RPE_by_avg_signal(df_trials, avg_signal_col, ax):

    # clean data, split df_trials
    df_trials_clean = df_trials.dropna(subset = [avg_signal_col, 'RPE_earned'])

    if len(df_trials_clean) == 0:
        print("no RPE_earned values found, skipping plot")
        return ax

    df_trials_neg = df_trials_clean.query('RPE_earned < 0')
    df_trials_pos = df_trials_clean.query('RPE_earned >= 0')


    # Scatter for RPE_earned < 0
    if not df_trials_neg.empty:
        sns.scatterplot(
            data=df_trials_neg,
            x='RPE_earned',
            y=avg_signal_col ,
            hue='ses_idx',
            s=40,
            alpha=0.7,
            legend=False,
            ax = ax
        )
        (x_fit, y_fit, slope) = analysis_utils.get_RPE_by_avg_signal_fit(df_trials_neg, avg_signal_col)
        ax.plot(x_fit, y_fit, color='blue', lw=2, label=f'RPE < 0: {slope:.3f}')


    # lmplot for RPE_earned > 0 (overlay line only)
    if not df_trials_pos.empty:
        # Plot scatter for positive as well for context
        sns.scatterplot(
            data=df_trials_pos,
            x='RPE_earned',
            y=avg_signal_col,
            hue='ses_idx',
            s=40,
            alpha=0.7,
            marker='X',
            legend=False,
            ax = ax
        )
        (x_fit, y_fit, slope) = analysis_utils.get_RPE_by_avg_signal_fit(df_trials_pos, avg_signal_col)
        ax.plot(x_fit, y_fit, color='red', lw=2, label=f'RPE >= 0: {slope:.3f}')
    ax.set_xlabel('RPE_earned')
    ax.set_ylabel(avg_signal_col)
    ax.legend(framealpha=0.5, title = "LM fitted slopes", fontsize='small')

    return ax



def plot_row_panels_RPE(nwbs, channel, panels):
    """
    Plot a row of summary panels for a given set of NWB sessions and a specific channel.

    This function generates a set of five panels summarizing neural and behavioral data:
        1. PSTH for left vs right choices.
        2. PSTH split by RPE-binned3 categories.
        3. Baseline z-scored df/f by number of past rewards.
        4. PSTH split by RPE-binned3 with baseline removed.
        5. Scatter and regression of RPE vs average signal.

    Args:
        nwbs (list): List of NWB session objects, each with a .df_trials DataFrame.
        channel (str): Channel name to plot.
        panels (list): List of matplotlib Axes objects (length 5) to plot into.

    Returns:
        list: The input list of matplotlib Axes, with plots drawn.
    """
    trial_width_choice = [-1, 4]
    df_trials_all = pd.concat([nwb.df_trials for nwb in nwbs])
    RPE_binned3_label_names = df_trials_all['RPE-binned3'].cat.categories.astype(str).tolist()
    if len(nwbs) > 1:
        error_type = 'sem_over_sessions'
        data_col = 'data_z'
    else:
        error_type = 'sem'
        data_col = 'data'
    
    # 1. Choice L vs R
    pf.plot_fip_psth_compare_alignments(
            nwbs,
            [{"left": nwb.df_trials.query("choice == 0").choice_time_in_session.values,
              "right": nwb.df_trials.query("choice == 1").choice_time_in_session.values} for nwb in nwbs],
            channel,
            tw=trial_width_choice,
            extra_colors={"left": 'b', "right": 'r'},
            data_column=data_col,
            error_type=error_type,
            ax=panels[0],
        )

    # 2 RPE_binned3
    get_RPE_binned3_dfs = lambda df_trials: [
        df_trials[df_trials['RPE-binned3'] == RPE]['choice_time_in_session'].values for RPE in RPE_binned3_label_names
    ]
    RPE_binned3_dfs_dicts = [
        dict(zip(RPE_binned3_label_names, get_RPE_binned3_dfs(nwb.df_trials))) for nwb in nwbs
    ]
    pf.plot_fip_psth_compare_alignments(
            nwbs, RPE_binned3_dfs_dicts, channel,
            extra_colors=dict(zip(RPE_binned3_label_names, sns.color_palette("mako", len(RPE_binned3_label_names)).as_hex())),
            tw=trial_width_choice, censor=True, data_column=data_col, error_type=error_type, ax=panels[1]
        )

    # 3. Baseline by num_reward_past (grand mean/SE)
    df_trials_all = df_trials_all.query(
        'num_reward_past > -7 and num_reward_past < 7'
    ).sort_values('trial')
    if len(nwbs) > 1:
        grouped = (
                df_trials_all
                .groupby(['ses_idx', 'num_reward_past'])[f'{data_col}_{channel}_baseline']
                .mean()
                .reset_index()
            )
        agg = (
                grouped
                .groupby('num_reward_past')[f'{data_col}_{channel}_baseline']
                .agg(['mean', 'sem'])
                .reset_index()
            )
        panels[2].bar(
                agg['num_reward_past'],
                agg['mean'],
                yerr=agg['sem'],
                color=sns.color_palette('vlag', len(agg)),
                capsize=4,
        )
        panels[2].set_ylabel(f'{data_col}_baseline')
    else:
        sns.barplot(
                x='num_reward_past',
                y=f'{data_col}_{channel}_baseline',
                data=df_trials_all,
                palette='vlag',
                hue='num_reward_past',
                errorbar='se',
                dodge=False,
                legend=False,
                ax=panels[2]
            )
    panels[2].set_title('Baseline of z-scored df/f')

    # 4 RPE_binned_3 with baseline removed
    pf.plot_fip_psth_compare_alignments(
            nwbs, RPE_binned3_dfs_dicts, channel,
            extra_colors=dict(zip(RPE_binned3_label_names, sns.color_palette("mako", len(RPE_binned3_label_names)).as_hex())),
            tw=trial_width_choice, censor=True, data_column=f"{data_col}_norm", error_type=error_type, ax=panels[3]
        )
    panels[3].set_ylabel('z-scored df/f \n (baseline removed)')

    # 5. Add the RPE vs avg signal
    avg_signal_cols = [c for c in df_trials_all.columns if c.startswith("avg_data") and channel[:3] in c]

    if len(avg_signal_cols) != 1:
        print("incorrect number of avg_signal_col found, skipping RPE vs avg signal plot")
        print(avg_signal_cols)
        return panels

    plot_RPE_by_avg_signal(df_trials_all, avg_signal_cols[0], ax = panels[4])
    # panels[4].set_xlim([-1, 1])
    
    for ax in panels:
        ax.set_title("")
        ax.set_xlabel("")

    for idx in [0, 1, 3]:
        panels[idx].legend([])
    
    return panels

def plot_row_panels_left_right_RPE(nwb_split, channel, panels, offsets):
    data_col = 'data'
    error_type = 'sem'

    trial_width_choice = [-1, 4]

    RPE_binned3_label_names = nwb_split.df_trials['RPE-binned3'].cat.categories.astype(str).tolist()

    # RPE
    get_RPE_binned3_dfs = lambda df_trials: [
        df_trials[df_trials['RPE-binned3'] == RPE]['choice_time_in_session'].values for RPE in RPE_binned3_label_names
    ]

    for i, df_trials_ch in enumerate([nwb_split.df_trials_left, nwb_split.df_trials_right]):

        # RPE 
        RPE_binned3_dfs_dicts = dict(zip(RPE_binned3_label_names, get_RPE_binned3_dfs(df_trials_ch)))

        pf.plot_fip_psth_compare_alignments(
                nwb_split, RPE_binned3_dfs_dicts, channel,
                extra_colors=dict(zip(RPE_binned3_label_names, sns.color_palette("mako", len(RPE_binned3_label_names)).as_hex())),
                tw=trial_width_choice, censor=True, data_column=data_col, error_type=error_type, ax=panels[i*3 + i]
            )
        panels[i*3 + i].set_title("")

        # BASELINE

        df_trials_ch = df_trials_ch.query(
            'num_reward_past > -7 and num_reward_past < 7'
        ).sort_values('trial')

        sns.barplot(
            x='num_reward_past',
            y=f'{data_col}_{channel}_baseline',
            data=df_trials_ch,
            palette='vlag',
            hue='num_reward_past',
            errorbar='se',
            dodge=False,
            legend=False,
            ax=panels[i*3 + i + 1]
        )
        panels[i*3 + i + 1].set_title("LEFT TRIALS" if i == 0 else "RIGHT TRIALS")

        # RPE with baseline removed
        pf.plot_fip_psth_compare_alignments(
                nwb_split, RPE_binned3_dfs_dicts, channel,
                extra_colors=dict(zip(RPE_binned3_label_names, sns.color_palette("mako", len(RPE_binned3_label_names)).as_hex())),
                tw=trial_width_choice, censor=True, data_column=data_col+'_norm', error_type=error_type, ax=panels[i * 3 + i + 2]
            )
        panels[i * 3 + i + 2].set_title("")

        panels[i * 3 + i + 2].fill_betweenx([0, 0.01], offsets[0], offsets[1], color = 'gray', alpha = 0.3 )

        # average signals
        avg_signal_cols = [c for c in df_trials_ch.columns if c.startswith("avg_data") and channel[:3] in c]

        if len(avg_signal_cols) != 1:
            print("incorrect number of avg_signal_col found, skipping RPE vs avg signal plot")
            print(avg_signal_cols)
            continue
        
        plot_RPE_by_avg_signal(df_trials_ch, avg_signal_cols[0], ax = panels[i * 3 + i + 3])
    return panels

def plot_all_sess_left_right_RPE_PSTH(df_sess, nwbs_all, channel, channel_loc, offsets, loc=None):
    """
    PSTH-focused version of plot_all_sess.
    Uses plot_row_panels_PSTH_legends for a top legend row and plot_row_panels_PSTH for the PSTH row,
    producing one two-row block per session in nwbs_all.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

    nrows = len(nwbs_all)
    ncols = 8
    subject_id = df_sess['subject_id'].unique()[0]

    # use constrained_layout to avoid tight_layout warnings with complex nested axes
    fig = plt.figure(figsize=(ncols * 5, nrows*4), constrained_layout=True)
    
    plt.suptitle(f"{subject_id} {channel_loc} ({channel})", fontsize=16)

    # allocate one extra top row for the shared legend panels
    outer = GridSpec(nrows + 1, 1, figure=fig)

    # df_trials_all for consistent legend labels/colors across rows
    df_trials_all = pd.concat([nwb.df_trials for nwb in nwbs_all])

    # keep reference to the PSTH axes for final labeling
    axes_rows = [None] * nrows

    # For each session, create a single-row grid of ncols for PSTH panels
    for row, nwb in enumerate(nwbs_all):
        # create a small title row above the 4 panels using a nested GridSpec
        inner = GridSpecFromSubplotSpec(2, ncols, subplot_spec=outer[row], height_ratios=[0.12, 0.88], hspace=0.0, wspace=0.3)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis('off')
        title_ax.set_title(f"{nwb}", fontsize=16, fontweight='bold')

        # create the n_cols panel axes for this row
        panels = [fig.add_subplot(inner[1, col]) for col in range(ncols)]

        # PSTH panels for this session (place in second row)
        plot_row_panels_left_right_RPE(nwb, channel, panels, offsets)

        axes_rows[row] = panels

    # set bottom row xlabels using the last row panels
    last_panels = axes_rows[-1]
    if last_panels is not None:
        last_panels[1].set_xlabel('num_reward_past')
        last_panels[5].set_xlabel('num_reward_past')
        last_panels[0].set_xlabel('Time (s) from choice')
        last_panels[2].set_xlabel('Time (s) from choice')
        last_panels[4].set_xlabel('Time (s) from choice')
        last_panels[6].set_xlabel('Time (s) from choice')
        last_panels[3].set_xlabel('RPE_earned')
        last_panels[7].set_xlabel('RPE_earned')



    if loc is not None:
        plt.savefig(f"{loc}all_sess_left_right_RPE_{subject_id}_{channel}_{channel_loc}.png", bbox_inches='tight', transparent=False)
        plt.close(fig)


def plot_weekly_grid(df_sess, nwbs_by_week, rpe_slope, channel, channel_loc, loc=None):

    week_intervals = sorted(df_sess['week_interval'].unique())
    subject_id = str(df_sess['subject_id'].unique()[0])

    nrows = 1 + len(week_intervals)
    ncols = N_COLS_PER_ROW


    fig = plt.figure(figsize=(ncols*5, nrows*4))
    plt.suptitle(f"{subject_id} {channel_loc} ({channel})", fontsize = 16)

    outer = GridSpec(nrows, 1, figure=fig)

    # axes_rows will hold lists of 4 axes for each row; index 0 reserved for the top summary row (unused)
    axes_rows = [None] * nrows
    # --- Rows 0: Slope information
    top_inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], hspace=0.0, wspace=0.3)
    top_panels = [fig.add_subplot(top_inner[0, col]) for col in range(2)]


    rpe_slope['date_num'] = rpe_slope['date'].map(pd.Timestamp.toordinal)
    rpe_slope['month_year'] = rpe_slope['date'].dt.to_period('M')
    month_starts = rpe_slope.groupby('month_year')['date_num'].min().values
    month_labels = rpe_slope.groupby('month_year')['date'].min().dt.strftime('%b-%Y').values

    sns.regplot(
        data=rpe_slope, 
        x='date_num', 
        y='slope (RPE >= 0)', 
        scatter=True, 
        ci=None, 
        line_kws={'color': 'red'},
        # fit_reg = False,
        scatter_kws={'color': 'k'},
        marker = '+',
        ax = top_panels[0]
    )
    sns.regplot(
        data=rpe_slope, 
        x='date_num', 
        y='slope (RPE < 0)', 
        scatter=True, 
        ci=None, 
        line_kws={'color': 'blue'},
        # fit_reg = False,
        scatter_kws={'color': 'k'},
        marker = '+',
        ax = top_panels[1]
    )


    top_panels[0].set_ylabel('Slope of positive RPE regression')
    top_panels[1].set_ylabel('Slope of negative RPE regression')
    
    top_panels[0].set_title(f'Positive RPE regression slope. Average slope = {rpe_slope['slope (RPE >= 0)'].mean():.4f}')
    top_panels[1].set_title(f'Negative RPE regression slope. Average slope = {rpe_slope['slope (RPE < 0)'].mean():.4f}')


    [panel.set_xlabel('Date') for panel in top_panels]
    [panel.tick_params(axis='x', labelrotation=-45) for panel in top_panels]
    [panel.set_xticks(month_starts) for panel in top_panels]
    [panel.set_xticklabels(month_labels) for panel in top_panels]


    # --- Rows 1+: Per week interval ---
    for week_i, nwbs in enumerate(nwbs_by_week):
        row = week_i + 1

        # create a small title row above the 4 panels using a nested GridSpec
        inner = GridSpecFromSubplotSpec(2, ncols, subplot_spec=outer[row], height_ratios=[0.12, 0.88], hspace=0.0, wspace=0.3)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis('off')
        nwb_dates = str(nwbs)[1:-1].replace(subject_id + '_', '')
        title_ax.set_title(f"Week {week_i} ({len(nwbs)} sessions): {nwb_dates}", fontsize=16, fontweight='bold')

        # create the n_cols panel axes for this row
        panels = [fig.add_subplot(inner[1, col]) for col in range(ncols)]

        panels = plot_row_panels_RPE(nwbs, channel, panels)
        axes_rows[row] = panels


    # set bottom row xlabels using the last row panels
    last_panels = axes_rows[-1]
    if last_panels is not None:
        last_panels[2].set_xlabel('num_reward_past')
        last_panels[0].set_xlabel('Time (s) from choice')
        last_panels[1].set_xlabel('Time (s) from choice')
        last_panels[3].set_xlabel('Time (s) from choice')

    # show legends on the first data row (row index 1) if it exists
    if nrows > 1 and axes_rows[1] is not None:
        for (col, legend_title) in zip([0, 1, 3], ['choice', 'RPE', 'RPE']):
            axes_rows[1][col].legend(framealpha=0.5, title = legend_title, fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if loc is not None:
        plt.savefig(f"{loc}weekly_{subject_id}_{channel}.png" )
        plt.close()


def plot_row_panels_PSTH_legends(df_trials_all, panels):
    """
    Draw legend panels for the five columns for the function plot_row_panels_PSTH
    df_trials_all: concatenated df_trials DataFrame (or None) used to build category labels.
    panels: list of 5 matplotlib axes to draw legends into.
    """
    # build label lists from df if available
    if df_trials_all is not None:
        RPE_labels = df_trials_all['RPE-binned3'].cat.categories.astype(str).tolist() if 'RPE-binned3' in df_trials_all else []
        Qch_labels = df_trials_all['Qch-binned3'].cat.categories.astype(str).tolist() if 'Qch-binned3' in df_trials_all else []
    else:
        RPE_labels = []
        Qch_labels = []

    # column 0: choice legend
    choice_labels = ['left', 'right', 'ignore']
    choice_colors = ['blue', 'red', 'purple']
    # proxy artists (empty) — we'll color the legend text via labelcolor
    handles = [Line2D([], [], linestyle='', marker='') for _ in choice_colors]
    panels[0].axis('off')
    panels[0].legend(handles, choice_labels, title='choice', ncol=1, loc='center', frameon=False, labelcolor=choice_colors)

    # column 1: reward legend (vertical colored text)
    rew_labels = ['rew', 'no-rew']
    rew_colors = ['magenta', 'grey']
    handles = [Line2D([], [], linestyle='', marker='') for _ in rew_colors]
    panels[1].axis('off')
    panels[1].legend(handles, rew_labels, title='reward', ncol=1, loc='center', frameon=False, labelcolor=rew_colors)

    # column 2: RPE bins legend (two columns)
    if RPE_labels:
        RPE_colors = sns.color_palette("mako", len(RPE_labels)).as_hex()
        handles = [Line2D([], [], linestyle='', marker='') for _ in RPE_colors]
        panels[2].axis('off')
        panels[2].legend(handles, RPE_labels, title='RPE', ncol=2, loc='center', frameon=False, labelcolor=RPE_colors)
    else:
        panels[2].text(0.5, 0.5, "RPE", ha='center', va='center')
    panels[2].axis('off')

    # column 3: Qch bins legend (two columns)
    if Qch_labels:
        Qch_colors = sns.color_palette("vlag", len(Qch_labels)).as_hex()
        handles = [Line2D([], [], linestyle='', marker='') for _ in Qch_colors]
        panels[3].axis('off')
        panels[3].legend(handles, Qch_labels, title='Qch', ncol=2, loc='center', frameon=False, labelcolor=Qch_colors)
    else:
        panels[3].text(0.5, 0.5, "Qch", ha='center', va='center')
    panels[3].axis('off')

    # column 2: RPE bins legend (two columns)
    if RPE_labels:
        RPE_colors = sns.color_palette("mako", len(RPE_labels)).as_hex()
        handles = [Line2D([], [], linestyle='', marker='') for _ in RPE_colors]
        panels[2].axis('off')
        panels[2].legend(handles, RPE_labels, title='RPE', ncol=2, loc='center', frameon=False, labelcolor=RPE_colors)
    else:
        panels[2].text(0.5, 0.5, "RPE", ha='center', va='center')
    panels[2].axis('off')

    # column 3: Qch bins legend (two columns)
    if Qch_labels:
        Qch_colors = sns.color_palette("vlag", len(Qch_labels)).as_hex()
        handles = [Line2D([], [], linestyle='', marker='') for _ in Qch_colors]
        panels[3].axis('off')
        panels[3].legend(handles, Qch_labels, title='Qch', ncol=2, loc='center', frameon=False, labelcolor=Qch_colors)
    else:
        panels[3].text(0.5, 0.5, "Qch", ha='center', va='center')
    panels[3].axis('off')

    # column 4: baseline legend (single colored label)
    handles = [Line2D([], [], linestyle='', marker='')]
    panels[4].axis('off')
    panels[4].legend([])

    return panels



def plot_row_panels_PSTH(nwbs, channel, panels, legend_panel = False):
    """
    Plot a row of summary panels for a given set of NWB sessions and a specific channel.

    This function generates a set of five panels summarizing neural and behavioral data:
        1. PSTH for left vs right choices.
        2. PSTH split by RPE-binned3 categories.
        3. Baseline z-scored df/f by number of past rewards.
        4. PSTH split by RPE-binned3 with baseline removed.
        5. Scatter and regression of RPE vs average signal.

    Args:
        nwbs (list): List of NWB session objects, each with a .df_trials DataFrame.
        channel (str): Channel name to plot.
        panels (list): List of matplotlib Axes objects (length 5) to plot into.

    Returns:
        list: The input list of matplotlib Axes, with plots drawn.
    """
    trial_width_choice = [-1, 4]
    df_trials_all = pd.concat([nwb.df_trials for nwb in nwbs])
    RPE_binned3_label_names = df_trials_all['RPE-binned3'].cat.categories.astype(str).tolist()
    Qch_binned3_label_names = df_trials_all['Qch-binned3'].cat.categories.astype(str).tolist()

    if legend_panel:
        return plot_row_panels_PSTH_legends(df_trials_all, panels)
    
    if len(nwbs) > 1:
        error_type = 'sem_over_sessions'
        data_col = 'data_z'
    else:
        error_type = 'sem'
        data_col = 'data'
    
    # 1. Choice L vs R
    pf.plot_fip_psth_compare_alignments(
            nwbs,
            [{"left": nwb.df_trials.query("choice == 0").choice_time_in_session.values,
              "right": nwb.df_trials.query("choice == 1").choice_time_in_session.values,
              "ignore":nwb.df_trials.query("choice == 2").choice_time_in_session.values} for nwb in nwbs],
            channel,
            tw=trial_width_choice,
            extra_colors={"left": 'b', "right": 'r', "ignore":"purple"},
            data_column=data_col,
            error_type=error_type,
            ax=panels[0],
        )

    # 2 Reward/No Reward 
    pf.plot_fip_psth_compare_alignments(
            nwbs,
            [{"rew": nwb.df_trials.query("earned_reward == 1").choice_time_in_session.values,
              "no-rew": nwb.df_trials.query("earned_reward == 0").choice_time_in_session.values} for nwb in nwbs],
            channel,
            tw=trial_width_choice,
            extra_colors={"rew": 'magenta', "no-rew": 'grey'},
            data_column=data_col,
            error_type=error_type,
            ax=panels[1],
        )
    

     
    # 3 RPE_binned3
    get_binned3_dfs = lambda df_trials, val, label_names: [
        df_trials[df_trials[val] == str(bin_labels)]['choice_time_in_session'].values for bin_labels in label_names
    ]
    RPE_binned3_dfs_dicts = [
        dict(zip(RPE_binned3_label_names, get_binned3_dfs(nwb.df_trials, "RPE-binned3", RPE_binned3_label_names))) for nwb in nwbs]
    pf.plot_fip_psth_compare_alignments(
            nwbs, RPE_binned3_dfs_dicts, channel,
            extra_colors=dict(zip(RPE_binned3_label_names, sns.color_palette("mako", len(RPE_binned3_label_names)).as_hex())),
            tw=trial_width_choice, censor=True, data_column=data_col, error_type=error_type, ax=panels[2]
        )
    
    # 4 Q_val binned3
    Qch_binned3_dfs_dicts = [
        dict(zip(Qch_binned3_label_names, get_binned3_dfs(nwb.df_trials, "Qch-binned3",Qch_binned3_label_names))) for nwb in nwbs]
    pf.plot_fip_psth_compare_alignments(
            nwbs, Qch_binned3_dfs_dicts, channel,
            extra_colors=dict(zip(Qch_binned3_label_names, sns.color_palette("vlag", len(Qch_binned3_label_names)).as_hex())),
            tw=trial_width_choice, censor=True, data_column=data_col, error_type=error_type, ax=panels[3]
        )



    # 5. Baseline by num_reward_past (grand mean/SE)
    df_trials_all = df_trials_all.query(
        'num_reward_past > -7 and num_reward_past < 7'
    ).sort_values('trial')
    if len(nwbs) > 1:
        grouped = (
                df_trials_all
                .groupby(['ses_idx', 'num_reward_past'])[f'{data_col}_{channel}_baseline']
                .mean()
                .reset_index()
            )
        agg = (
                grouped
                .groupby('num_reward_past')[f'{data_col}_{channel}_baseline']
                .agg(['mean', 'sem'])
                .reset_index()
            )
        panels[4].bar(
                agg['num_reward_past'],
                agg['mean'],
                yerr=agg['sem'],
                color=sns.color_palette('vlag', len(agg)),
                capsize=4,
        )
        panels[4].set_ylabel(f'{data_col}_baseline')
    else:
        sns.barplot(
                x='num_reward_past',
                y=f'{data_col}_{channel}_baseline',
                data=df_trials_all,
                palette='vlag',
                hue='num_reward_past',
                errorbar='se',
                dodge=False,
                legend=False,
                ax=panels[4]
            )
    panels[4].set_title('Baseline of z-scored df/f')

    
    for ax in panels:
        ax.set_title("")
        ax.set_xlabel("")

    for idx in np.arange(1,4):
        panels[idx].set_ylabel("")
    return panels

def plot_row_panels_PSTH_multiple_channels(nwbs, channels, panels, legend_panel = False):
    for channel in channels:
        plot_row_panels_PSTH(nwbs, channel, panels, legend_panel)

def plot_all_sess_PSTH(df_sess, nwbs_all, channel, channel_loc, loc=None):
    """
    PSTH-focused version of plot_all_sess.
    Uses plot_row_panels_PSTH_legends for a top legend row and plot_row_panels_PSTH for the PSTH row,
    producing one two-row block per session in nwbs_all.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

    nrows = len(nwbs_all)
    ncols = N_COLS_PER_ROW
    subject_id = df_sess['subject_id'].unique()[0]

    # use constrained_layout to avoid tight_layout warnings with complex nested axes
    fig = plt.figure(figsize=(ncols * 5, max(4, (nrows) * 4 + 1)), constrained_layout=True)
    
    plt.suptitle(f"{subject_id} {channel_loc} ({channel})", fontsize=16)

    # allocate one extra top row for the shared legend panels
    outer = GridSpec(nrows + 1, 1, figure=fig)

    # df_trials_all for consistent legend labels/colors across rows
    df_trials_all = pd.concat([nwb.df_trials for nwb in nwbs_all])

    # Top: single legend row spanning the top outer[0]
    legend_inner = GridSpecFromSubplotSpec(1, ncols, subplot_spec=outer[0], height_ratios=[0.15], hspace=0.0, wspace=0.3)
    legend_axes = [fig.add_subplot(legend_inner[0, col]) for col in range(ncols)]
    plot_row_panels_PSTH_legends(df_trials_all, legend_axes)

    # keep reference to the PSTH axes for final labeling
    axes_rows = [None] * nrows

    # For each session, create a single-row grid of ncols for PSTH panels
    for row, nwb in enumerate(nwbs_all):
        inner = GridSpecFromSubplotSpec(
            2,
            ncols,
            subplot_spec=outer[row + 1],
            height_ratios=[0.12, 0.88],
            hspace=0.0,
            wspace=0.3,
        )
        # title row (single axis spanning all columns)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis('off')
        title_ax.set_title(f"{nwb}", fontsize=16, fontweight='bold')

        # PSTH panels for this session (place in second row)
        psth_axes = [fig.add_subplot(inner[1, col]) for col in range(ncols)]
        plot_row_panels_PSTH([nwb], channel, psth_axes, legend_panel=False)

        axes_rows[row] = psth_axes
    
    # set bottom row xlabels using the last row panels
    for panel_legend in [axes_rows[-1],axes_rows[0]]:
        if panel_legend is not None:
            panel_legend[-1].set_xlabel('Consecutive No-Reward and Reward Trials')
            panel_legend[0].set_xlabel('Time (s) from choice')
            panel_legend[1].set_xlabel('Time (s) from choice')
            panel_legend[2].set_xlabel('Time (s) from choice')
            panel_legend[3].set_xlabel('Time (s) from choice')
        if len(axes_rows) < 5:
            break

    if loc is not None:
        plt.savefig(f"{loc}all_sess_PSTH_{subject_id}_{channel}_{channel_loc}.png", bbox_inches='tight', transparent=False, dpi=300)
        plt.close(fig)



def plot_avg_final_N_sess(df_sess, nwbs_by_week, channel_dict, final_N_sess = 5, loc = None):
    # set pdf plot requirements
    mpl.rcParams['pdf.fonttype'] = 42 # allow text of pdf to be edited in illustrator

    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False


    if len(df_sess['subject_id'].unique()) > 1:
        raise ValueError("df_sess contains multiple subjects, please provide a df_sess for a single subject.")
    
    subject_id = df_sess['subject_id'].unique()[0]
    

    channels = list(channel_dict.keys())
    ncols = N_COLS_PER_ROW
    nrows = len(channels)
    nwbs_all = [nwb for nwb_week in nwbs_by_week for nwb in nwb_week]
    nwbs = nwbs_all[-final_N_sess:]
    
    
    nwb_dates = str(nwbs)[1:-1].replace(subject_id + '_', '')

    fig = plt.figure(figsize=(ncols*5, nrows*4))
    plt.suptitle(f"{subject_id} Final {final_N_sess} Sessions Summary Figs" + 
                    f"\n ({nwb_dates})", fontsize = 16)

    outer = GridSpec(nrows, 1, figure=fig)

    # axes_rows will hold lists of 4 axes for each row; index 0 reserved for the top summary row (unused)
    axes_rows = [None] * nrows
    
    # --- Rows 1+: Per week interval ---
    for row, channel in enumerate(channels):

        # create a small title row above the 4 panels using a nested GridSpec
        inner = GridSpecFromSubplotSpec(2, ncols, subplot_spec=outer[row], height_ratios=[0.12, 0.88], hspace=0.0, wspace=0.3)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis('off')
        title_ax.set_title(f"Channel {channel} @ {channel_dict[channel]}", fontsize=16, fontweight='bold')

        # create the n_cols panel axes for this row
        panels = [fig.add_subplot(inner[1, col]) for col in range(ncols)]

        panels = plot_row_panels_RPE(nwbs, channel, panels)
        axes_rows[row] = panels


    # set bottom row xlabels using the last row panels
    last_panels = axes_rows[-1]
    if last_panels is not None:
        last_panels[2].set_xlabel('num_reward_past')
        last_panels[0].set_xlabel('Time (s) from choice')
        last_panels[1].set_xlabel('Time (s) from choice')
        last_panels[-1].set_xlabel('Time (s) from choice')

    # show legends on the first data row (row index 1) if it exists
    for (col, legend_title) in zip([0, 1, 3], ['choice', 'RPE', 'RPE']):
        axes_rows[0][col].legend(framealpha=0.5, title = legend_title, fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if loc is not None:
        plt.savefig(f"{loc}avg_signal_{subject_id}_{channel}.png",bbox_inches='tight',transparent = False, dpi = 1000)
        plt.close()
def set_bar_percentages(ax, df, x_col, hue_col):
    """
    Set bar heights to percent within each hue for a seaborn countplot.
    Optionally add hatching to bars for a specific hue value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the countplot.
    df : pd.DataFrame
        DataFrame used for the plot.
    x_col : str
        Column for x-axis categories.
    hue_col : str
        Column for hue categories.
    """
    hues = np.sort(df[hue_col].unique())
    x_vals = np.sort(df[x_col].unique())
    n_x = len(x_vals)
    n_hue = len(hues)
    hatch_kwargs = dict(edgecolor='black', alpha=0.8)

    bar_idx = 0
    for hue_idx, hue in enumerate(hues):
        for x_idx, x_val in enumerate(x_vals):
            df_x = df[df[x_col] == x_val]
            total = len(df_x)            # per-x normalization
            count = (df_x[hue_col] == hue).sum()
            percent = 100 * count / total if total > 0 else 0
            if count > 0:
                ax.patches[bar_idx].set_height(percent)
                bar_idx = bar_idx + 1

def plot_per_sess_behavior_data(nwb, fig, panels):

    title_ax = fig.add_subplot(panels[0, :])
    title_ax.axis('off')
    title_ax.set_title(f"{nwb}", fontsize=14, fontweight='bold')
    # --- First row --- 
    # Top row (full width)

    # Top: foraging session plot
    big_ax_top = fig.add_subplot(panels[1,:])

    [_, foraging_session_axes] = pb.plot_foraging_session_nwb(nwb, ax=big_ax_top)
    foraging_session_axes[1].set_xlabel("")


    # Bottom slice:
    big_ax_bottom = fig.add_subplot(panels[2,:],sharex = foraging_session_axes[1])
    big_ax_bottom.plot(nwb.df_trials['Q_sum'].values, label = 'Q_sum', color = 'green')
    big_ax_bottom.plot(nwb.df_trials['Q_chosen'].values, label = 'Q_chosen', color = 'magenta')
    big_ax_bottom.legend(fontsize = 'x-small', title = "", bbox_to_anchor=(0.5, 1.05), 
                # x=0.5 center, y>1 places legend above the axis
                ncol=2,
                frameon=False)
    big_ax_bottom.spines['top'].set_visible(False)
    big_ax_bottom.spines['right'].set_visible(False)
    big_ax_bottom.set_xlabel("Trial Number")


    # ax1: Response time histogram (left 4 columns)
    ax1 = fig.add_subplot(panels[-1, :3])
    ax1.set_title("Response Time Histogram")

    stay_leave_palette = {'False': 'Lime', 'True': 'Green', 
                          0: 'Lime', 1: 'Green',
                          'leave':'Lime', 'stay':'Green'}
    nwb.df_trials['stay_label'] = nwb.df_trials['stay'].map({False: 'leave', True: 'stay'})

    sns.histplot(
        data=nwb.df_trials,
        x='response_time',
        hue='stay_label',
        palette=stay_leave_palette,
        bins=100,
        alpha=0.8,
        stat='count',
        legend=True,
        ax=ax1
    )
    ax1.set_xlabel('Response Time (s)')
    ax1.set_ylabel('Count')
    ax1.set_xlim(0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize = "small", frameon = False)


    # ax2: % Stay/Leave (very narrow)
    ax2 = fig.add_subplot(panels[-1,3])
    df_trials_prev_rew = pd.concat([nwb.df_trials, nwb.df_trials.query('rewarded_prev == True')],
                             keys = ['all trials', 'prev rew']).reset_index(level=[0])
    ax2.set_title("% Stay/Leave")
    # Plot overall % Stay/Leave
    stay_bars = sns.countplot(
        x='level_0',
        data=df_trials_prev_rew,
        order = ['all trials', 'prev rew'],
        hue_order = ['leave', 'stay'],
        hue = 'stay_label',
        palette = {'stay':'Green', 'leave':'Lime'},
        stat='percent',
        ax=ax2,
        width=0.5,
        legend=False
    )
    set_bar_percentages(stay_bars, df_trials_prev_rew, x_col='level_0', hue_col='stay_label')
    

    
    ax2.set_ylim([0, 100])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis='x', labelrotation=45)


    # ax3: % Stay/Leave | Reward (very narrow)
    ax3 = fig.add_subplot(panels[-1, 4])
    df_trials_rew = pd.concat([nwb.df_trials, nwb.df_trials.query('reward_all == True')],
                             keys = ['all trials', 'rew trials']).reset_index(level=[0])
    ax3.set_title("% Choice")

    # Map choice to labels
    choice_labels = {0: 'left', 1: 'right', 2: 'ignore'}
    df_trials_rew['choice_label'] = df_trials_rew['choice'].map(choice_labels)
    # Create temporary df with rewarded_prev trials duplicated to plot both together

    choice_bars = sns.countplot(
        x='level_0',
        data=df_trials_rew,
        order = ['all trials', 'rew trials'],
        hue_order = ['ignore','left', 'right'],
        hue = 'choice_label',
        palette = {'right':(0.945, 0.580, 0.573), 'left':(0.522, 0.522, 0.969,1), 'ignore':'grey'},
        stat='percent',
        ax=ax3,
        width=0.5,
        legend = False
    )
    set_bar_percentages(choice_bars, df_trials_rew, x_col='level_0', hue_col='choice_label')
    

    ax3.set_ylim([0, 100])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.tick_params(axis='x', labelrotation=45)
    
    plt.tight_layout()
   

def plot_all_sess_RPE(df_sess, nwbs_all, channel, channel_loc, loc=None):
    # set pdf plot requirements
    mpl.rcParams['pdf.fonttype'] = 42 # allow text of pdf to be edited in illustrator
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False


    nrows = len(nwbs_all)
    ncols = N_COLS_PER_ROW
    subject_id = df_sess['subject_id'].unique()[0]



    fig = plt.figure(figsize=(ncols*5, nrows*4))
    plt.suptitle(f"{subject_id} {channel_loc} ({channel})", fontsize = 16)

    outer = GridSpec(nrows, 1, figure=fig)

    # axes_rows will hold lists of 4 axes for each row; index 0 reserved for the top summary row (unused)
    axes_rows = [None] * nrows

    # -- Plot one row per session --- 
    for row, nwb in enumerate(nwbs_all):

        # create a small title row above the 4 panels using a nested GridSpec
        inner = GridSpecFromSubplotSpec(2, ncols, subplot_spec=outer[row], height_ratios=[0.12, 0.88], hspace=0.0, wspace=0.3)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis('off')
        title_ax.set_title(f"{nwb}", fontsize=16, fontweight='bold')

        # create the n_cols panel axes for this row
        panels = [fig.add_subplot(inner[1, col]) for col in range(ncols)]

        panels = plot_row_panels_RPE([nwb], channel, panels)
        axes_rows[row] = panels


    # set bottom row xlabels using the last row panels
    last_panels = axes_rows[-1]
    if last_panels is not None:
        last_panels[2].set_xlabel('num_reward_past')
        last_panels[0].set_xlabel('Time (s) from choice')
        last_panels[1].set_xlabel('Time (s) from choice')
        last_panels[-1].set_xlabel('Time (s) from choice')

    # show legends on the first data row (row index 1) if it exists
    if nrows > 1 and axes_rows[1] is not None:
        for (col, legend_title) in zip([0, 1, 3], ['choice', 'RPE', 'RPE']):
            axes_rows[1][col].legend(framealpha=0.5, title = legend_title, fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if loc is not None:
        plt.savefig(f"{loc}all_sess_RPE_{subject_id}_{channel}_{channel_loc}.pdf",bbox_inches='tight',transparent = True, dpi = 1000)
        plt.close()
def plot_all_sess_behavior(df_sess, nwbs_all,loc=None):
    """
    plot_all_sess the DA version-- 
    plots L/R, split by RPE, baseline, split by RPE after baseline removal, slope for average response. 
    """
    # set sizese
    mpl.rcParams.update({
    "font.size": FONTSIZE - 2,
    "legend.fontsize": FONTSIZE-2,
    "axes.titlesize": FONTSIZE - 2,
    "axes.labelsize": FONTSIZE - 2,
    "xtick.labelsize": FONTSIZE - 2,
    "ytick.labelsize": FONTSIZE - 2,
    "figure.titlesize": FONTSIZE -2
})

    # set pdf plot requirements
    mpl.rcParams['pdf.fonttype'] = 42 # allow text of pdf to be edited in illustrator
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False


    nrows = len(nwbs_all)
    subject_id = df_sess['subject_id'].unique()[0]



    fig = plt.figure(figsize=(8, nrows*6), constrained_layout = True)

    outer = GridSpec(nrows, 1, figure=fig)

    # axes_rows will hold lists of 4 axes for each row; index 0 reserved for the top summary row (unused)
    axes_rows = [None] * nrows

    # -- Plot one row per session --- 
    for row, nwb in enumerate(nwbs_all):

        # create a small title row above the 4 panels using a nested GridSpec
        inner = GridSpecFromSubplotSpec(4, 5, subplot_spec=outer[row],height_ratios=[0.1, 3, 1, 1], wspace = 0.2, hspace = 0.6)
        panels = plot_per_sess_behavior_data(nwb, fig, inner)
        axes_rows[row] = panels

    fig.canvas.draw()
    fig.subplots_adjust(top=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=1.08)


    if loc is not None:
        plt.savefig(f'{loc}{nwbs_all[0].session_id.replace("behavior_","")}_behavior.png'
                            ,bbox_inches='tight',transparent = False, dpi = 1000)
        plt.close()