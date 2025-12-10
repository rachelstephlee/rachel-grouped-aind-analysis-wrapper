import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from aind_dynamic_foraging_basic_analysis.plot import plot_fip as pf
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
from scipy import stats


output_col_name = lambda channel, data_column, alignment_event: f"avg_{data_column}_{channel[:3]}_{alignment_event.split("_in_")[0]}"


N_COLS_PER_ROW = 5

def get_RPE_by_avg_signal_fit(data, avg_signal_col):


    x = data['RPE_earned'].values
    y = data[avg_signal_col].values
    try:
        lr = stats.linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = lr.intercept + lr.slope * x_fit
        slope = lr.slope
    except ValueError as e:
        print(f"Error in linear regression: {e}")
        x_fit = np.nan * np.arange(100)
        y_fit = np.nan * np.arange(100)
        slope = np.nan
    return (x_fit, y_fit, slope)

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
        (x_fit, y_fit, slope) = get_RPE_by_avg_signal_fit(df_trials_neg, avg_signal_col)
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
        (x_fit, y_fit, slope) = get_RPE_by_avg_signal_fit(df_trials_pos, avg_signal_col)
        ax.plot(x_fit, y_fit, color='red', lw=2, label=f'RPE >= 0: {slope:.3f}')
    ax.set_xlabel('RPE_earned')
    ax.set_ylabel(avg_signal_col)
    ax.legend(framealpha=0.5, title = "LM fitted slopes", fontsize='small')

    return ax



def plot_row_panels(nwbs, channel, panels):
    
    trial_width_choice = [-1, 4]
    df_trials_all = pd.concat([nwb.df_trials for nwb in nwbs])
    RPE_binned3_label_names = df_trials_all['RPE-binned3'].cat.categories.astype(str).tolist()
    if len(nwbs) > 1:
        error_type = 'sem_over_sessions'
    else:
        error_type = 'sem'
    
    # 1. Choice L vs R
    pf.plot_fip_psth_compare_alignments(
            nwbs,
            [{"left": nwb.df_trials.query("choice == 0").choice_time_in_session.values,
              "right": nwb.df_trials.query("choice == 1").choice_time_in_session.values} for nwb in nwbs],
            channel,
            tw=trial_width_choice,
            extra_colors={"left": 'b', "right": 'r'},
            data_column="data_z",
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
            tw=trial_width_choice, censor=True, data_column="data_z", error_type=error_type, ax=panels[1]
        )

    # 3. Baseline by num_reward_past (grand mean/SE)
    df_trials_all = df_trials_all.query(
        'num_reward_past > -7 and num_reward_past < 7'
    ).sort_values('trial')
    if len(nwbs) > 1:
        grouped = (
                df_trials_all
                .groupby(['ses_idx', 'num_reward_past'])[f'data_z_{channel}_baseline']
                .mean()
                .reset_index()
            )
        agg = (
                grouped
                .groupby('num_reward_past')[f'data_z_{channel}_baseline']
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
        panels[2].set_ylabel(f'data_z_{channel}_baseline')
    else:
        sns.barplot(
                x='num_reward_past',
                y=f'data_z_{channel}_baseline',
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
            tw=trial_width_choice, censor=True, data_column="data_z_norm", error_type=error_type, ax=panels[3]
        )
    panels[3].set_ylabel('z-scored df/f \n (baseline removed)')

    # 5. Add the RPE vs avg signal 
    df_trials_all = pd.concat([nwb.df_trials for nwb in nwbs])
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

        panels = plot_row_panels(nwbs, channel, panels)
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


def plot_all_sess(df_sess, nwbs_all, channel, channel_loc, loc=None):
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

        panels = plot_row_panels([nwb], channel, panels)
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
        plt.savefig(f"{loc}all_sess_{subject_id}_{channel}.pdf",bbox_inches='tight',transparent = True, dpi = 1000)
        plt.close()

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

        panels = plot_row_panels(nwbs, channel, panels)
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
        plt.savefig(f"{loc}avg_signal_{subject_id}_{channel}.pdf",bbox_inches='tight',transparent = True, dpi = 1000)
        plt.close()

def plot_per_channel_behavior_data(nwb, loc=None):
    fig = plt.figure(figsize=(12, 6))  # Removed 3rd row, so reduce height
    gs = gridspec.GridSpec(3,7, figure=fig, height_ratios=[1,0.75, 1])

    if isinstance(nwb,list):
        session_id_title = ', '.join([nwb_i.session_id for nwb_i in nwb])
    else:
        session_id_title = nwb.session_id
    plt.suptitle(f'{session_id_title}')

    # --- First row ---
    # Top row (full width)

    big_ax_top = fig.add_subplot(gs[0, :])
    big_ax_bottom = fig.add_subplot(gs[1, :], sharex=big_ax_top)

    pss.plot_session_scroller(nwb, plot_list=[], ax = [big_ax_top, big_ax_bottom], fig = fig, 
                              metrics = [["Q_left","Q_right", "Q_Delta",(-1, 1.2)]])

    big_ax_top.set_xlim(-200, 1400)
    big_ax_bottom.set_xlim(-200,1400)
    big_ax_top.set_title(big_ax_bottom.get_title())
    big_ax_bottom.set_xlabel(big_ax_top.get_xlabel())
    big_ax_bottom.set_title("")
    big_ax_top.set_xlabel("")
    big_ax_bottom.set_ylabel("Q value")
    big_ax_bottom.legend()


    # ax1: Response time histogram (left 4 columns)
    ax1 = fig.add_subplot(gs[2, :3])  
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
        ax=ax1
    )
    ax1.set_xlabel('Response Time (s)')
    ax1.set_ylabel('Count')
    ax1.set_xlim(0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)


    # ax2: % Stay/Leave (very narrow)
    ax2 = fig.add_subplot(gs[2, 3])
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
    

    ax2.set_ylabel('Percent')
    ax2.set_ylim([0, 100])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel("")
    ax2.tick_params(axis='x', labelrotation=45)


    # ax3: % Stay/Leave | Reward (very narrow)
    ax3 = fig.add_subplot(gs[2, 4])
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
    ax3.tick_params(axis='x', labelrotation=45)


    # ax5: Coefficient plot (right quarter)
    ax5 = fig.add_subplot(gs[2, 5:])
    ax5.set_title("Last Lick\n Response Time") 
    # Create histogram with hue for reward
    sns.histplot(
        data=nwb.last_licks,
        x='response_time',
        hue='reward_all',
        palette={0: 'grey', 1: 'red'},
        bins=100,
        alpha=0.8,
        stat='count',
        ax = ax5
    )

    ax5.set_xlabel('response time (s)')
    ax5.set_ylabel('count')
    ax5.set_xlim(0)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    plt.tight_layout()
    if loc is not None:
        plt.savefig(f'{loc}{nwb.session_id.replace("behavior_","")}_behavior.png')
        plt.close()
        try:
            nwb.df_logistic = get_logistic_regression(
            df_sessions=pd.DataFrame(
                {
                    "subject_id": [nwb.session_id.replace('behavior_','').split('_')[0]],
                    "session_date": [nwb.session_id.replace('behavior_','').split('_')[1]],
                }
            ),
            model="Su2022",
            if_download_figures=True,  # Also download fitting plots
            download_path=loc,
            )

            if nwb.nwb_file_loc is not None:
                with NWBZarrIO(path=nwb.nwb_file_loc, mode="r") as io:
                    nwb_actual = io.read()
                lick_analysis.plot_lick_analysis(nwb_actual)
            else: 
                print("no lick analysis plots generated because nwb is not found in data")


            plt.savefig(f'{loc}{nwb.session_id.replace("behavior_","")}_lick_analysis.png')
            plt.close()
        except Exception as e:
            print(f"Error in logistic regression or lick analysis plotting: {e}")