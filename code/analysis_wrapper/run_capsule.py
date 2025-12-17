import json
import logging
import os
from pathlib import Path
from hdmf_zarr import NWBZarrIO

import warnings


import sys
script_dir = "/root/capsule/code/analysis_wrapper"
if script_dir in sys.path:
    sys.path.remove(script_dir)
# sys.path.insert(0,"/root/capsule/")

sys.path.insert(0, str(Path(__file__).resolve().parent))


from analysis_pipeline_utils.metadata import construct_processing_record, docdb_record_exists, write_results_and_metadata
from analysis_pipeline_utils.analysis_dispatch_model import AnalysisDispatchModel
import analysis_wrapper.utils as utils
from analysis_wrapper.analysis_model import (
    SummaryPlotsAnalysisSpecification, SummaryPlotsAnalysisSpecificationCLI
)
import subprocess
import analysis_wrapper.rachel_analysis_framework_utils as r_utils

# for analysis code
import numpy as np
import pandas as pd 
from analysis_wrapper.plots import summary_plots
import analysis_util
from aind_dynamic_foraging_basic_analysis.metrics import trial_metrics


DATA_PATH: Path = Path("/data")  # TODO: don't hardcode
ANALYSIS_BUCKET = os.getenv("ANALYSIS_BUCKET")
logger = logging.getLogger(__name__)


      

def run_analysis(
    analysis_dispatch_inputs: AnalysisDispatchModel,
    **parameters,
) -> None:
    processing = construct_processing_record(analysis_dispatch_inputs,**parameters)
    
    dry_run = parameters["dry_run"]

    if docdb_record_exists(processing):
        logger.info("Record already exists, skipping.")
        return

    (df_sess, df_trials, df_events, df_fip) = r_utils.get_nwb_processed(analysis_dispatch_inputs.file_location, **parameters)



    ############## prepare computations for plotting ###############

    ##### PART I: REWARD #######
    df_trials['reward_all'] = df_trials['earned_reward'] + df_trials['extra_reward']
    # Compute num_reward_past and num_no_reward_past
    df_trials['rewarded_prev'] = df_trials.groupby('ses_idx')['reward_all'].shift(1)  # Shift to look at past values

    df_trials['num_reward_past'] = df_trials.groupby(
                            (df_trials['rewarded_prev'] != df_trials['reward_all']).cumsum()).cumcount() + 1

    # Set 'NA' for mismatched reward types
    df_trials.loc[df_trials['reward_all'] == 0, 'num_reward_past'] = df_trials.loc[df_trials['reward_all'] == 0, 'num_reward_past']* -1 

    ##### PART II: BINNING RPE #######
    # get RPE binned columns. 
    RPE_binned3_label_names = [str(np.round(i,2)) for i in np.arange(-1,0.99,1/3)]

    bins = np.arange(-1,1.01,1/3)
    bins[-1] = 1.001

    df_trials['RPE-binned3'] = pd.cut(df_trials['RPE_earned'],# all versus earned not a huge difference
                        bins = bins, right = True, labels=RPE_binned3_label_names)

    ##### PART III: BINNING QCHOSEN #######
    bins = [0.0, 1/3, 2/3, 1.01]
    q_labels = ["Qch 0", "Qch 0.33", "Qch 0.66"]

    q_bin = pd.cut(df_trials['Q_chosen'], bins=bins, labels=q_labels, include_lowest=True, right=True)
    reward_label = df_trials['earned_reward'].map({True: "R+", False: "R-"})

    # build combined label series (None where q_bin is NA)
    reward_Qcat_series = pd.Series(
        np.where(q_bin.isna(), None, reward_label.astype(str) + " (" + q_bin.astype(str) + ")"),
        index=df_trials.index
    )

    # ordered categories you requested
    Qch_binned3_label_names = [
        "R- (Qch 0)", "R- (Qch 0.33)", "R- (Qch 0.66)",
        "R+ (Qch 0)", "R+ (Qch 0.33)", "R+ (Qch 0.66)"
    ]

    # assign final ordered categorical to dataframe (no intermediate column left behind)
    df_trials['Qch-binned3'] = pd.Categorical(reward_Qcat_series, categories=Qch_binned3_label_names, ordered=True)

    ##### PART IV: GETTING STAY/LEAVE #######
    _choice_shifted = df_trials.groupby('ses_idx')['choice'].shift(1)
    df_trials['stay'] = df_trials['choice'] == _choice_shifted
    df_trials['switch'] = df_trials['choice'] != _choice_shifted
    df_trials['response_time'] = df_trials['choice_time_in_trial'] -  df_trials['goCue_start_time_in_trial']

    ############## finished computations for plotting ###############

    (df_sess, nwbs_by_week) = analysis_util.get_dummy_nwbs_by_week(df_sess, df_trials, df_events, df_fip) 


    # TODO: will need to refactor code so there's flexibility on the plots that come out
    #       consult alex? or figure it out on my own. 
    # get average activity 
    # data_column = 'data_z_norm'
    # alignment_event='choice_time_in_session'
    # rpe_slope_dict = {}
    # for channel in list(analysis_specification["channels"].keys()):
    #     avg_signal_col = summary_plots.output_col_name(channel, data_column, alignment_event)
    #     for nwb_week in nwbs_by_week:
        
    #         nwb_week = trial_metrics.get_average_signal_window_multi(
    #                         nwb_week,
    #                         alignment_event='choice_time_in_session',
    #                         offsets=[0.33, 1],
    #                         channel=channel,
    #                         data_column=data_column,
    #                         output_col = avg_signal_col
    #                     )
        
    #     # get rpe slope per session 

    #     df_trials_all = pd.concat([nwb.df_trials for nwb_week in nwbs_by_week for nwb in nwb_week])
    #     rpe_slope = []
    #     for ses_idx in sorted(df_trials_all['ses_idx'].unique()):
            
    #         data = df_trials_all[df_trials_all['ses_idx'] == ses_idx]
    #         data = data.dropna(subset = [avg_signal_col, 'RPE_earned'])
    #         if len(data) == 0:
    #             continue
    #         data_neg = data[data['RPE_earned'] < 0]
    #         data_pos = data[data['RPE_earned'] >= 0]

    #         ses_date = pd.to_datetime(ses_idx.split('_')[1])
    #         (_,_, slope_pos) = summary_plots.get_RPE_by_avg_signal_fit(data_pos, avg_signal_col)
    #         (_,_, slope_neg) = summary_plots.get_RPE_by_avg_signal_fit(data_neg, avg_signal_col)
    #         rpe_slope.append([ses_date, slope_pos, slope_neg])
    #     rpe_slope = pd.DataFrame(rpe_slope, columns=['date', 'slope (RPE >= 0)', 'slope (RPE < 0)'])
    #     rpe_slope_dict[channel] = rpe_slope

    # subject_id = df_sess['subject_id'].unique()[0]
    # # Concatenate with keys, turning dict keys into an index
    # combined_rpe_slope = pd.concat(rpe_slope_dict, names=["channel"])
    # combined_rpe_slope = combined_rpe_slope.reset_index(level="channel").reset_index(drop=True)

    # combined_rpe_slope.to_csv(f"/results/{subject_id}_rpe_slope.csv")


    # plot summary plots
    if dry_run:
        plot_loc = '/root/capsule/scratch/plots_TEST/'
    else:
        plot_loc = '/results/plots/'

    if not os.path.exists(plot_loc):
        os.makedirs(plot_loc)

    if "avg_lastN_sess" in parameters["plot_types"]:
        summary_plots.plot_avg_final_N_sess(df_sess, nwbs_by_week, parameters["channels"], final_N_sess = 5, loc = plot_loc)
    
    nwbs_all = [nwb for nwb_week in nwbs_by_week for nwb in nwb_week]
    for channel, channel_loc in parameters['channels'].items():
        if "all_sess" in parameters["plot_types"]:
            summary_plots.plot_all_sess_PSTH(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)
            summary_plots.plot_all_sess(df_sess, nwbs_all, loc = plot_loc)
        if "weekly" in parameters["plot_types"]:
            summary_plots.plot_weekly_grid(df_sess, nwbs_by_week, rpe_slope_dict[channel], channel, channel_loc, loc=plot_loc)
        
    # # # DRY RUN (comment in or out)
    if not dry_run:
        logger.info("Running analysis and posting results")
        write_results_and_metadata(processing, ANALYSIS_BUCKET)
        logger.info("Successfully wrote record to docdb and s3")
    else:
        logger.info("Dry run complete. Results not posted")


# Most of the below code will not need to change per-analysis
# and will be moved to a shared library
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_model_paths = tuple(DATA_PATH.glob('job_dict/*'))
    logger.info(f"Found {len(input_model_paths)} input job models to run analysis on.")
    analysis_specs = None

    analysis_spec_path = tuple(DATA_PATH.glob("analysis_parameters.json"))
    if analysis_spec_path:
        with open(analysis_spec_path[0], "r") as f:
            analysis_specs = json.load(f)

        logger.info(
            "Found analysis specification json. Parsing list of analysis specifications"
        )
    else:
        logger.info(
            "No analysis parameters json found. Defaulting to parameters passed in via input arguments"
        )

    ### WAY TO PARSE FROM USER DEFINED APP PANEL
    if analysis_specs is None:
        analysis_specs = SummaryPlotsAnalysisSpecificationCLI().model_dump_json()

    logger.info(f"Analysis Specification: {analysis_specs}")

    for model_path in input_model_paths:
        with open(model_path, "r") as f:
            analysis_dispatch_inputs = AnalysisDispatchModel.model_validate(json.load(f))
        
        analysis_specification = SummaryPlotsAnalysisSpecification.model_validate(analysis_specs).model_dump()


        run_analysis(analysis_dispatch_inputs = analysis_dispatch_inputs, **analysis_specification)
        
