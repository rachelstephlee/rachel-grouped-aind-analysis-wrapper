import json
import logging
import os
from pathlib import Path
from hdmf_zarr import NWBZarrIO

import warnings
from aind_dynamic_foraging_data_utils import code_ocean_utils as co_utils
from aind_dynamic_foraging_data_utils import nwb_utils, alignment, enrich_dfs
from aind_dynamic_foraging_basic_analysis.metrics import trial_metrics

from aind_analysis_arch_result_access.han_pipeline import get_session_table

import sys
script_dir = "/root/capsule/code/analysis_wrapper"
if script_dir in sys.path:
    sys.path.remove(script_dir)
# sys.path.insert(0,"/root/capsule/")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_util

from analysis_pipeline_utils.metadata import construct_processing_record, docdb_record_exists, write_results_and_metadata
from analysis_pipeline_utils.analysis_dispatch_model import AnalysisDispatchModel
import analysis_wrapper.utils as utils
from analysis_wrapper.analysis_model import (
    SummaryPlotsAnalysisSpecification, SummaryPlotsAnalysisSpecificationCLI
)
import subprocess

# for analysis code
import numpy as np
import pandas as pd 
from analysis_wrapper.plots import summary_plots


DATA_PATH: Path = Path("/data")  # TODO: don't hardcode
ANALYSIS_BUCKET = os.getenv("ANALYSIS_BUCKET")
logger = logging.getLogger(__name__)


def get_nwb_processed(file_locations, **parameters) -> None:
    interested_channels = list(parameters["channels"].keys())
    df_sess = nwb_utils.create_df_session(file_locations)
    df_sess['s3_location'] = file_locations

    # check for multiple sessions on the same day
    dup_mask = df_sess.duplicated(subset=['ses_idx'], keep=False)
    if dup_mask.any():
        warnings.warn(f"Duplicate sessions found for ses_idx: {df_sess[dup_mask]['ses_idx'].tolist()}."
                        "Keeping the one with more finished trials.")
        df_sess = (df_sess.sort_values(by=['ses_idx','finished_trials'], ascending=False)
                         .drop_duplicates(subset=['ses_idx'], keep='first')
                  )
    # sort sessions
    df_sess = (df_sess.sort_values(by=['session_date']) 
                         .reset_index(drop=True)
              )
    # only read last N sessions unless daily, weekly plots are requested
    if "weekly" and "all_sess" not in parameters["plot_types"] and parameters["plot_types"] != "":
        df_sess = df_sess.tail(parameters["last_N_sess"])
    
    (df_trials, df_events, df_fip) = co_utils.get_all_df_for_nwb(filename_sessions=df_sess['s3_location'].values, interested_channels = interested_channels)

    if parameters["pipeline_v14"]: # TODO HACKY fix, take out once we fixed johannes' PR 
        df_fip = df_fip.rename(columns={'timestamps':'timestamps_WRONG', 'raw_timestamps': 'timestamps'})
        # print("pipeline_v14 has the WRONG timestamp column timing for df_fip"
        #     " set to the first df_fip timestamp = 0 rather than first goCue = 0", file=sys.stderr)
        
    df_trials_fm, df_sess_fm = co_utils.get_foraging_model_info(df_trials, df_sess, loc = None, model_name = parameters["fitted_model"])
    df_trials_enriched = enrich_dfs.enrich_df_trials_fm(df_trials_fm)
    if len(df_fip):
        [df_fip_all, df_trials_fip_enriched] = enrich_dfs.enrich_fip_in_df_trials(df_fip, df_trials_enriched)
        (df_fip_final, df_trials_final, df_trials_fip) = enrich_dfs.remove_tonic_df_fip(df_fip_all, df_trials_enriched, df_trials_fip_enriched)
    else:
        warnings.warn(f"channels {interested_channels} not found in df_fip.")
        df_fip_final = df_fip
        df_trials_final = df_trials 
    
    # return all dataframes
    return (df_sess, df_trials_final, df_events, df_fip_final) 
      

def run_analysis(analysis_dispatch_inputs: AnalysisDispatchModel, **parameters) -> None:
    processing = construct_processing_record(analysis_dispatch_inputs, **parameters)
    
# DRY RUN
    if docdb_record_exists(processing):
        logger.info("Record already exists, skipping.")
        return

    df = get_session_table(if_load_bpod=False)
    subject_id = analysis_dispatch_inputs.file_location[0].split('behavior_')[1].split('_')[0]
    df_trained = df[(df['subject_id'] == subject_id) & (df['current_stage_actual'].isin(['STAGE_FINAL','GRADUATED']))]
    session_names = [
        f"{row['subject_id']}_{row['session_date'].strftime('%Y-%m-%d')}"
        for _, row in df_trained.iterrows()
    ]
    filtered_file_locations = [
        f for f in analysis_dispatch_inputs.file_location
        if any(session_name in f for session_name in session_names)
    ]

    (df_sess, df_trials, df_events, df_fip) = get_nwb_processed(analysis_dispatch_inputs.file_location, **parameters)

    # prepare computations for plotting 

    df_trials['reward_all'] = df_trials['earned_reward'] + df_trials['extra_reward']
    # Compute num_reward_past and num_no_reward_past
    df_trials['reward_shifted'] = df_trials.groupby('ses_idx')['reward_all'].shift(1)  # Shift to look at past values

    df_trials['num_reward_past'] = df_trials.groupby(
                            (df_trials['reward_shifted'] != df_trials['reward_all']).cumsum()).cumcount() + 1

    # Set 'NA' for mismatched reward types
    df_trials.loc[df_trials['reward_all'] == 0, 'num_reward_past'] = df_trials.loc[df_trials['reward_all'] == 0, 'num_reward_past']* -1 

    # Drop the temporary column
    df_trials.drop(columns=['reward_shifted'], inplace=True)


    RPE_binned3_label_names = [str(np.round(i,2)) for i in np.arange(-1,1.1,1/3)]

    df_trials['RPE-binned3'] = pd.cut(df_trials['RPE_all'],# all versus earned not a huge difference
                        np.arange(-1,1.5,1/3), labels=[str(np.round(i,2)) for i in np.arange(-1,1.01,1/3)])

    (df_sess, nwbs_by_week) = analysis_util.get_dummy_nwbs_by_week(df_sess, df_trials, df_events, df_fip) 


    # get average activity 
    data_column = 'data_z_norm'
    alignment_event='choice_time_in_session'
    rpe_slope_dict = {}
    for channel in list(analysis_specification["channels"].keys()):
        avg_signal_col = summary_plots.output_col_name(channel, data_column, alignment_event)
        for nwb_week in nwbs_by_week:
        
            nwb_week = trial_metrics.get_average_signal_window_multi(
                            nwb_week,
                            alignment_event='choice_time_in_session',
                            offsets=[0.33, 1],
                            channel=channel,
                            data_column=data_column,
                            output_col = avg_signal_col
                        )
        
        # get rpe slope per session 

        df_trials_all = pd.concat([nwb.df_trials for nwb_week in nwbs_by_week for nwb in nwb_week])
        rpe_slope = []
        for ses_idx in sorted(df_trials_all['ses_idx'].unique()):
            
            data = df_trials_all[df_trials_all['ses_idx'] == ses_idx]
            data = data.dropna(subset = [avg_signal_col, 'RPE_all'])
            if len(data) == 0:
                continue
            data_neg = data[data['RPE_all'] < 0]
            data_pos = data[data['RPE_all'] >= 0]

            ses_date = pd.to_datetime(ses_idx.split('_')[1])
            (_,_, slope_pos) = summary_plots.get_RPE_by_avg_signal_fit(data_pos, avg_signal_col)
            (_,_, slope_neg) = summary_plots.get_RPE_by_avg_signal_fit(data_neg, avg_signal_col)
            rpe_slope.append([ses_date, slope_pos, slope_neg])
        rpe_slope = pd.DataFrame(rpe_slope, columns=['date', 'slope (RPE >= 0)', 'slope (RPE < 0)'])
        rpe_slope_dict[channel] = rpe_slope

    subject_id = df_sess['subject_id'].unique()[0]
    # Concatenate with keys, turning dict keys into an index
    combined_rpe_slope = pd.concat(rpe_slope_dict, names=["channel"])
    combined_rpe_slope = combined_rpe_slope.reset_index(level="channel").reset_index(drop=True)

    combined_rpe_slope.to_csv(f"/results/{subject_id}_rpe_slope.csv")


    # plot summary plots
    plot_loc = '/results/plots/'

    if not os.path.exists(plot_loc):
        os.makedirs(plot_loc)

    if "avg_lastN_sess" in parameters["plot_types"]:
        summary_plots.plot_avg_final_N_sess(df_sess, nwbs_by_week, parameters["channels"], final_N_sess = 5, loc = plot_loc)
    nwbs_all = [nwb for nwb_week in nwbs_by_week for nwb in nwb_week]
    for channel, channel_loc in parameters['channels'].items():
        if "all_sess" in parameters["plot_types"]:
            summary_plots.plot_all_sess(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)
        if "weekly" in parameters["plot_types"]:
            summary_plots.plot_weekly_grid(df_sess, nwbs_by_week, rpe_slope_dict[channel], channel, channel_loc, loc=plot_loc)
        
    # # DRY RUN (comment in or out)
    write_results_and_metadata(processing, ANALYSIS_BUCKET)
    logger.info(f"Successfully wrote record to docdb and s3")


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


        run_analysis(analysis_dispatch_inputs, **analysis_specification)
