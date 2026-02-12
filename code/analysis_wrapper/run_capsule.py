import json
import logging
import os
from pathlib import Path
from hdmf_zarr import NWBZarrIO



import sys
script_dir = "/root/capsule/code/analysis_wrapper"
if script_dir in sys.path:
    sys.path.remove(script_dir)
# sys.path.insert(0,"/root/capsule/")

sys.path.insert(0, str(Path(__file__).resolve().parent))


from analysis_pipeline_utils.metadata import construct_processing_record, docdb_record_exists, write_results_and_metadata
from analysis_pipeline_utils.analysis_dispatch_model import AnalysisDispatchModel
from analysis_wrapper.analysis_model import (
    SummaryPlotsAnalysisSpecification, SummaryPlotsAnalysisSpecificationCLI
)
import subprocess
from rachel_analysis_utils import nwb_utils as r_utils
from rachel_analysis_utils import analysis_utils as analysis_utils


# for analysis code
import numpy as np
import pandas as pd 
from analysis_wrapper.plots import summary_plots
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
    
    if dry_run:
        logger.info("DRY RUN!!!!!!! ")

    (df_sess, df_trials, df_events, df_fip) = r_utils.get_nwb_processed(analysis_dispatch_inputs.file_location, **parameters)



    ############## prepare computations for plotting ###############

    df_trials = analysis_utils.enrich_df_trials(df_trials)

    (df_sess, nwbs_by_week) = r_utils.get_dummy_nwbs_by_week(df_sess, df_trials, df_events, df_fip) 

    if "RPE" in parameters["plot_types"]:
        (nwbs_by_week, combined_rpe_slope) = analysis_utils.add_AUC_and_rpe_slope(nwbs_by_week, parameters)


    # plot summary plots
    if dry_run:
        plot_loc = '/root/capsule/results/plots_TEST/'
    else:
        plot_loc = '/results/plots/'

    if not os.path.exists(plot_loc):
        os.makedirs(plot_loc)


    nwbs_all = [nwb for nwb_week in nwbs_by_week for nwb in nwb_week]

    for channel, channel_loc in parameters['channels'].items():
        if parameters['preprocessing'] is not 'raw':
            channel = channel +  '_' + parameters['preprocessing'] 
        
        if "all_sess" in parameters["plot_types"]:
            logger.info("running NEURAL PSTH")
            summary_plots.plot_all_sess_PSTH(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)

        if "RPE" in parameters["plot_types"]:
            logger.info("running NEURAL PSTH with RPE focus")
            summary_plots.plot_all_sess_RPE(df_sess, nwbs_all, channel, channel_loc, loc = plot_loc)
            
        if "weekly" in parameters["plot_types"]:
            summary_plots.plot_weekly_grid(df_sess, nwbs_by_week, rpe_slope_dict[channel], channel, channel_loc, loc=plot_loc)
    
    if "behavior" in parameters["plot_types"]:
        logger.info("running ALL SESS behavior")
        if len(nwbs_all) > 5:
            nwb_batches = [nwbs_all[i:i+5] for i in range(0, len(nwbs_all), 5)]
            for nwb_batch in nwb_batches:
                summary_plots.plot_all_sess_behavior(df_sess, nwb_batch, loc = plot_loc)
    if "avg_lastN_sess" in parameters["plot_types"]:
        summary_plots.plot_avg_final_N_sess(df_sess, nwbs_by_week, parameters["channels"], final_N_sess = 5, loc = plot_loc)


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
        
