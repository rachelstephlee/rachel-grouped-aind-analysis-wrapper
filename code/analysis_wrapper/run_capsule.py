import json
import logging
import os
from pathlib import Path

import sys
script_dir = "/root/capsule/code/analysis_wrapper"
if script_dir in sys.path:
    sys.path.remove(script_dir)
# sys.path.insert(0,"/root/capsule/")


from analysis_pipeline_utils.metadata import construct_processing_record, docdb_record_exists, write_results_and_metadata
from analysis_pipeline_utils.analysis_dispatch_model import AnalysisDispatchModel
import analysis_wrapper.utils as utils
from analysis_wrapper.analysis_model import (
    SummaryPlotsAnalysisSpecification, SummaryPlotsAnalysisSpecificationCLI
)
import subprocess

DATA_PATH: Path = Path("/data")  # TODO: don't hardcode
ANALYSIS_BUCKET = os.getenv("ANALYSIS_BUCKET")
logger = logging.getLogger(__name__)


        

def run_analysis(analysis_dispatch_inputs: AnalysisDispatchModel, **parameters) -> None:
    processing = construct_processing_record(analysis_dispatch_inputs, **parameters)
    if docdb_record_exists(processing):
        logger.info("Record already exists, skipping.")
        return

    # all_metadata = utils.get_metadata(analysis_dispatch_inputs)
    # Execute analysis and write to results folder
    # using the passed parameters
    # SEE EXAMPLE BELOW
    # Use NWBZarrIO to reads
    for location in analysis_dispatch_inputs.file_location:
        with NWBZarrIO(location, 'r') as io:
            nwbfile = io.read()
    #     run_your_analysis(nwbfile, **parameters)
    # OR
    #     subprocess.run(["--param_1": parameters["param_1"]])
    # (df_trials, df_events, df_fip) = co_utils.get_all_df_for_nwb(filename_sessions=analysis_dispatch_inputs.file_location, interested_channels = parameters["channels"])



            
    # acquisition_keys = list(nwbfile.acquisition.keys())
    # with open('/results/acquisition_keys.json', 'w') as f:
    #     json.dump(acquisition_keys, f)
        

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
