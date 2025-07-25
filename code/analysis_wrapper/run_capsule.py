import json
import logging
import os
from pathlib import Path

from aind_analysis_results.analysis_dispatch_model import AnalysisDispatchModel
from aind_analysis_results.metadata import (construct_processing_record,
                                            docdb_record_exists,
                                            write_results_and_metadata)

import analysis_wrapper.utils as utils
from analysis_wrapper.example_analysis_model import ExampleAnalysisSpecification, ExampleAnalysisOutputs

DATA_PATH = Path("/data")  # TODO: don't hardcode
ANALYSIS_BUCKET = os.getenv("ANALYSIS_BUCKET")
logger = logging.getLogger(__name__)


def run_analysis(
    analysis_dispatch_inputs: AnalysisDispatchModel, **parameters
) -> None:
    processing = construct_processing_record(
        analysis_dispatch_inputs, **parameters
    )
    if docdb_record_exists(processing):
        logger.info("Record already exists, skipping.")
        return

    # Execute analysis and write to results folder
    # using the passed parameters
    # SEE EXAMPLE BELOW
    # Use NWBZarrIO to reads
    # for location in analysis_dispatch_inputs.file_location:
    #     with NWBZarrIO(location, 'r') as io:
    #         nwbfile = io.read()

    # acquisition_keys = list(nwbfile.acquisition.keys())
    # with open('/results/acquisition_keys.json', 'w') as f:
    #     json.dump(acquisition_keys, f)

    processing.output_parameters = ExampleAnalysisOutputs(
        isi_violations=["example_violation_1", "example_violation_2"],
        additional_info=(
            "This is an example of additional information about the analysis."
        )[0],
    )
    write_results_and_metadata(processing, ANALYSIS_BUCKET)
    logger.info("Successfully wrote record to docdb and s3")


# Most of the below code will not need to change per-analysis
# and will be moved to a shared library
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_model_paths = tuple(DATA_PATH.glob("job_dict/*"))
    logger.info(
        f"Found {len(input_model_paths)} input job models to run analysis on."
    )

    for model_path in input_model_paths:
        with open(model_path, "r") as f:
            analysis_dispatch_inputs = AnalysisDispatchModel.model_validate(
                json.load(f)
            )
        merged_parameters = utils.get_analysis_model_parameters(
             analysis_dispatch_inputs, ExampleAnalysisSpecification, 
             analysis_parameters_json_path=DATA_PATH / "analysis_parameters.json"
        )
        analysis_specification = ExampleAnalysisSpecification.model_validate(
            merged_parameters
        ).model_dump()
        logger.info(f"Running with analysis specs {analysis_specification}")
        run_analysis(analysis_dispatch_inputs, **analysis_specification)
