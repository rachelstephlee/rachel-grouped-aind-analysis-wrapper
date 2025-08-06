import json
import logging
import os

from analysis_pipeline_utils.analysis_dispatch_model import \
    AnalysisDispatchModel
from analysis_pipeline_utils.metadata import (construct_processing_record,
                                              docdb_record_exists,
                                              write_results_and_metadata)
from analysis_pipeline_utils.utils_analysis_wrapper import (
    get_analysis_model_parameters, make_cli_model)

from example_analysis_model import (ExampleAnalysisOutputs,
                                    ExampleAnalysisSpecification)

ANALYSIS_BUCKET = os.getenv("ANALYSIS_BUCKET")
logger = logging.getLogger(__name__)


def run_analysis(
    analysis_dispatch_inputs: AnalysisDispatchModel,
    dry_run: bool = True,
    **parameters,
) -> None:
    """
    Runs the analysis

    Parameters
    ----------
    analysis_dispatch_inputs: AnalysisDispatchModel
        The input model with input data
        from dispatcher

    dry_run: bool, Default True
        Dry run of analysis. If true,
        does not post results

    parameters
        The analysis model parameters

    """
    processing = construct_processing_record(
        analysis_dispatch_inputs, **parameters
    )
    if docdb_record_exists(processing):
        logger.info("Record already exists, skipping.")
        return

    # Execute analysis and write to results folder
    # using the passed parameters
    # Example:
    # Use NWBZarrIO to reads
    # for location in analysis_dispatch_inputs.file_location:
    #     with NWBZarrIO(location, 'r') as io:
    #         nwbfile = io.read()
    #     run_your_analysis(nwbfile, **parameters)
    # OR
    #     subprocess.run(["--param_1": parameters["param_1"]])

    processing.output_parameters = ExampleAnalysisOutputs(
        isi_violations=["example_violation_1", "example_violation_2"],
        additional_info=(
            "This is an example of additional information about the analysis."
        )[0],
    )

    if not dry_run:
        logger.info("Running analysis and posting results")
        write_results_and_metadata(processing, ANALYSIS_BUCKET)
        logger.info("Successfully wrote record to docdb and s3")
    else:
        logger.info("Dry run complete. Results not posted")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cli_cls = make_cli_model(ExampleAnalysisSpecification)
    cli_model = cli_cls()
    logger.info(f"Command line args {cli_model.model_dump()}")
    input_model_paths = tuple(cli_model.input_directory.glob("job_dict/*"))
    logger.info(
        f"Found {len(input_model_paths)} input job models to run analysis on."
    )

    for model_path in input_model_paths:
        with open(model_path, "r") as f:
            analysis_dispatch_inputs = AnalysisDispatchModel.model_validate(
                json.load(f)
            )
        merged_parameters = get_analysis_model_parameters(
            analysis_dispatch_inputs,
            cli_model,
            ExampleAnalysisSpecification,
            analysis_parameters_json_path=cli_model.input_directory
            / "analysis_parameters.json",
        )
        analysis_specification = ExampleAnalysisSpecification.model_validate(
            merged_parameters
        ).model_dump()
        logger.info(f"Running with analysis specs {analysis_specification}")
        run_analysis(
            analysis_dispatch_inputs,
            bool(cli_model.dry_run),
            **analysis_specification,
        )
