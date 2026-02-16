from pydantic import Field
from typing import List, Optional, Union

from aind_data_schema.base import GenericModel

from analysis_pipeline_utils.metadata import \
    get_metadata_for_records

from analysis_pipeline_utils.analysis_dispatch_model import \
    AnalysisDispatchModel

from analysis_pipeline_utils.utils_analysis_wrapper import (
    run_analysis_jobs)
from dotenv import load_dotenv

# TODO: use pydantic settings instead
load_dotenv("settings.env")

# ======================================================================
# USER MUST EDIT THIS SECTION
#
# 1. Implement  your analysis specification and any output analysis model
# 2. Update the aliases below
#
# Do NOT modify any code outside this section except run_analysis().
# ======================================================================

"""
This is an example of an analysis-specific schema
for the parameters required by that analysis
"""

class ExampleAnalysisParameters(GenericModel):
    """
    Represents the specification for an analysis, including its name,
    version, libraries to track, and parameters.
    """

    analysis_name: str = Field(
        ..., description="User-defined name for the analysis"
    )
    analysis_tag: str = Field(
        ...,
        description=(
            "User-defined tag to organize results "
            "for querying analysis output",
        ),
    )
    isi_violations_cutoff: float = Field(
        ..., description="The value to be using when filtering units by this"
    )


class ExampleAnalysisOutputs(GenericModel):
    """
    Represents the outputs of an analysis, including a list of ISI violations.
    """

    isi_violations: List[Union[str, int]] = Field(
        ..., description="List of ISI violations detected by the analysis"
    )
    additional_info: Optional[str] = Field(
        default=None, description="Additional information about the analysis"
    )

AnalysisInputModel = ExampleAnalysisParameters
AnalysisOutputModel = ExampleAnalysisOutputs


### USER EDITABLE FUNCTION WHERE ANALYSIS IS EXECUTED
def run_analysis(
    analysis_dispatch_inputs: AnalysisDispatchModel,
    analysis_parameters: AnalysisInputModel
) -> dict | None:

    # Execute analysis and write to results folder
    # using the passed parameters
    # Example of fetching metadata record from the dispatcher model:
    # Returns a list of records where each record is a dictionary with the metadata. Example below:
    #     metadata_records = get_metadata_for_records(analysis_dispatch_inputs)
    #     first_record = metadata_records[0]
    #     data_description = first_record["data_description"]
    # Example:
    # Use NWBZarrIO to reads
    # for location in analysis_dispatch_inputs.file_location:
    #     with NWBZarrIO(location, 'r') as io:
    #         nwbfile = io.read()
    #     run_your_analysis(nwbfile, analysis_specification)
    # OR
    #     subprocess.run(["--param_1": analysis_specification.param_1])
    

    ### RETURN DICTIONARY MODEL OF OUTPUT PARAMETERS
    output_parameters = {
        "isi_violations": ["example_violation_1", "example_violation_2"],
        "additional_info":
            "This is an example of additional information about the analysis."
    }

    # IF NO OUTPUT PARAMETERS DESIRED, RETURN NONE
    return output_parameters

if __name__ == "__main__":
    run_analysis_jobs(
        analysis_input_model=AnalysisInputModel,
        analysis_output_model=AnalysisOutputModel,
        run_function=run_analysis
    )

