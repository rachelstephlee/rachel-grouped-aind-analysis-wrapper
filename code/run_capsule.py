from analysis_pipeline_utils.analysis_dispatch_model import \
    AnalysisDispatchModel

from analysis_pipeline_utils.utils_analysis_wrapper import (
    run_analysis_jobs)

# ======================================================================
# USER MUST EDIT THIS SECTION
#
# 1. Import your analysis specification and outputs
# 2. Update the aliases below
#
# Do NOT modify any code outside this section except run_analysis().
# ======================================================================

from example_analysis_model import (
    ExampleAnalysisSpecification,
    ExampleAnalysisOutputs,
)

AnalysisSpecification = ExampleAnalysisSpecification
AnalysisOutputModel = ExampleAnalysisOutputs


### USER EDITABLE FUNCTION WHERE ANALYSIS IS EXECUTED
def run_analysis(
    analysis_dispatch_inputs: AnalysisDispatchModel,
    analysis_specification: AnalysisSpecification
) -> dict | None:
    """
    Runs the analysis

    Parameters
    ----------
    analysis_dispatch_inputs: AnalysisDispatchModel
        The input model from the dispatcher
    
    analysis_specification: AnalysisSpecification
        The user specified analysis parameters model

    Returns
    -------
    dict | None
        Any output parameters that will be stored or None if no output model
    """

    # Execute analysis and write to results folder
    # using the passed parameters
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
        analysis_input_model=AnalysisSpecification,
        analysis_output_model=AnalysisOutputModel,
        run_function=run_analysis
    )

