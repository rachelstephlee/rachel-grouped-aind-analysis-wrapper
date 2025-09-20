"""
This is an example of an analysis-specific schema for the parameters required by that analysis
"""

from typing import List, Optional, Union

from aind_data_schema.base import GenericModel
from pydantic import Field
from pydantic_settings import BaseSettings


class SummaryPlotsAnalysisSpecification(GenericModel):
    """
    Represents the specification for an analysis, including its name,
    version, libraries to track, and parameters.
    """
    name: str=Field(description="name of analysis")
    plot_types: str=Field(description="types of plots to generate", default = "avg_lastN_sess")
    last_N_sess: int=Field(description="number of last sessions to plot", default = 5)
    channels: dict[str, str] = Field(..., description="Dictionary of channels to plot from. Keys = channel name, Value = intended location and measurement")
    fitted_model: str=Field(default = "QLearning_L2F1_CKfull_softmax", description="Qlearning model fitted to get RPE")

# only saving plots, no outputs needed 
# class SummaryResultsAnalysisOutputs(GenericModel):
#     """
#     Represents the outputs of an analysis, including a list of ISI violations.
#     """

#     isi_violations: List[Union[str, int]] = Field(
#         ..., description="List of ISI violations detected by the analysis"
#     )
#     additional_info: Optional[str] = Field(
#         default=None, description="Additional information about the analysis"
#     )

class SummaryPlotsAnalysisSpecificationCLI(
    SummaryPlotsAnalysisSpecification, BaseSettings, cli_parse_args=True
):
    """
    This class is needed only if you want to parse settings passed from the command line (including the app builder)
    """

    pass
