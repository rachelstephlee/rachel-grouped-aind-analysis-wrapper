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
    plot_types: list[str]=Field(description="types of plots to generate", default = ["all_sess"])
    last_N_sess: int=Field(description="number of last sessions to plot", default = 5)
    channels: dict[str, str] = Field(..., description="Dictionary of channels to plot from. \
                    Keys = patch cord to load, Value = intended location and measurement. \
                    NO preprocessing method included in suffix. \
                    When curation_csv is set the values are ignored in favor of the CSV's targets, \
                    and every key must be covered by the CSV or the run errors.")
    preprocessing: str=Field(description="preprocessing_method", default = "dff-bright_mc-iso-IRLS")
    curation_csv: str | None=Field(default=None, description="Fiber curation results file, relative to the \
                    data folder of aind_bwnm_fiber_data_curation_utils and without the .csv \
                    extension, e.g. \
                    'bilateral_4_channels/curation_results/bilateral_4_channels_results'. \
                    Sets each channel's intended measurement per session and drops fibers that \
                    failed curation. None means no curation.")

    fitted_model: str=Field(default = "QLearning_L2F1_CKfull_softmax", description="Qlearning model fitted to get RPE")
    dry_run: bool=Field(default=True, description="Dry run")
    save_dfs: bool=Field(default=False, description="Save the dataframes for the analysis")
    plot_save_format: str=Field(default="png", description="Format to save plots in, e.g. png, pdf, etc.")

# removing pearson pairs for now
    # pearson_pairs: list[tuple[str, str]] = Field(
    #     default_factory=list,
    #     description="List of channel pairs to compute Pearson r for; each pair is (channel1, channel2) without the preprocessing method. \
    #                 With curation_csv set, name the pairs by target (e.g. 'latNAcc(L)-DA') rather than patch cord."
    # )
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
