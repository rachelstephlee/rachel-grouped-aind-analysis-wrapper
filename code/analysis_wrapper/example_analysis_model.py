"""
This is an example of an analysis-specific schema for the parameters required by that analysis
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExampleAnalysisSpecification(BaseModel):
    """
    Represents the specification for an analysis, including its name,
    version, libraries to track, and parameters.
    """

    analysis_name: str = Field(..., title="User-defined name for the analysis")
    analysis_tag: str = Field(..., title="User-defined tag to organize results for querying analysis output")
    isi_violations_cutoff: float = Field(
         ..., title="The value to be using when filtering units by this"
    )

class ExampleAnalysisSpecificationCLI(
    ExampleAnalysisSpecification, BaseSettings, cli_parse_args=True
):
    """
    This class is needed only if you want to parse settings passed from the command line (including the app builder)
    """

    pass
