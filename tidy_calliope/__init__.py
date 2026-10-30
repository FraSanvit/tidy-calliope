"""tidy-calliope: post-processing of Calliope 0.7 results into tidy tables."""

__version__ = "0.1.1"

from tidy_calliope.config import load_config  # noqa: E402
from tidy_calliope.core import process, process_file  # noqa: E402
from tidy_calliope.reader import ResultFileError  # noqa: E402

__all__ = ["ResultFileError", "load_config", "process", "process_file", "__version__"]
