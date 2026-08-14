"""
Status message functions used throughout Rela2x.

This module provides lightweight helper functions for printing status
messages and section headers when verbose output is enabled.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

from rela2x._core import _settings


def status(msg: str) -> None:
    """
    Print a status message when verbose output is enabled.

    Parameters
    ----------
    msg : str
        Message to print.
    """

    # Print the message only when verbose output is enabled.
    if _settings.VERBOSE:
        print(msg)


def status_section(title: str) -> None:
    """
    Print a titled separator for a new section in the status output.

    This function is intended for visually separating different stages of the
    computation when verbose output is enabled. It prints a separator line,
    the centred title, and a second separator line.

    Parameters
    ----------
    title : str
        Title to print for the new section in the status output.
    """

    # Print the section header only when verbose output is enabled.
    if _settings.VERBOSE:
        print("#" * 80)
        print(title.center(80))
        print("#" * 80)
        print()
