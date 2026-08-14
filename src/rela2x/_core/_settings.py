"""
Global settings controlling the behaviour of Rela2x.

The relaxation theory and the verbosity of the progress output are both held
as module-level variables rather than passed around explicitly, so that they
can be set once and take effect throughout the package. Other modules read
them as ``_settings.RELAXATION_THEORY``/``_settings.VERBOSE`` rather than
importing the values directly, so that they always observe the current setting.

The variables should always be changed through `set_relaxation_theory` and
`set_verbose`, which validate the requested value.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

# Level of theory for the relaxation superoperator.
# 'sc' for semiclassical, 'qm' for quantum mechanical.
RELAXATION_THEORY = 'sc'

# Whether the progress messages of `_status.status` and `_status.status_section` are printed.
VERBOSE = True


def set_relaxation_theory(theory: str) -> None:
    """
    Set the level of theory used for the relaxation superoperator.

    Parameters
    ----------
    theory : str
        Level of theory: ``'sc'`` for semiclassical (default) or ``'qm'``
        for quantum mechanical (Lindbladian).

    Raises
    ------
    ValueError
        If `theory` is not one of ``'sc'`` or ``'qm'``.
    """

    # Reject anything other than the two supported levels of theory.
    if theory not in ['sc', 'qm']:
        raise ValueError("Invalid relaxation theory. Choose 'sc' for semiclassical or 'qm' for quantum mechanical.")

    # Rebind the module-level variable so that every reader observes the change.
    global RELAXATION_THEORY
    RELAXATION_THEORY = theory


def set_verbose(verbose: bool) -> None:
    """
    Set whether Rela2x prints progress messages during construction of the
    superoperators.

    Parameters
    ----------
    verbose : bool
        Whether to print progress messages. Default is True.

    Raises
    ------
    ValueError
        If `verbose` is not a boolean.
    """

    # Reject anything other than an actual boolean.
    if not isinstance(verbose, bool):
        raise ValueError("Invalid verbose setting. Choose True or False.")

    # Rebind the module-level variable so that every reader observes the change.
    global VERBOSE
    VERBOSE = verbose
