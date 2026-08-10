"""
Global settings controlling the behaviour of Rela2x.

The relaxation theory is held as a module-level variable rather than passed
around explicitly, so that it can be set once and take effect throughout the
package. Other modules read it as ``_settings.RELAXATION_THEORY`` rather than
importing the value directly, so that they always observe the current setting.

The variable should always be changed through `set_relaxation_theory`, which
validates the requested level of theory.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

# Level of theory for the relaxation superoperator.
# 'sc' for semiclassical, 'qm' for quantum mechanical.
RELAXATION_THEORY = 'sc'


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
