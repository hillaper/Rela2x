"""
Spectral density functions and relaxation constants.

The module provides the Lorentzian spectral density, the Schofield factor that
distinguishes the quantum mechanical theory from the semiclassical one, and the
symbolic spectral density functions attached to pairs of interactions.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import re

import sympy as smp

from rela2x._core import _settings
from rela2x._core._constants import beta, tau_c
from rela2x._core._utils import sort_interactions


def Lorentzian(
    w: smp.Expr,
    tau_c: smp.Expr,
    fast_motion_limit: bool=False,
    slow_motion_limit: bool=False,
) -> smp.Expr:
    """
    Evaluate a Lorentzian function (normalized to `tau_c` at w = 0), used for
    spectral density functions.

    Parameters
    ----------
    w : sympy.Expr
        Frequency.
    tau_c : sympy.Expr
        Correlation time.
    fast_motion_limit : bool, optional
        Whether to use the fast-motion limit where ``(w * tau_c) << 1``. Default is False.
    slow_motion_limit : bool, optional
        Whether to use the slow-motion limit where ``(w * tau_c) >> 1``. Default is False.

    Returns
    -------
    sympy.Expr
        ``J(w) = tau_c / (1 + (w * tau_c)^2)``, or the requested motional-limit approximation thereof.
    """
    # Handle division by zero at w = 0.
    if w == 0:
        return tau_c
    else:
        if fast_motion_limit:
            return tau_c
        elif slow_motion_limit:
            return 1 / (w**2 * tau_c)
        else:
            return tau_c / (1 + (w * tau_c)**2)


def Schofield_theta(w: smp.Expr) -> smp.Expr:
    """
    Evaluate the Schofield thermal correction factor used in the quantum
    mechanical spectral density function.

    NOTE: ``beta = hbar / (k_B * T)``, defined in `_constants.py`.

    Parameters
    ----------
    w : sympy.Expr
        Frequency.

    Returns
    -------
    sympy.Expr
        ``exp(-beta * w / 2)``.
    """
    return smp.exp(-beta * w / 2)


def _even_function(name: str) -> smp.core.function.UndefinedFunction:
    """
    Create an undefined function that is even in its argument.

    The spectral density functions satisfy J(w) = J(-w) when the imaginary 
    part of the Fourier transform of the time correlation function
    (responsible for the dynamic frequency shift) is neglected. Building that symmetry
    into the function itself means the two arguments collapse onto a single expression,
    without an absolute value having to be written around every frequency.

    Parameters
    ----------
    name : str
        Name of the function.

    Returns
    -------
    sympy.core.function.UndefinedFunction
        Undefined function that canonicalises the sign of its argument.
    """
    def eval_even(cls, argument):
        """
        Rewrite a negated argument, leaving anything else for SymPy to handle.

        Parameters
        ----------
        cls : sympy.core.function.UndefinedFunction
            The function being applied.
        argument : sympy.Expr
            Argument the function is applied to.

        Returns
        -------
        sympy.Expr or None
            The function of the negated argument if the argument carries a leading minus
            sign, and None otherwise, which leaves the application unevaluated.
        """
        if argument.could_extract_minus_sign():
            return cls(-argument)

    function = smp.Function(name)
    function.eval = classmethod(eval_even)

    return function


def J_w(
    intr1: str,
    intr2: str,
    l: int,
    argument: smp.Expr,
) -> smp.Expr:
    """
    Build the symbolic spectral density function J(w) for a pair of interactions.

    Parameters
    ----------
    intr1 : str
        Name of the first interaction.
    intr2 : str
        Name of the second interaction.
    l : int
        Rank of the spherical tensor operator (the projection q = 0, by Hubbard's result).
    argument : sympy.Expr
        Argument of the spectral density function (a combination of angular frequencies).

    Returns
    -------
    sympy.Expr
        ``J(w)`` for the semiclassical theory, or ``J(w) * exp(-beta * w / 2)``
        for the quantum mechanical theory, depending on `_settings.RELAXATION_THEORY`.
    """
    intr_sorted = sort_interactions(intr1, intr2)

    # The projection carried by the subscript is zero for every spectral density function,
    # as a consequence of Hubbard's result for time-correlation functions of
    # irreducible spherical tensor components. 
    q = 0

    # If the same interaction appears twice, use it only once in the superscript.
    # NOTE: The function is even in its argument, so J(w) and J(-w) collapse onto the same
    # expression without the argument having to be wrapped in an absolute value.
    if isinstance(intr_sorted, str):
        expr = _even_function(f'J^{{{intr_sorted}}}_{{{l}{q}}}')(argument)
    else:
        expr = _even_function(f'J^{{{intr_sorted[0]}, {intr_sorted[1]}}}_{{{l}{q}}}')(argument)

    if _settings.RELAXATION_THEORY == 'sc':
        return expr
    elif _settings.RELAXATION_THEORY == 'qm':
        return expr * Schofield_theta(argument)


def J_w_isotropic_rotational_diffusion(
    intr1: str,
    intr2: str,
    l: int,
    argument: smp.Expr,
    fast_motion_limit: bool=False,
    slow_motion_limit: bool=False,
) -> smp.Expr:
    """
    Build the isotropic rotational diffusion spectral density function, which
    takes the form of a Lorentzian function.

    Parameters
    ----------
    intr1 : str
        Name of the first interaction.
    intr2 : str
        Name of the second interaction.
    l : int
        Rank of the spherical tensor operator. NOTE: applies only for l > 0.
    argument : sympy.Expr
        Argument of the spectral density function (a combination of angular frequencies).
    fast_motion_limit : bool, optional
        Whether to use the fast-motion limit where ``(w * tau_c) << 1``. Default is False.
    slow_motion_limit : bool, optional
        Whether to use the slow-motion limit where ``(w * tau_c) >> 1``. Default is False.

    Returns
    -------
    sympy.Expr
        Spectral density function J(w) in the isotropic rotational diffusion model.
    """
    intr_sorted = sort_interactions(intr1, intr2)
    tau_c_l = 6*tau_c / (l*(l+1))

    # The projection carried by the subscript is zero, matching the spectral density functions.
    q = 0

    if isinstance(intr_sorted, str):
        G = smp.Function(f'G^{{{intr_sorted}}}_{{{l}{q}}}')(0)
    else:
        G = smp.Function(f'G^{{{intr_sorted[0]}, {intr_sorted[1]}}}_{{{l}{q}}}')(0)

    J_w = 2*G * Lorentzian(argument, tau_c_l, fast_motion_limit=fast_motion_limit, slow_motion_limit=slow_motion_limit)
    return J_w


# Helper functions for RelaxationSuperoperator object
def extract_J_w_symbols_and_args(J: smp.Function) -> tuple[tuple[str, ...], tuple[int, ...], smp.Expr]:
    """
    Extract the interaction names, (l, q) indices and argument encoded in a
    spectral density function symbol J(w) (or G(0) in the case of isotropic
    rotational diffusion).

    NOTE: Used when substituting symbols and functions in the relaxation superoperator.

    Parameters
    ----------
    J : sympy.Function
        Spectral density function symbol.

    Returns
    -------
    intrs : tuple of str
        Interaction names encoded in `J`.
    lq : tuple of int
        Rank and projection ``(l, q)`` of the spherical tensor operator encoded in `J`.
    arg : sympy.Expr
        Argument of the spectral density function.
    """
    J_str = str(J.func)

    intrs = re.findall(r'\^{(.*?)}', J_str)[0]
    lq_string = re.findall(r'\_\{(\d+)\}', J_str)[0]

    intrs = tuple(intrs.split(', ')) if isinstance(intrs, tuple) else tuple([intrs, intrs])

    # The subscript holds the rank followed by the projection. The projection is the single
    # digit zero for every spectral density function, so it occupies the final character alone.
    lq = (int(lq_string[:-1]), int(lq_string[-1]))
    arg = J.args[0]

    return intrs, lq, arg
