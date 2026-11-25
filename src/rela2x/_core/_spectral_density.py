"""
Spectral density functions and relaxation constants.

The module provides the Lorentzian spectral density, the Schofield factor that
distinguishes the quantum mechanical theory from the semiclassical one, the
symbolic spectral density functions attached to pairs of interactions, and the
expressions for their amplitudes G(0) in terms of the interaction parameters.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import re

import sympy as smp

from rela2x._core import _settings
from rela2x._core._constants import beta, tau_c
from rela2x._core._nmr_isotopes import spin_quantum_numbers
from rela2x._core._symbols import (
    C_Q_symbol,
    Delta_J_symbol,
    Delta_sigma_symbol,
    J_a_symbol,
    amplitude_symbol,
    b_coupling_symbol,
    sigma_a_symbol,
    w_symbol,
)
from rela2x._core._utils import sort_interactions


# Mechanism names that `interaction_amplitude` understands, with the interaction type each of them requires.
RECOGNISED_MECHANISMS = {'CSA': '1L', 'SH': '1L', 'DD': '2', 'Q': '1Q', 'J': '2'}


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


# Amplitudes G(0) of the spectral density functions
def axial_tensor(
    delta: smp.Expr,
    axis: list,
) -> smp.Matrix:
    """
    Build an axially symmetric, traceless rank-2 coupling tensor.

    Parameters
    ----------
    delta : sympy.Expr
        Anisotropy ``A_parallel - A_perpendicular`` of the tensor, as an
        angular frequency.
    axis : list
        Cartesian components of the symmetry axis. The length is irrelevant.

    Returns
    -------
    sympy.Matrix
        The tensor ``delta * (n n^T - 1/3)``, with `n` the unit vector along `axis`.
    """
    n = smp.Matrix(axis)
    return delta * (n * n.T / n.dot(n) - smp.eye(3) / 3)


def G0_from_tensors(
    A_1: smp.MatrixBase,
    A_2: smp.MatrixBase,
    l: int=2,
) -> smp.Expr:
    """
    Evaluate the amplitude G(0) of two interactions from their coupling tensors.

    NOTE: Holds for a rigid molecule in an isotropic liquid,
    ``G(0) = (A_1 : A_2) / (2l + 1)``, where ``:`` is the sum of the products of
    the corresponding elements of the rank-l parts of the two tensors. Both
    tensors must be given in the same molecular frame. No axial symmetry is
    assumed.

    NOTE: The tensors enter the Hamiltonian, in angular frequency units, as
    ``I . A . S`` (two spins), ``I . A . n`` (spin and magnetic field, with `n`
    the unit vector along the field and the field strength contained in `A`) or
    ``S . A . S`` (quadrupolar).

    Parameters
    ----------
    A_1 : sympy.Matrix or array_like
        3x3 Cartesian coupling tensor of the first interaction.
    A_2 : sympy.Matrix or array_like
        3x3 Cartesian coupling tensor of the second interaction.
    l : int, optional
        Rank: 1 for the antisymmetric parts, 2 for the symmetric traceless
        parts of the tensors. Default is 2.

    Returns
    -------
    sympy.Expr
        Amplitude G(0).
    """
    # Extract the rank-l part of both tensors.
    parts = []
    for A in (A_1, A_2):
        A = smp.Matrix(A)
        if l == 1:
            parts.append((A - A.T) / 2)
        else:
            parts.append((A + A.T) / 2 - A.trace() / 3 * smp.eye(3))

    # Contract the two parts and average over the 2l + 1 components.
    return smp.simplify(sum(parts[0].multiply_elementwise(parts[1])) / (2 * l + 1))


def G0_from_amplitudes(
    amplitude_1: smp.Expr,
    amplitude_2: smp.Expr,
    theta: smp.Expr=0,
    l: int=2,
) -> smp.Expr:
    """
    Evaluate the amplitude G(0) of two axially symmetric interactions.

    NOTE: The special case of `G0_from_tensors` for ``l = 2`` tensors of axial
    symmetry, ``G(0) = (2/15) delta_1 delta_2 P_2(cos(theta))``, and for
    ``l = 1`` antisymmetric tensors, ``G(0) = (2/3) a_1 a_2 cos(theta)``.

    Parameters
    ----------
    amplitude_1 : sympy.Expr
        For ``l = 2`` the anisotropy of the first coupling tensor, for ``l = 1``
        the length of its axial vector, as an angular frequency.
    amplitude_2 : sympy.Expr
        The same for the second coupling tensor.
    theta : sympy.Expr, optional
        Angle between the two axes. Default is 0, as for an auto-correlation.
    l : int, optional
        Rank, 1 or 2. Default is 2.

    Returns
    -------
    sympy.Expr
        Amplitude G(0).
    """
    if l == 1:
        return smp.Rational(2, 3) * amplitude_1 * amplitude_2 * smp.cos(theta)
    return smp.Rational(2, 15) * amplitude_1 * amplitude_2 * (3 * smp.cos(theta)**2 - 1) / 2


def interaction_amplitude(
    mechanism: str,
    interaction_type: str,
    spin_indices: tuple[int, ...],
    spin_system: list[str],
    l: int,
) -> tuple[smp.Expr, str]:
    """
    Build the symbolic amplitude of an axially symmetric interaction from its mechanism name.

    NOTE: A mechanism is recognised if its name is a key of
    `RECOGNISED_MECHANISMS` and its interaction type is the one listed there.
    Any other mechanism gets a generic amplitude.

    NOTE: The amplitude is the anisotropy ``A_parallel - A_perpendicular``
    (``l = 2``) or the length of the axial vector of the antisymmetric part
    (``l = 1``) of the coupling tensor of `G0_from_tensors`:

    ============  =========================  ===============================
    Mechanism     l = 2                      l = 1
    ============  =========================  ===============================
    CSA, SH       ``-w * Delta_sigma``       ``-w * sigma_a``
    DD            ``-3 * b``                 0
    Q             ``3 pi C_Q / (2S(2S-1))``  0
    J             ``2 pi Delta_J``           ``2 pi J_a``
    ============  =========================  ===============================

    Parameters
    ----------
    mechanism : str
        Name of the mechanism, e.g. ``'CSA'``.
    interaction_type : str
        Interaction type: ``'1L'``, ``'1Q'`` or ``'2'``.
    spin_indices : tuple of int
        Indices of the spins of the interaction, 0-based.
    spin_system : list of str
        Nuclear isotopes that define the spin system.
    l : int
        Rank, 1 or 2.

    Returns
    -------
    amplitude : sympy.Expr
        Amplitude of the interaction, as an angular frequency.
    description : str
        What the mechanism was taken to be.
    """
    indices = [i + 1 for i in spin_indices]

    # Give any unrecognised mechanism a generic amplitude.
    if RECOGNISED_MECHANISMS.get(mechanism) != interaction_type:
        label = mechanism + ''.join(str(i) for i in indices)
        return amplitude_symbol(label, l), 'unrecognised, generic amplitude'

    if mechanism in ('CSA', 'SH'):
        # NOTE: The Hamiltonian -y I . (1 - sigma) . B gives the coupling tensor -w sigma, with w = -y B.
        w = w_symbol(spin_system[spin_indices[0]])
        anisotropy = Delta_sigma_symbol(*indices) if l == 2 else sigma_a_symbol(*indices)
        return -w * anisotropy, 'nuclear shielding'

    if mechanism == 'DD':
        amplitude = -3 * b_coupling_symbol(*indices) if l == 2 else 0
        return amplitude, 'dipole-dipole coupling'

    if mechanism == 'Q':
        if l != 2:
            return 0, 'quadrupole coupling'
        S = smp.nsimplify(spin_quantum_numbers([spin_system[spin_indices[0]]])[0])
        return 3 * smp.pi * C_Q_symbol(*indices) / (2 * S * (2 * S - 1)), 'quadrupole coupling'

    anisotropy = Delta_J_symbol(*indices) if l == 2 else J_a_symbol(*indices)
    return 2 * smp.pi * anisotropy, 'J-coupling'


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
