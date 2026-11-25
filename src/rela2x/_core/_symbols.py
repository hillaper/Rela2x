"""
Symbolic representations of spin operators, expectation values and interaction parameters.

The symbols generated here carry no matrix representation; they are the printed
labels attached to basis operators and to the observables appearing in the
equations of motion, together with the symbolic parameters of the interactions
themselves.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp
import sympy.physics.quantum as smpq

from rela2x._core._constants import t
from rela2x._core._utils import sort_interactions


# Cartesian spin operators
def op_S_symbol(
    direction: str,
    index: int,
) -> smpq.Operator:
    """
    Build a symbolic Cartesian spin operator.

    Parameters
    ----------
    direction : str
        Direction of the spin operator (``'x'``, ``'y'``, ``'z'``, ``'+'`` or ``'-'``).
    index : int
        Spin index.

    Returns
    -------
    sympy.physics.quantum.Operator
        Spin operator symbol.
    """
    return smpq.Operator(f'\\hat{{S}}_{{{direction}}}^{{({index})}}')


def product_op_S_symbol(
    directions: list[str],
    indices: list[int],
) -> smp.Expr:
    """
    Build a symbolic product of Cartesian spin operators.

    Parameters
    ----------
    directions : list of str
        Directions of the spin operators.
    indices : list of int
        Spin indices.

    Returns
    -------
    sympy.Expr
        Product-operator symbol.
    """
    product_op_S = 1
    for direction, index in zip(directions, indices):
        product_op_S *= op_S_symbol(direction, index)
    return product_op_S


# Spherical tensor operators
def op_T_symbol(
    l: int,
    q: int,
    index: int,
) -> smpq.Operator:
    """
    Build a symbolic spherical tensor operator.

    Parameters
    ----------
    l : int
        Rank of the spherical tensor operator.
    q : int
        Projection of the spherical tensor operator.
    index : int
        Spin index.

    Returns
    -------
    sympy.physics.quantum.Operator
        Spherical tensor operator symbol.
    """
    return smpq.Operator(f'\\hat{{T}}_{{{l}{q}}}^{{({index})}}')


def product_op_T_symbol(
    ls: list[int],
    qs: list[int],
    indices: list[int],
) -> smp.Expr:
    """
    Build a symbolic product of spherical tensor operators.

    Parameters
    ----------
    ls : list of int
        Ranks of the spherical tensor operators.
    qs : list of int
        Projections of the spherical tensor operators.
    indices : list of int
        Spin indices.

    Returns
    -------
    sympy.Expr
        Product-operator symbol.
    """
    product_op_T = 1
    for l, q, index in zip(ls, qs, indices):
        product_op_T *= op_T_symbol(l, q, index)
    return product_op_T


def expectation_value(op_symbol: smp.Expr) -> smp.Symbol:
    """
    Build the symbolic expectation value of an operator.

    Parameters
    ----------
    op_symbol : sympy.Expr
        Operator symbol.

    Returns
    -------
    sympy.Symbol
        Symbol representing the expectation value of `op_symbol`.
    """
    # Join product-operator factors directly, without the '*' that the default
    # noncommutative Mul printer would otherwise insert between them.
    if op_symbol.is_Mul:
        op_str = ''.join(str(factor) for factor in op_symbol.args)
    else:
        op_str = str(op_symbol)
    return smp.Symbol('\\langle ' + op_str + '\\rangle')


def f_expectation_value_t(op_symbol: smp.Expr) -> smp.Function:
    """
    Build the symbolic, time-dependent expectation value of an operator.

    Parameters
    ----------
    op_symbol : sympy.Expr
        Operator symbol.

    Returns
    -------
    sympy.Function
        Time-dependent function representing the expectation value of `op_symbol`.
    """
    return smp.Function(expectation_value(op_symbol))(t)


# Interaction parameters
def w_symbol(spin_name: str) -> smp.Symbol:
    """
    Build the symbolic Larmor frequency of a spin.

    NOTE: The symbol is keyed by the isotope label rather than by the spin
    index, so that spins sharing a label also share the symbol. Chemically
    inequivalent nuclei of the same isotope are therefore distinguished by
    giving them distinct labels (see `ISOTOPES` in `_nmr_isotopes.py`), which
    is how the chemical shift enters both the relaxation superoperator and the
    coherent Hamiltonian.

    NOTE: Both the secular approximation of the relaxation superoperator and
    the Zeeman Hamiltonian are built from this function, so that the two
    provably share their Larmor frequency symbols.

    Parameters
    ----------
    spin_name : str
        Nuclear isotope label of the spin.

    Returns
    -------
    sympy.Symbol
        Larmor frequency symbol of the spin, as an angular frequency.
    """
    return smp.Symbol(f'\\omega_{{{spin_name}}}', real=True)


def J_coupling_symbol(
    spin_index_1: int,
    spin_index_2: int,
) -> smp.Symbol:
    """
    Build the symbolic J-coupling constant between two spins.

    NOTE: The spin indices are 1-based, matching the spin operator symbols and
    the interaction names given to the relaxation mechanisms. The labels of the
    two spins are concatenated, so they become ambiguous beyond nine spins,
    which is far outside the system sizes Rela2x is intended for.

    Parameters
    ----------
    spin_index_1 : int
        Index of the first spin.
    spin_index_2 : int
        Index of the second spin.

    Returns
    -------
    sympy.Symbol
        J-coupling constant symbol, as an ordinary frequency (in Hz).
    """
    return smp.Symbol(f'J_{{{spin_index_1}{spin_index_2}}}', real=True)


def D_coupling_symbol(
    spin_index_1: int,
    spin_index_2: int,
) -> smp.Symbol:
    """
    Build the symbolic residual dipolar coupling constant between two spins.

    NOTE: The spin indices are 1-based, exactly as for `J_coupling_symbol`,
    whose labelling convention and nine-spin ambiguity this function shares.

    NOTE: The symbol carries the motionally averaged dipolar coupling that
    survives incomplete averaging, not the rigid-limit dipolar coupling
    constant. The fluctuation about that average is what drives the dipolar
    contribution to the relaxation superoperator, and is described there by the
    spectral density functions instead.

    Parameters
    ----------
    spin_index_1 : int
        Index of the first spin.
    spin_index_2 : int
        Index of the second spin.

    Returns
    -------
    sympy.Symbol
        Residual dipolar coupling constant symbol, as an ordinary frequency
        (in Hz), defined so that it is the splitting it produces in the
        spectrum of a heteronuclear pair.
    """
    return smp.Symbol(f'D_{{{spin_index_1}{spin_index_2}}}', real=True)


def w_Q_symbol(spin_index: int) -> smp.Symbol:
    """
    Build the symbolic residual quadrupolar coupling constant of a spin.

    NOTE: The symbol is keyed by the spin index rather than by the isotope
    label, unlike the Larmor frequency symbol of `w_symbol`, because the
    residual quadrupolar coupling is a property of the local environment of an
    individual nucleus rather than of its isotope.

    NOTE: As for the residual dipolar coupling, the symbol carries the
    motionally averaged quadrupolar coupling rather than the rigid-limit one
    (see `D_coupling_symbol`).

    Parameters
    ----------
    spin_index : int
        Index of the spin.

    Returns
    -------
    sympy.Symbol
        Residual quadrupolar coupling constant symbol, as an angular frequency,
        defined so that it is the spacing it produces between adjacent
        single-quantum transitions of the spin.
    """
    return smp.Symbol(f'\\omega_{{Q}}^{{({spin_index})}}', real=True)


# Parameters of the fluctuating interactions, entering the amplitudes G(0) of the spectral density functions
def Delta_sigma_symbol(spin_index: int) -> smp.Symbol:
    """
    Build the symbolic shielding anisotropy of a spin.

    Parameters
    ----------
    spin_index : int
        Index of the spin, 1-based.

    Returns
    -------
    sympy.Symbol
        Shielding anisotropy ``sigma_parallel - sigma_perpendicular`` of an
        axially symmetric shielding tensor (dimensionless).
    """
    return smp.Symbol(f'\\Delta\\sigma^{{({spin_index})}}', real=True)


def sigma_a_symbol(spin_index: int) -> smp.Symbol:
    """
    Build the symbolic antisymmetric shielding of a spin.

    Parameters
    ----------
    spin_index : int
        Index of the spin, 1-based.

    Returns
    -------
    sympy.Symbol
        Length of the axial vector of the antisymmetric part of the shielding
        tensor (dimensionless).
    """
    return smp.Symbol(f'\\sigma_{{a}}^{{({spin_index})}}', real=True)


def b_coupling_symbol(
    spin_index_1: int,
    spin_index_2: int,
) -> smp.Symbol:
    """
    Build the symbolic dipolar coupling constant between two spins.

    NOTE: The rigid-geometry constant that drives dipolar relaxation, as opposed
    to the motionally averaged residual coupling of `D_coupling_symbol`.

    Parameters
    ----------
    spin_index_1 : int
        Index of the first spin, 1-based.
    spin_index_2 : int
        Index of the second spin, 1-based.

    Returns
    -------
    sympy.Symbol
        ``b = mu_0 hbar y_1 y_2 / (4 pi r^3)``, as an angular frequency.
    """
    return smp.Symbol(f'b_{{{spin_index_1}{spin_index_2}}}', real=True)


def C_Q_symbol(spin_index: int) -> smp.Symbol:
    """
    Build the symbolic quadrupolar coupling constant of a spin.

    Parameters
    ----------
    spin_index : int
        Index of the spin, 1-based.

    Returns
    -------
    sympy.Symbol
        ``C_Q = e^2 q Q / h``, as an ordinary frequency (in Hz).
    """
    return smp.Symbol(f'C_{{Q}}^{{({spin_index})}}', real=True)


def Delta_J_symbol(
    spin_index_1: int,
    spin_index_2: int,
) -> smp.Symbol:
    """
    Build the symbolic J-coupling anisotropy between two spins.

    Parameters
    ----------
    spin_index_1 : int
        Index of the first spin, 1-based.
    spin_index_2 : int
        Index of the second spin, 1-based.

    Returns
    -------
    sympy.Symbol
        Anisotropy ``J_parallel - J_perpendicular`` of an axially symmetric
        J-coupling tensor (in Hz).
    """
    return smp.Symbol(f'\\Delta J_{{{spin_index_1}{spin_index_2}}}', real=True)


def J_a_symbol(
    spin_index_1: int,
    spin_index_2: int,
) -> smp.Symbol:
    """
    Build the symbolic antisymmetric J-coupling between two spins.

    Parameters
    ----------
    spin_index_1 : int
        Index of the first spin, 1-based.
    spin_index_2 : int
        Index of the second spin, 1-based.

    Returns
    -------
    sympy.Symbol
        Length of the axial vector of the antisymmetric part of the J-coupling
        tensor (in Hz).
    """
    return smp.Symbol(f'J^{{a}}_{{{spin_index_1}{spin_index_2}}}', real=True)


def amplitude_symbol(
    label: str,
    l: int=2,
) -> smp.Symbol:
    """
    Build the generic symbolic amplitude of an interaction of unrecognised kind.

    Parameters
    ----------
    label : str
        Interaction label, i.e. mechanism name followed by the spin indices.
    l : int, optional
        Rank of the interaction. Default is 2.

    Returns
    -------
    sympy.Symbol
        For ``l = 2`` the anisotropy ``Delta`` of an axially symmetric coupling
        tensor, and for ``l = 1`` the length of the axial vector of an
        antisymmetric one, both as angular frequencies.
    """
    if l == 2:
        return smp.Symbol(f'\\Delta_{{{label}}}', real=True)
    return smp.Symbol(f'a_{{{label}}}', real=True)


def theta_symbol(
    label_1: str,
    label_2: str,
    l: int=2,
) -> smp.Symbol:
    """
    Build the symbolic angle between the axes of two interactions.

    NOTE: The same symbol results whichever order the two labels are given in.

    Parameters
    ----------
    label_1 : str
        Label of the first interaction, e.g. ``'CSA1'``.
    label_2 : str
        Label of the second interaction, e.g. ``'DD12'``.
    l : int, optional
        Rank. For ``l = 2`` the angle is that between the symmetry axes of the
        coupling tensors, and for ``l = 1`` that between the axial vectors of
        their antisymmetric parts. Default is 2.

    Returns
    -------
    sympy.Symbol
        Angle between the two axes.
    """
    # Order the labels as in the names of the spectral density functions.
    names = sort_interactions(label_1, label_2)
    if not isinstance(names, str):
        names = ', '.join(names)
    rank = '' if l == 2 else f'^{{({l})}}'
    return smp.Symbol(f'\\theta{rank}_{{{names}}}', real=True)
