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
