"""
Symbolic representations of spin operators and expectation values.

The symbols generated here carry no matrix representation; they are the printed
labels attached to basis operators and to the observables appearing in the
equations of motion.
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
    return smp.Symbol('\\langle ' + str(op_symbol) + '\\rangle')


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
