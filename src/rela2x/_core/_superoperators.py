"""
Superoperators for Liouville space.

The module contains the vectorization of Hilbert-space operators together with
the left- and right-multiplication, commutation, double-commutation and
double-commutation superoperators built from them.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import numpy as np
import sympy as smp

from rela2x._core._la import Kronecker_product


# Vectorizations:
def vectorize(op: smp.MatrixBase) -> smp.Matrix:
    """
    Vectorize a matrix into a Liouville-space (super)vector.

    NOTE: Uses the row-wise vectorization convention, consistent with the
    rest of the superoperator definitions in this module.

    Parameters
    ----------
    op : sympy.Matrix
        Matrix (Hilbert-space operator) to be vectorized.

    Returns
    -------
    sympy.Matrix
        Column vector containing the row-flattened elements of `op`.
    """
    return smp.Matrix(np.array(op).flatten())


def vectorize_all(ops: list[smp.MatrixBase]) -> list[smp.Matrix]:
    """
    Vectorize a list of matrices into Liouville-space (super)vectors.

    Parameters
    ----------
    ops : list of sympy.Matrix
        Matrices (Hilbert-space operators) to be vectorized.

    Returns
    -------
    list of sympy.Matrix
        Vectorized form of each operator in `ops`.
    """
    return [vectorize(op) for op in ops]


# Liouville-space matrix representations:
def sop_rmul(op: smp.MatrixBase) -> smp.MatrixBase:
    """
    Build the right-multiplication superoperator of a Hilbert-space operator.

    Parameters
    ----------
    op : sympy.Matrix
        Hilbert-space operator.

    Returns
    -------
    sympy.Matrix
        Right-multiplication superoperator corresponding to `op`.
    """
    return Kronecker_product(smp.eye(op.shape[0]), op.T)


def sop_lmul(op: smp.MatrixBase) -> smp.MatrixBase:
    """
    Build the left-multiplication superoperator of a Hilbert-space operator.

    Parameters
    ----------
    op : sympy.Matrix
        Hilbert-space operator.

    Returns
    -------
    sympy.Matrix
        Left-multiplication superoperator corresponding to `op`.
    """
    return Kronecker_product(op, smp.eye(op.shape[0]))


def sop_commutator(op: smp.MatrixBase) -> smp.MatrixBase:
    """
    Build the commutation superoperator of a Hilbert-space operator.

    Parameters
    ----------
    op : sympy.Matrix
        Hilbert-space operator.

    Returns
    -------
    sympy.Matrix
        Commutation superoperator corresponding to `op`.
    """
    return sop_lmul(op) - sop_rmul(op)


def sop_double_commutator(
    op1: smp.MatrixBase,
    op2: smp.MatrixBase,
) -> smp.MatrixBase:
    """
    Build the double-commutation superoperator of two Hilbert-space operators.

    Parameters
    ----------
    op1 : sympy.Matrix
        First Hilbert-space operator.
    op2 : sympy.Matrix
        Second Hilbert-space operator.

    Returns
    -------
    sympy.Matrix
        Double-commutation superoperator, i.e. the product of the
        commutation superoperators of `op1` and `op2`.
    """
    return sop_commutator(op1) @ sop_commutator(op2)


def sop_D(
    op1: smp.MatrixBase,
    op2: smp.MatrixBase,
) -> smp.MatrixBase:
    """
    Build the Lindbladian dissipation superoperator of two Hilbert-space operators.

    Parameters
    ----------
    op1 : sympy.Matrix
        First Hilbert-space operator.
    op2 : sympy.Matrix
        Second Hilbert-space operator.

    Returns
    -------
    sympy.Matrix
        Lindbladian dissipation superoperator corresponding to `op1` and `op2`.
    """
    return sop_lmul(op1) @ sop_rmul(op2)\
           - smp.Rational(1, 2) * (sop_lmul(op2 @ op1) + sop_rmul(op2 @ op1))
