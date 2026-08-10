"""
Linear-algebra utilities used throughout Rela2x.

The module collects the Kronecker product, commutators, and the Liouville-space
inner product, norm and amplitude, together with the routines for changing the
basis of an operator and decomposing it in a given basis.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp


def Kronecker_product(*m: smp.MatrixBase) -> smp.MatrixBase:
    """
    Compute the symbolic Kronecker product of multiple matrices.

    Parameters
    ----------
    *m : sympy.Matrix
        Matrices to be combined via the Kronecker product, in order.

    Returns
    -------
    result : sympy.Matrix
        Kronecker product of the input matrices.
    """

    # Fold the Kronecker product over the matrices from left to right.
    result = m[0]
    for i in range(1, len(m)):
        result = smp.Matrix(result.shape[0]*m[i].shape[0], result.shape[1]*m[i].shape[1],\
                 lambda p, q: result[p//m[i].shape[0], q//m[i].shape[1]] * m[i][p%m[i].shape[0], q%m[i].shape[1]])
    return result


def commutator(
    op1: smp.MatrixBase,
    op2: smp.MatrixBase,
) -> smp.MatrixBase:
    """
    Compute the symbolic commutator [op1, op2] = op1 * op2 - op2 * op1.

    Parameters
    ----------
    op1 : sympy.Matrix
        First operator.
    op2 : sympy.Matrix
        Second operator.

    Returns
    -------
    sympy.Matrix
        Commutator of the two operators.
    """
    return op1 * op2 - op2 * op1


# Liouville bracket, norm and amplitude
def Lv_bracket(
    op1: smp.MatrixBase,
    op2: smp.MatrixBase,
) -> smp.Expr:
    """
    Compute the symbolic Hilbert-Schmidt (Liouville) inner product of two operators.

    Parameters
    ----------
    op1 : sympy.Matrix
        First operator.
    op2 : sympy.Matrix
        Second operator.

    Returns
    -------
    sympy.Expr
        Liouville bracket ``Tr(op1^dagger * op2)`` of the two operators.
    """
    return smp.trace(op1.H * op2)


def Lv_norm(op: smp.MatrixBase) -> smp.Expr:
    """
    Compute the symbolic Liouville (Hilbert-Schmidt) norm of an operator.

    Parameters
    ----------
    op : sympy.Matrix
        Operator whose norm is to be computed.

    Returns
    -------
    sympy.Expr
        Liouville norm of `op`.
    """
    return smp.sqrt(Lv_bracket(op, op))


def Lv_amplitude(
    op1: smp.MatrixBase,
    op2: smp.MatrixBase,
) -> smp.Expr:
    """
    Compute the symbolic Liouville amplitude of `op1` contained in `op2`.
    Note that the order of the operators matters.

    Parameters
    ----------
    op1 : sympy.Matrix
        Reference operator whose amplitude in `op2` is sought.
    op2 : sympy.Matrix
        Operator to be projected onto `op1`.

    Returns
    -------
    sympy.Expr
        Amplitude of `op1` contained in `op2`, i.e. ``Lv_bracket(op1, op2) / Lv_bracket(op1, op1)``.
    """
    return Lv_bracket(op1, op2) / Lv_bracket(op1, op1)


def op_change_of_basis(
    op: smp.MatrixBase,
    basis: list[smp.MatrixBase],
) -> smp.Matrix:
    """
    Perform a symbolic change of basis of an operator.

    Parameters
    ----------
    op : sympy.Matrix
        Operator to be transformed, given in its original matrix representation.
    basis : list of sympy.Matrix
        New basis set, given as a list of matrix representations of the basis states/operators.

    Returns
    -------
    op_new : sympy.Matrix
        Operator expressed in the new basis.
    """
    op_new = smp.zeros(op.shape[0], op.shape[1], complex=True)

    # Project the operator onto every pair of new basis states.
    for i in range(op.shape[0]):
        for j in range(op.shape[1]):
            op_new[i, j] = basis[i].H * op * basis[j]
            op_new[i, j] = smp.expand(op_new[i, j])
    return op_new


# Function to compute the amplitude of each basis state/operator in a given operator
# and return a symbolic expression for the operator in terms of the basis states/operators.
def op_decomposition(
    op: smp.MatrixBase,
    basis: list[smp.MatrixBase],
    basis_symbols: list[smp.Expr],
) -> smp.Expr:
    """
    Perform a symbolic decomposition of an operator in terms of a basis set.

    Parameters
    ----------
    op : sympy.Matrix
        Operator to be decomposed, given in its matrix representation.
    basis : list of sympy.Matrix
        Basis set, given as a list of matrix representations of the basis states/operators.
    basis_symbols : list of sympy.Symbol
        Symbols corresponding to the basis states/operators in `basis`.

    Returns
    -------
    op_decomposed : sympy.Expr
        Symbolic expression for `op` in terms of the basis states/operators.
    """
    op_decomposed = 0

    # Accumulate the contribution of every basis state with a non-zero amplitude.
    for i in range(len(basis)):
        amplitude = Lv_amplitude(basis[i], op)
        amplitude = smp.nsimplify(amplitude)
        if amplitude != 0:
            op_decomposed += amplitude * basis_symbols[i]
    return smp.simplify(op_decomposed)
