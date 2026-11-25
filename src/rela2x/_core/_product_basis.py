"""
Superoperators of product operators built directly in a product operator basis.

A multiplication superoperator of a product operator factorises into the
single-spin multiplication matrices of its factors, which are the structure
coefficients of the single-spin basis (see https://doi.org/10.1063/1.3398146).
The superoperators are therefore obtained without Kronecker products of the
Zeeman-basis matrices and without a change of basis. Matrices are stored
sparsely as columns, ``{column: {row: value}}``, with exact SymPy values.

NOTE: The matrix elements refer to observables, i.e. they equal those of
`Superoperator.to_basis` followed by `Superoperator.to_observables`.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

from functools import lru_cache
from itertools import product

import sympy as smp

from rela2x._core._operators import op_Sx, op_Sy, op_Sz, op_T


def _nonzero_elements(op: smp.MatrixBase) -> dict:
    """
    Nonzero elements of a matrix.

    Parameters
    ----------
    op : sympy.Matrix
        Matrix.

    Returns
    -------
    dict
        Elements as ``{(row, column): value}``.
    """
    return {(i, j): op[i, j] for i in range(op.shape[0]) for j in range(op.shape[1]) if op[i, j] != 0}


@lru_cache(maxsize=None)
def _single_spin_basis(
    S: float,
    basis: str,
) -> tuple:
    """
    Basis operators of a single spin and their squared Liouville norms.

    Parameters
    ----------
    S : float
        Spin quantum number.
    basis : {'T', 'C'}
        Spherical tensor operators ordered by l and then q, or the Cartesian
        operators E, Sx, Sy and Sz.

    Returns
    -------
    tuple
        One pair (nonzero elements, squared norm) per basis operator.
    """
    if basis == 'T':
        ops = [op_T(S, l, q) for l in range(int(2*S) + 1) for q in range(-l, l + 1)]
    else:
        ops = [smp.eye(int(2*S + 1)), op_Sx(S), op_Sy(S), op_Sz(S)]

    return tuple((_nonzero_elements(op), smp.expand((op.H @ op).trace())) for op in ops)


@lru_cache(maxsize=None)
def _single_spin_multiplication(
    S: float,
    basis: str,
    l: int,
    q: int,
    conjugate: bool,
    side: str,
) -> dict:
    """
    Matrix of left or right multiplication by a spherical tensor operator of a
    single spin, in the single-spin basis normalised to observables:
    element (a, b) is Tr(B_a^dagger A B_b) / ||B_b||^2.

    Parameters
    ----------
    S : float
        Spin quantum number.
    basis : {'T', 'C'}
        Single-spin basis.
    l : int
        Rank of the operator A.
    q : int
        Projection of the operator A.
    conjugate : bool
        Whether A is the Hermitian conjugate of the spherical tensor operator.
    side : {'left', 'right'}
        Side of the multiplication.

    Returns
    -------
    dict
        Columns as ``{b: [(a, value), ...]}``.
    """
    # Take the multiplying operator from the spherical tensor basis, which is built only once per spin
    A = _single_spin_basis(S, 'T')[l*(l + 1) + q][0]
    if conjugate:
        A = {(j, i): smp.conjugate(x) for (i, j), x in A.items()}

    states = _single_spin_basis(S, basis)
    columns = {}
    for b, (B_b, norm_b) in enumerate(states):

        # Product of the operator with the basis operator
        first, second = (A, B_b) if side == 'left' else (B_b, A)
        image = {}
        for (i, k1), x in first.items():
            for (k2, j), y in second.items():
                if k1 == k2:
                    image[i, j] = image.get((i, j), 0) + x*y

        # Project the product onto every basis operator
        columns[b] = []
        for a, (B_a, _) in enumerate(states):
            overlap = sum(smp.conjugate(x) * image[key] for key, x in B_a.items() if key in image)
            value = smp.expand(overlap / norm_b)
            if value != 0:
                columns[b].append((a, value))

    return columns


class ProductBasis:
    """
    Multiplication superoperators of product operators in a product operator basis.

    A product operator is given as a tuple of factors (spin, l, q, conjugate),
    one spherical tensor operator on each of the spins it acts on, and the
    identity on all other spins.

    Parameters
    ----------
    S : list of float
        Spin quantum numbers.
    basis : {'T', 'C'}
        Product operator basis.
    basis_indices : list of tuple
        Basis operator indices, in the order of the basis.
    """
    def __init__(
        self,
        S: list[float],
        basis: str,
        basis_indices: list[tuple],
    ) -> None:
        """
        Initialise the product basis.

        Parameters
        ----------
        S : list of float
            Spin quantum numbers.
        basis : {'T', 'C'}
            Product operator basis.
        basis_indices : list of tuple
            Basis operator indices, in the order of the basis.
        """
        self.S = S
        self.basis = basis

        # Single-spin basis index of every spin for every basis operator, zero standing for the identity
        self.states = []
        for index in basis_indices:
            state = [0] * len(S)
            for factor in index:
                if basis == 'T':
                    spin, l, q = factor
                    state[spin - 1] = l*(l + 1) + q
                else:
                    spin, direction = factor
                    state[spin - 1] = 'Exyz'.index(direction)
            self.states.append(tuple(state))

        # Position of every combination of single-spin indices in the basis
        self.position = {state: i for i, state in enumerate(self.states)}

        # Multiplication superoperators built so far
        self._cache = {}

    @property
    def dim(self) -> int:
        """
        Dimension of the basis.

        Returns
        -------
        int
            Number of basis operators.
        """
        return len(self.states)

    def multiplication(
        self,
        operator: tuple,
        side: str,
    ) -> dict:
        """
        Superoperator of left or right multiplication by a product operator.

        Parameters
        ----------
        operator : tuple
            Factors (spin, l, q, conjugate) on distinct spins, spins indexed from zero.
        side : {'left', 'right'}
            Side of the multiplication.

        Returns
        -------
        dict
            Columns of the superoperator as ``{column: {row: value}}``.
        """
        key = (operator, side)
        if key in self._cache:
            return self._cache[key]

        # Single-spin multiplication matrices of the factors
        spins = [factor[0] for factor in operator]
        matrices = [_single_spin_multiplication(self.S[spin], self.basis, l, q, conjugate, side)
                    for spin, l, q, conjugate in operator]

        # Combine the single-spin elements for every basis operator, the other spins being unchanged
        columns = {}
        for j, state in enumerate(self.states):
            column = {}
            for entries in product(*[matrix[state[spin]] for spin, matrix in zip(spins, matrices)]):
                row = list(state)
                value = smp.S.One
                for spin, (a, element) in zip(spins, entries):
                    row[spin] = a
                    value *= element
                column[self.position[tuple(row)]] = value
            if column:
                columns[j] = column

        self._cache[key] = columns
        return columns


def sparse_matmul(
    A: dict,
    B: dict,
) -> dict:
    """
    Product of two sparse matrices stored as columns.

    Parameters
    ----------
    A : dict
        Left matrix.
    B : dict
        Right matrix.

    Returns
    -------
    dict
        Columns of A @ B.
    """
    result = {}
    for j, column in B.items():
        target = {}
        for k, b in column.items():
            for i, a in A.get(k, {}).items():
                target[i] = target.get(i, 0) + a*b
        if target:
            result[j] = target
    return result


def sparse_add(
    A: dict,
    B: dict,
    factor: smp.Expr,
) -> dict:
    """
    Sum A + factor*B of two sparse matrices stored as columns.

    Parameters
    ----------
    A : dict
        First matrix.
    B : dict
        Second matrix.
    factor : sympy.Expr
        Factor of the second matrix.

    Returns
    -------
    dict
        Columns of the sum.
    """
    result = {j: dict(column) for j, column in A.items()}
    for j, column in B.items():
        target = result.setdefault(j, {})
        for i, value in column.items():
            target[i] = target.get(i, 0) + factor*value
    return result
