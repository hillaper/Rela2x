"""
Superoperators for Liouville space.

The module contains the vectorization of Hilbert-space operators together with
the left- and right-multiplication, commutation, and
double-commutation superoperators built from them, and the `Operator` and
`Superoperator` classes that wrap a matrix representation of either into a
physics object with its own symbols, functions and basis machinery.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import numpy as np
import sympy as smp
import time

from rela2x._core._basis import (
    S_index_from_string,
    T_index_from_string,
    basis_index_position,
    coherence_order_filter,
    spin_order_filter,
    type_filter,
)
from rela2x._core._la import Kronecker_product, op_change_of_basis
from rela2x._core._status import status
from rela2x._core._visualization import visualize_operator


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
def sop_right_mul(op: smp.MatrixBase) -> smp.MatrixBase:
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


def sop_left_mul(op: smp.MatrixBase) -> smp.MatrixBase:
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
    return sop_left_mul(op) - sop_right_mul(op)


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
    return sop_left_mul(op1) @ sop_right_mul(op2)\
           - smp.Rational(1, 2) * (sop_left_mul(op2 @ op1) + sop_right_mul(op2 @ op1))


class Operator:
    """
    General class for operators.

    Parameters
    ----------
    op : sympy.Matrix
        Matrix representation of the operator.

    Attributes
    ----------
    op : sympy.Matrix
        Matrix representation of the operator.
    symbols_in : list of sympy.Symbol
        Symbols appearing in the matrix representation of the operator.
    functions_in : list of sympy.Function
        Functions appearing in the matrix representation of the operator.

    Raises
    ------
    ValueError
        If `op` is not a SymPy matrix.
    """

    def __init__(
        self,
        op: smp.MatrixBase,
    ) -> None:
        """
        Initialise the operator and extract its symbols and functions.

        Parameters
        ----------
        op : sympy.Matrix
            Matrix representation of the operator.
        """

        # Check that the input is a SymPy matrix.
        # NOTE: MatrixBase covers both the mutable and the immutable dense matrix types.
        # SymPy operations such as simplify() return immutable matrices, so both occur in practice.
        if not isinstance(op, smp.MatrixBase):
            raise ValueError("The operator input for the Operator class has to be a SymPy matrix.")
        self.op = op
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

    # Matrix algebra
    def to_basis(
        self,
        basis: list[smp.MatrixBase],
    ) -> None:
        """
        Convert `self.op` to a different basis.

        Parameters
        ----------
        basis : list of sympy.Matrix
            New basis operators.
        """
        self.op = op_change_of_basis(self.op, basis)

    # Symbols, functions and substitutions
    def get_symbols(self) -> list[smp.Symbol]:
        """
        Get the symbols appearing in `self.op`.

        Returns
        -------
        list of sympy.Symbol
            Symbols appearing in the operator, sorted alphabetically.
        """
        return sorted(list(self.op.free_symbols), key=lambda x: str(x))

    def get_functions(self) -> list[smp.Function]:
        """
        Get the functions appearing in `self.op`.

        Returns
        -------
        list of sympy.Function
            Functions appearing in the operator, sorted alphabetically.
        """
        return sorted(list(self.op.atoms(smp.Function)), key=lambda x: str(x))

    def substitute(
        self,
        substitutions_dict: dict,
    ) -> None:
        """
        Substitute symbols and functions in `self.op` with numerical values.

        NOTE: Useful for converting a symbolic SymPy expression to a
        numerical NumPy array.

        Parameters
        ----------
        substitutions_dict : dict
            Substitutions of the form ``{symbol: value}``.
        """
        self.op = self.op.subs(substitutions_dict)

        # Update symbols_in and functions_in to reflect the substitution.
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

    # Visualization
    def visualize(
        self,
        rows_start: int=0,
        rows_end: int | None=None,
        basis_symbols: list[smp.Expr] | None=None,
        fontsize: int=8,
    ) -> None:
        """
        Visualize `self.op`. See `visualize_operator` for more information.

        Parameters
        ----------
        rows_start : int, optional
            Starting row/column index for the visualization. Default is 0.
        rows_end : int, optional
            Ending row/column index for the visualization. Default is None (last index).
        basis_symbols : list of sympy.Symbol, optional
            Basis operator symbols, drawn as a legend for the basis states. Default is None.
        fontsize : int, optional
            Font size for the basis symbol labels. Default is 8.
        """
        visualize_operator(self.op, rows_start=rows_start, rows_end=rows_end,
                           basis_symbols=basis_symbols, fontsize=fontsize)


class Superoperator(Operator):
    """
    General class for superoperators. Inherits from `Operator`.

    See `Operator` for more information.

    NOTE: The basis the superoperator is expressed in is carried alongside its
    matrix representation, so that matrix elements can be looked up and
    filtered by the operators they connect.

    Parameters
    ----------
    sop : sympy.Matrix
        Matrix representation of the superoperator.
    basis_symbols : list of sympy.Symbol, optional
        Basis operator symbols. Default is None.
    basis_norms : list of sympy.Expr, optional
        Liouville norms of the (unnormalized) basis operators. Default is None.
    basis_indices : list of tuple, optional
        Basis operator indices, used for matrix element lookups and filtering. Default is None.

    Attributes
    ----------
    basis_symbols : list of sympy.Symbol
        Basis operator symbols.
    basis_norms : list of sympy.Expr
        Liouville norms of the (unnormalized) basis operators.
    basis_indices : list of tuple
        Basis operator indices, used for matrix element lookups and filtering.
    """
    def __init__(
        self,
        sop: smp.MatrixBase,
        basis_symbols: list[smp.Expr] | None=None,
        basis_norms: list[smp.Expr] | None=None,
        basis_indices: list[tuple] | None=None,
    ) -> None:
        """
        Initialise the superoperator and extract its symbols and functions.

        Parameters
        ----------
        sop : sympy.Matrix
            Matrix representation of the superoperator.
        basis_symbols : list of sympy.Symbol, optional
            Basis operator symbols. Default is None.
        basis_norms : list of sympy.Expr, optional
            Liouville norms of the (unnormalized) basis operators. Default is None.
        basis_indices : list of tuple, optional
            Basis operator indices, used for matrix element lookups and filtering. Default is None.
        """
        super().__init__(sop)
        self.basis_symbols = basis_symbols
        self.basis_norms = basis_norms
        self.basis_indices = basis_indices

    @property
    def generator(self) -> smp.MatrixBase:
        """
        Generator of the equation of motion, defined by ``d/dt v = generator @ v``.

        NOTE: Subclasses whose sign convention differs from that of the
        generator override this property, so that every superoperator carries
        its own convention and the equations of motion can be assembled from
        any of them without further branching.

        Returns
        -------
        sympy.Matrix
            Matrix representation of the generator.
        """
        return self.op

    # Change of basis for superoperators
    def to_basis(
        self,
        basis: list[smp.MatrixBase],
    ) -> None:
        """
        Convert `self.op` to a different basis.

        Parameters
        ----------
        basis : list of sympy.Matrix
            New basis operators (in Hilbert space; vectorized internally
            before the Liouville-space change of basis).
        """
        basis_vectorized = vectorize_all(basis)

        # Record the start time for status reporting.
        time_start = time.time()
        status('Changing basis...')

        self.op = op_change_of_basis(self.op, basis_vectorized)

        status(f'Basis changed in {time.time() - time_start:.2f} seconds.\n')

    def to_observables(self) -> None:
        """
        Fix the basis operator normalization in `self.op`, in order to obtain
        matrix elements and equations of motion that correspond to observables.

        NOTE: The rescaling is the diagonal similarity transform
        ``diag(basis_norms) @ self.op @ diag(basis_norms)^-1``, and is
        therefore linear in `self.op`. Applying it separately to the coherent
        and incoherent parts of a Liouvillian and combining them afterwards is
        thus equivalent to applying it to their combination.

        NOTE: Not idempotent. Calling this method twice on the same object
        rescales `self.op` a second time, silently producing incorrect
        matrix elements and equations of motion.
        """
        op = self.op
        norms = self.basis_norms

        # Rescale every matrix element by the norms of its row and column basis operators.
        # NOTE: A new matrix is constructed, so that the method also accepts an immutable
        # self.op, as returned by, e.g., simplify().
        rescaled = smp.Matrix(op.shape[0], op.shape[1],
                              lambda i, j: op[i, j] * (norms[i] / norms[j]))

        # Simplify the superoperator.
        self.op = smp.simplify(rescaled)

    def element(
        self,
        spin_index_op_index_1: str,
        spin_index_op_index_2: str | None=None,
    ) -> smp.Expr | None:
        """
        Get the matrix element of the superoperator between two basis operators.

        NOTE: `RelaxationSuperoperator.rate` and `HamiltonianSuperoperator.frequency`
        are the physics-specific names for this same lookup, on *R* and *H*
        respectively; both delegate to this method and to its argument format.

        Parameters
        ----------
        spin_index_op_index_1 : str
            Spin index and operator index of the first basis operator. For
            spherical tensor operators: the spin index, then the spherical
            tensor rank, then the projection, as in ``'110'``. A negative
            projection carries its minus sign, as in ``'11-1'``. For
            Cartesian operators: a string of 2 characters, the first being
            the spin index and the second being the Cartesian direction
            (x, y or z). Product operators are separated by ``'*'``.
        spin_index_op_index_2 : str, optional
            Spin index and operator index of the second basis operator, in
            the same format as `spin_index_op_index_1`. If None, the diagonal
            element of `spin_index_op_index_1` is returned. Default is None.

        Returns
        -------
        sympy.Expr or None
            Matrix element between the two operators, or None if either
            operator is not found in the basis.
        """
        # Interpret the specifications as Cartesian or as spherical tensor operators.
        if spin_index_op_index_1[-1] in 'xyz':
            to_index = S_index_from_string
        else:
            to_index = T_index_from_string

        # Locate the requested operators in the basis.
        index_1 = basis_index_position(self.basis_indices, to_index(spin_index_op_index_1))
        if spin_index_op_index_2 is None:
            index_2 = index_1
        else:
            index_2 = basis_index_position(self.basis_indices,
                                             to_index(spin_index_op_index_2))

        # Return nothing if either operator was not found in the basis.
        # NOTE: The lookup function above reports the failure and returns None.
        if index_1 is None or index_2 is None:
            return None

        return self.op[index_1, index_2]

    def filter(
        self,
        filter_name: str,
        filter_value: int | list[int],
    ) -> None:
        """
        Filter the superoperator and its basis down to a physically relevant subspace.

        NOTE: Works only for the irreducible spherical tensor basis, since spin
        order, coherence order and type are properties of the (spin index, l, q)
        triples making up that basis, with no counterpart in the Cartesian basis.

        NOTE: In-place and irreversible. The rows/columns of `self.op` whose
        basis operator index fails the criterion are discarded, together with
        the corresponding entries of `self.basis_symbols`, `self.basis_norms`
        and `self.basis_indices`, so the four stay in step. There is no way to
        recover the discarded rows/columns afterwards; recompute the
        superoperator if a different filter is needed.

        Three criteria are available, each read directly off a basis operator's
        ``(spin index, l, q)`` index (see `T_index_spin_order`,
        `T_index_coherence_order` and `T_index_type`):

        - ``'c'``, coherence order: the sum of the projections q of the basis
          operator's single-spin factors, ``p = sum_i q_i``. The standard NMR
          quantum number: p = 0 for the longitudinal/zero-quantum subspace,
          p = +-1 for the observable single-quantum coherences, p = +-2 for
          double-quantum coherences, and so on.
        - ``'s'``, spin order: the number of spins carrying a non-identity
          operator in the product, i.e. how many-body the term is (1 for a
          single-spin operator, 2 for a bilinear one, etc.).
        - ``'t'``, type: population (0) if every single-spin factor carries
          q = 0, coherence (1) otherwise. NOTE: This is not the same split as
          coherence order 0 vs. non-zero. A zero-quantum coherence such as
          ``T_{1,1}^(1) T_{1,-1}^(2)`` has coherence order 0 but is a
          coherence (type 1), because its individual spins carry non-zero
          projections that happen to cancel in the sum.

        Parameters
        ----------
        filter_name : {'c', 's', 't'}
            Filter criterion: ``'c'`` for coherence order, ``'s'`` for spin
            order, ``'t'`` for type.
        filter_value : int or list of int
            Values of the criterion to retain. ``'c'`` and ``'s'`` take a
            list of integers, so several orders can be kept at once (e.g.
            ``R.filter('c', [0])`` keeps only the p = 0 subspace). ``'t'``
            takes a single integer, 0 or 1, not a list.
        """
        if filter_name == 'c':
            self.op, self.basis_symbols, self.basis_norms, self.basis_indices = coherence_order_filter(
                self.op, self.basis_symbols, self.basis_norms, self.basis_indices, filter_value)
        elif filter_name == 's':
            self.op, self.basis_symbols, self.basis_norms, self.basis_indices = spin_order_filter(
                self.op, self.basis_symbols, self.basis_norms, self.basis_indices, filter_value)
        elif filter_name == 't':
            self.op, self.basis_symbols, self.basis_norms, self.basis_indices = type_filter(
                self.op, self.basis_symbols, self.basis_norms, self.basis_indices, filter_value)

        # Update symbols_in and functions_in to reflect the filtering.
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()
