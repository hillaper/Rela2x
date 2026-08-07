"""
The main module for the Rela²x package.

Author:
    Perttu Hilla, 2024-2026
    perttu.hilla@oulu.fi (or perttuhilla@gmail.com)
    NMR Research Unit, University of Oulu.
"""

####################################################################################################
# Imports.
####################################################################################################
# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be used in type
# hints while the package keeps supporting the Python versions declared in pyproject.toml.
from __future__ import annotations

# General
import re
import hashlib
import itertools

import numpy as np

import matplotlib.pyplot as plt

import sympy as smp
import sympy.physics.quantum as smpq
from sympy.physics.quantum.cg import CG

# Rela2x specific
from rela2x import settings
from rela2x import nmr_isotopes

# NOTE: The remaining symbolic constants and variables are re-exported to the user by __init__.py.
from rela2x.constants_and_variables import beta, t, tau_c

# NOTE: The __future__ import above binds a module-level name of its own, which "from rela2x import *"
# would otherwise pick up. Removing it keeps the public namespace to the Rela2x interface itself.
# The postponed evaluation of annotations is a compile-time property and is unaffected.
del annotations

####################################################################################################
# Settings and modes of the program.
####################################################################################################
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
    settings.RELAXATION_THEORY = theory

####################################################################################################
# Mathematical tools.
# NOTE: General tools defined here, more specific functionalities in classes defined later.
####################################################################################################
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

####################################################################################################
# Miscellaneous tools.
# NOTE: General tools defined here, more specific functionalities in classes.
####################################################################################################
# Information extraction from input of NMR isotopes
def spin_quantum_numbers(isotopes: list[str]) -> list[float]:
    """
    Look up the spin quantum numbers of a list of nuclear isotopes.

    Parameters
    ----------
    isotopes : list of str
        Nuclear isotopes, given as their string labels in `nmr_isotopes.ISOTOPES`.

    Returns
    -------
    S : list of float
        Spin quantum numbers, in the same order as `isotopes`.

    Raises
    ------
    ValueError
        If an isotope in `isotopes` is not found in `nmr_isotopes.ISOTOPES`.
    """
    try:
        return [nmr_isotopes.ISOTOPES[isotope][0] for isotope in isotopes]
    except KeyError:
        raise ValueError("Given NMR isotope not found in the nmr_isotopes.py file.")

# Information extraction from product basis indices
# NOTE: Each basis operator is described by a tuple of (spin index, l, q) triples, one per spin
# carrying a non-identity operator, ordered by spin index. The identity operator of the whole
# system is described by an empty tuple. Spin indices are 1-based, matching the operator symbols.
# The Cartesian basis is described analogously by (spin index, direction) pairs.
def T_index_spin_order(T_index: tuple) -> int:
    """
    Determine the spin order of a basis operator, i.e. the number of single-spin
    operators in the product.

    Parameters
    ----------
    T_index : tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.

    Returns
    -------
    int
        Spin order of the basis operator.
    """
    return len(T_index)

def T_index_coherence_order(T_index: tuple) -> int:
    """
    Determine the coherence order of a basis operator, i.e. the sum of the
    projections of its single-spin operators.

    Parameters
    ----------
    T_index : tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.

    Returns
    -------
    int
        Coherence order of the basis operator.
    """
    return sum(q for _, _, q in T_index)

def T_index_type(T_index: tuple) -> int:
    """
    Determine the type of a basis operator.

    An operator is a coherence if any of its single-spin operators carries a
    non-zero projection, and a population otherwise.

    Parameters
    ----------
    T_index : tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.

    Returns
    -------
    int
        Type of the basis operator: 0 for a population, 1 for a coherence.
    """
    return 1 if any(q != 0 for _, _, q in T_index) else 0

def T_index_spin_projection(T_index: tuple, spin_index: int) -> int:
    """
    Extract the projection q carried by a given spin in a basis operator.

    Parameters
    ----------
    T_index : tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.
    spin_index : int
        Index (1-based) of the spin whose projection is sought.

    Returns
    -------
    int
        Projection q of the requested spin, or 0 if that spin carries the identity.
    """
    for index, _, q in T_index:
        if index == spin_index:
            return q

    return 0

def T_index_from_string(spin_index_lqs: str) -> tuple:
    """
    Convert a spherical tensor operator specification into a basis operator index.

    Parameters
    ----------
    spin_index_lqs : str
        String of the form ``'210'`` for spin 2, rank l = 1, projection q = 0,
        or ``'110*210'`` for a product of two such operators. A negative
        projection carries its minus sign, as in ``'11-1'``.

    Returns
    -------
    tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.
    """
    factors = []

    # Each factor is written as the spin index, then the rank, then the projection.
    for factor in spin_index_lqs.split('*'):
        factors.append((int(factor[0]), int(factor[1]), int(factor[2:])))

    return tuple(factors)

def S_index_from_string(spin_index_directions: str) -> tuple:
    """
    Convert a Cartesian spin operator specification into a basis operator index.

    Parameters
    ----------
    spin_index_directions : str
        String of the form ``'1x'`` for spin 1 in the x-direction, or ``'1z*2z'``
        for a product of two such operators.

    Returns
    -------
    tuple
        Basis operator index, as a tuple of (spin index, direction) pairs.
    """
    factors = []

    # Each factor is written as the spin index followed by the Cartesian direction.
    for factor in spin_index_directions.split('*'):
        factors.append((int(factor[0]), factor[1]))

    return tuple(factors)

def basis_index_list_index(basis_indices: list[tuple], target_index: tuple) -> int | None:
    """
    Find the position of a basis operator within a list of basis operator indices.

    Parameters
    ----------
    basis_indices : list of tuple
        Basis operator indices to search.
    target_index : tuple
        Basis operator index to look for.

    Returns
    -------
    int or None
        Position of `target_index` in `basis_indices`, or None if it is absent.
    """
    if target_index in basis_indices:
        return basis_indices.index(target_index)

    print(f'No match found from the basis operators for {target_index}.')
    return None

# String hashing (for sorting purposes)
def string_to_number(string: str) -> int:
    """
    Hash a string to an integer for use as a deterministic sort key.

    Parameters
    ----------
    string : str
        String to be hashed.

    Returns
    -------
    int
        SHA-256 hash of `string`, interpreted as an integer.
    """
    return int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16)

# Convenience function
def sort_interactions(
    intr1: str,
    intr2: str,
) -> str | list[str]:
    """
    Sort a pair of interaction names into a canonical order.

    NOTE: Used purely for cosmetic purposes, so that a given pair of
    interactions is always labelled consistently regardless of call order.

    Parameters
    ----------
    intr1 : str
        First interaction name.
    intr2 : str
        Second interaction name.

    Returns
    -------
    str or list of str
        `intr1` if the two interaction names are identical, otherwise the
        pair sorted into a deterministic order via a string hash.
    """
    if str(intr1) == str(intr2):
        return str(intr1)
    else:
        # Hash the strings and return the sorted pair.
        return sorted([str(intr1), str(intr2)], key=string_to_number)

# List and matrix operations
def pick_from_list(
    lst: list,
    kept_indices: list[int],
) -> list:
    """
    Select a subset of elements from a list.

    Parameters
    ----------
    lst : list
        List to select from.
    kept_indices : list of int
        Indices of the elements to keep.

    Returns
    -------
    list
        Elements of `lst` at `kept_indices`.
    """
    return [lst[i] for i in kept_indices]

def pick_from_matrix(
    matrix: smp.MatrixBase,
    kept_indices: list[int],
) -> smp.MatrixBase:
    """
    Select a subset of rows and columns from a matrix.

    Parameters
    ----------
    matrix : sympy.Matrix
        Matrix to select from.
    kept_indices : list of int
        Row and column indices to keep.

    Returns
    -------
    sympy.Matrix
        Submatrix of `matrix` restricted to `kept_indices` in both dimensions.
    """
    return matrix[kept_indices, :][:, kept_indices]

def cut_list(
    lst: list,
    removed_indices: list[int],
) -> list:
    """
    Remove a subset of elements from a list.

    Parameters
    ----------
    lst : list
        List to cut down.
    removed_indices : list of int
        Indices of the elements to remove.

    Returns
    -------
    list
        `lst` with the elements at `removed_indices` removed.
    """
    return [item for i, item in enumerate(lst) if i not in removed_indices]

def cut_matrix(
    matrix: smp.MatrixBase,
    removed_indices: list[int],
) -> smp.MatrixBase:
    """
    Remove a subset of rows and columns from a matrix.

    Parameters
    ----------
    matrix : sympy.Matrix
        Matrix to cut down.
    removed_indices : list of int
        Row and column indices to remove.

    Returns
    -------
    sympy.Matrix
        `matrix` with the rows and columns at `removed_indices` removed.
    """
    # Remove rows.
    rows_to_keep = [i for i in range(matrix.rows) if i not in removed_indices]
    matrix_loc = matrix[rows_to_keep, :]

    # Remove columns.
    cols_to_keep = [i for i in range(matrix.cols) if i not in removed_indices]
    matrix_loc = matrix_loc[:, cols_to_keep]

    return matrix_loc

# Filter functions based on allowed coherences, spin orders and types
# NOTE: General input and return structure defined in coherence_order_filter.
def coherence_order_filter(
    operator: smp.MatrixBase,
    basis_state_symbols: list[smp.Expr],
    basis_indices: list[tuple],
    allowed_coherences: list[int],
) -> tuple[smp.MatrixBase, list[smp.Expr], list[tuple]]:
    """
    Filter an operator and its basis to a set of allowed coherence orders.

    Parameters
    ----------
    operator : sympy.Matrix
        Operator (matrix representation) to be filtered.
    basis_state_symbols : list of sympy.Expr
        Basis state symbols corresponding to the rows/columns of `operator`.
    basis_indices : list of tuple
        Basis operator indices corresponding to the rows/columns of `operator`.
    allowed_coherences : list of int
        Coherence orders to retain.

    Returns
    -------
    operator : sympy.Matrix
        Filtered operator.
    basis_state_symbols : list of sympy.Expr
        Filtered basis state symbols.
    basis_indices : list of tuple
        Filtered basis operator indices.
    """
    coherences = [T_index_coherence_order(index) for index in basis_indices]
    indexes_to_delete = [i for i, coherence in enumerate(coherences)
                         if coherence not in allowed_coherences]

    return (cut_matrix(operator, indexes_to_delete),
            cut_list(basis_state_symbols, indexes_to_delete),
            cut_list(basis_indices, indexes_to_delete))

def spin_order_filter(
    operator: smp.MatrixBase,
    basis_state_symbols: list[smp.Expr],
    basis_indices: list[tuple],
    allowed_spin_orders: list[int],
) -> tuple[smp.MatrixBase, list[smp.Expr], list[tuple]]:
    """
    Filter an operator and its basis to a set of allowed spin orders.

    See `coherence_order_filter` for the parameter and return structure.
    """
    spin_orders = [T_index_spin_order(index) for index in basis_indices]
    indexes_to_delete = [i for i, spin_order in enumerate(spin_orders)
                         if spin_order not in allowed_spin_orders]

    return (cut_matrix(operator, indexes_to_delete),
            cut_list(basis_state_symbols, indexes_to_delete),
            cut_list(basis_indices, indexes_to_delete))

def type_filter(
    operator: smp.MatrixBase,
    basis_state_symbols: list[smp.Expr],
    basis_indices: list[tuple],
    allowed_type: int,
) -> tuple[smp.MatrixBase, list[smp.Expr], list[tuple]]:
    """
    Filter an operator and its basis to a single allowed type.

    `allowed_type` is a single integer: 0 for populations, 1 for coherences.
    See `coherence_order_filter` for the parameter and return structure.
    """
    basis_types = [T_index_type(index) for index in basis_indices]
    indexes_to_delete = [i for i, basis_type in enumerate(basis_types)
                         if basis_type != allowed_type]

    return (cut_matrix(operator, indexes_to_delete),
            cut_list(basis_state_symbols, indexes_to_delete),
            cut_list(basis_indices, indexes_to_delete))

# List operations
def list_indexes(lst: list) -> list[int]:
    """
    Return the valid indices of a list.

    Parameters
    ----------
    lst : list
        List whose indices are sought.

    Returns
    -------
    list of int
        Indices ``0, 1, ..., len(lst) - 1``.
    """
    return list(range(len(lst)))

# Combinatorics
def all_combinations(
    N: int,
    *args: list,
    reverse: bool=False,
) -> list[tuple]:
    """
    Generate all combinations of N lists.

    NOTE: Used for generating the direct product basis of spherical tensor
    (or Cartesian) operators from the single-spin operator lists.

    Parameters
    ----------
    N : int
        Number of lists to combine (i.e. the number of spins).
    *args : list
        Lists to be combined.
    reverse : bool, optional
        Whether to reverse the order of the generated combinations, which is
        convenient for basis sorting. Default is False.

    Returns
    -------
    list of tuple
        All combinations, each containing one element from each of the N
        selected lists.
    """
    list_combinations = list(itertools.combinations(args, N))

    # For each combination of lists, generate all combinations with one element from each list
    all_combinations = []
    for lists in list_combinations:
        all_combinations.extend(itertools.product(*lists))

    # Reverse the order of the combinations (convenient for basis sorting)
    if reverse:
        all_combinations = list(reversed(all_combinations))

    return all_combinations

####################################################################################################
# Visualization tools.
# NOTE: Requesting an empty section, i.e. rows_start equal to rows_end, raises from inside NumPy.
####################################################################################################
def matrix_nonzeros(matrix: smp.MatrixBase) -> smp.MatrixBase:
    """
    Build a binary mask of the nonzero elements of a matrix.

    Parameters
    ----------
    matrix : sympy.Matrix
        Matrix to inspect.

    Returns
    -------
    sympy.Matrix
        Matrix of the same shape as `matrix`, with 1 at nonzero elements and 0 elsewhere.
    """
    return matrix.applyfunc(lambda x: 1 if x != 0 else 0)

def _plot_nonzero_pattern(
    operator_nonzeros: np.ndarray,
    rows_start: int=0,
    rows_end: int | None=None,
    basis_symbols: list[smp.Expr] | None=None,
    fontsize: int=8,
) -> None:
    """
    Draw the nonzero-element pattern of an operator as a matrix plot.

    NOTE: Internal helper shared by `visualize_operator` and `visualize_many_operators`,
    which differ only in how the array of nonzero counts is assembled.

    Parameters
    ----------
    operator_nonzeros : numpy.ndarray
        Array of nonzero counts, already restricted to the rows and columns to be drawn.
    rows_start : int, optional
        Starting row/column index of the visualized section, used to align `basis_symbols`
        with the drawn rows. Default is 0.
    rows_end : int, optional
        Ending row/column index of the visualized section. Default is None (last index).
    basis_symbols : list of sympy.Symbol, optional
        Basis operator symbols, drawn as a legend for the basis states. Default is None.
    fontsize : int, optional
        Font size for the basis symbol labels. Default is 8.
    """

    # Increase font size for small matrices
    if operator_nonzeros.shape[0] < 16:
        fontsize += 2

    # Create the plot with a suitable size
    if operator_nonzeros.shape[0] <= 16:
        _, ax = plt.subplots(figsize=(4, 4), dpi=150)
    else:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Create color map for the nonzeros
    norm = plt.Normalize(0, np.amax(operator_nonzeros))
    cmap = plt.cm.Blues

    # Plot the nonzeros
    ax.imshow(operator_nonzeros, cmap=cmap, alpha=0.9, norm=norm)

    # Shift the grid
    ax.set_xticks(np.arange(-.5, operator_nonzeros.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, operator_nonzeros.shape[0], 1), minor=True)
    if operator_nonzeros.shape[0] <= 64:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    elif operator_nonzeros.shape[0] <= 128:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    else:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2)

    # Move x-axis ticks to the top
    ax.xaxis.tick_top()

    # Add tick labels if basis symbols are not given
    if basis_symbols is None:
        # Set major ticks to start from 1
        # Include only every second or fourth tick if the matrix is large
        if operator_nonzeros.shape[0] <= 16:
            ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
            ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1))
        elif operator_nonzeros.shape[0] <= 64:
            ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 2))
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 2))
            ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1, 2))
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 2))
        else:
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 4))
            ax.set_xticklabels([])
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 4))

        # Apply font size to ticks
        if operator_nonzeros.shape[0] <= 16:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
        elif operator_nonzeros.shape[0] <= 64:
            ax.tick_params(axis='both', which='major', labelsize=fontsize + 1)
        else:
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    # Add basis symbols next to y tick labels if given
    elif basis_symbols is not None:
        basis_symbols = basis_symbols[rows_start:rows_end]

        # Check that the symbols describe the section being drawn.
        if len(basis_symbols) != operator_nonzeros.shape[0]:
            raise ValueError(f"Got {len(basis_symbols)} basis symbols for a section with "
                             f"{operator_nonzeros.shape[0]} rows. The basis symbols must "
                             f"correspond to the operator being visualized.")

        basis_symbols = [f'${symbol}$'.replace('*', '').replace(' ', '') for symbol in basis_symbols]
        basis_symbols = [f'{label}{" " * (5 - len(str(i + 1)))}{i + 1}' for i, label in enumerate(basis_symbols)]

        ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
        ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
        ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
        ax.set_yticklabels(basis_symbols)

        # Apply font size to ticks
        if operator_nonzeros.shape[0] <= 16:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
        else:
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    plt.tight_layout()
    plt.show()

def visualize_operator(
    operator: smp.MatrixBase,
    rows_start: int=0,
    rows_end: int | None=None,
    basis_symbols: list[smp.Expr] | None=None,
    fontsize: int=8,
) -> None:
    """
    Visualize a given operator (its matrix representation) as a nonzero-element plot.

    NOTE: Apart from basic plotting, this is mostly intended for pretty
    visualization purposes. The plot is shown automatically.

    Parameters
    ----------
    operator : sympy.Matrix
        Operator to be visualized.
    rows_start : int, optional
        Starting row/column index for the visualization. Default is 0.
    rows_end : int, optional
        Ending row/column index for the visualization. Default is None (last index).
    basis_symbols : list of sympy.Symbol, optional
        Basis operator symbols, drawn as a legend for the basis states. Default is None.
    fontsize : int, optional
        Font size for the basis symbol labels. Default is 8.
    """

    # Restrict the operator to the section to be drawn and mark its nonzero elements.
    operator = operator[rows_start:rows_end, rows_start:rows_end]
    operator_nonzeros = np.array(matrix_nonzeros(operator), dtype=np.float32)

    _plot_nonzero_pattern(operator_nonzeros, rows_start=rows_start, rows_end=rows_end,
                          basis_symbols=basis_symbols, fontsize=fontsize)

def visualize_many_operators(
    operators: list[smp.MatrixBase],
    rows_start: int=0,
    rows_end: int | None=None,
    basis_symbols: list[smp.Expr] | None=None,
    fontsize: int=8,
) -> None:
    """
    Visualize the combined nonzero-element pattern of a list of operators
    (their matrix representations).

    NOTE: Useful for, e.g., visualizing differences between several relaxation
    superoperators. Apart from basic plotting, this is mostly intended for
    pretty visualization purposes. The plot is shown automatically.

    Parameters
    ----------
    operators : list of sympy.Matrix
        Operators to be visualized.
    rows_start : int, optional
        Starting row/column index for the visualization. Default is 0.
    rows_end : int, optional
        Ending row/column index for the visualization. Default is None (last index).
    basis_symbols : list of sympy.Symbol, optional
        Basis operator symbols, drawn as a legend for the basis states. Default is None.
    fontsize : int, optional
        Font size for the basis symbol labels. Default is 8.
    """

    # Restrict each operator to the section to be drawn and sum their nonzero patterns.
    operators = [operator[rows_start:rows_end, rows_start:rows_end] for operator in operators]
    operators_nonzeros = [np.array(matrix_nonzeros(operator), dtype=np.float32)
                          for operator in operators]
    operator_nonzeros = np.sum(operators_nonzeros, axis=0)

    _plot_nonzero_pattern(operator_nonzeros, rows_start=rows_start, rows_end=rows_end,
                          basis_symbols=basis_symbols, fontsize=fontsize)
    
####################################################################################################
# Symbolic operators and expectation values.
####################################################################################################
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

####################################################################################################
# Spin operators (Zeeeman basis).
####################################################################################################
# Matrix representations:
def op_Sx(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum operator in the x-direction.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum operator ``Sx`` for spin quantum number `S`.
    """
    m = np.arange(-S, S+1)
    Sx = smp.zeros(len(m), len(m), complex=True)

    for i in range(len(m)):
        for j in range(len(m)):
            if i == j+1:
                Sx[i, j] = smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])
            elif i == j-1:
                Sx[i, j] = smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])

    return smp.Matrix(Sx.T, complex=True).applyfunc(smp.nsimplify)

def op_Sy(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum operator in the y-direction.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum operator ``Sy`` for spin quantum number `S`.
    """
    m = np.arange(-S, S+1)
    Sy = smp.zeros(len(m), len(m), complex=True)

    for i in range(len(m)):
        for j in range(len(m)):
            if i == j+1:
                Sy[i, j] = -smp.I * smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])
            elif i == j-1:
                Sy[i, j] = smp.I * smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])

    return smp.Matrix(Sy.T, complex=True).applyfunc(smp.nsimplify)

def op_Sz(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum operator in the z-direction.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum operator ``Sz`` for spin quantum number `S`.
    """
    m = np.arange(-S, S+1)
    m = np.flip(m)

    Sz = smp.zeros(len(m), len(m), complex=True)
    for i in range(len(m)):
        Sz[i, i] = m[i]

    return smp.Matrix(Sz.T, complex=True).applyfunc(smp.nsimplify)

def op_Sp(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum raising operator.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum raising operator for spin quantum number `S`.
    """
    return op_Sx(S) + smp.I * op_Sy(S)

def op_Sm(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum lowering operator.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum lowering operator for spin quantum number `S`.
    """
    return op_Sx(S) - smp.I * op_Sy(S)

def op_Svec(S: float) -> list[smp.Matrix]:
    """
    Build the Cartesian spin angular momentum vector operator.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    list of sympy.Matrix
        Cartesian spin operators ``[Sx, Sy, Sz]`` for spin quantum number `S`.
    """
    return [op_Sx(S), op_Sy(S), op_Sz(S)]

####################################################################################################
# Spherical tensors and spherical tensor operators.
# NOTE: These functions use dictionaries of the form {(l, q): T_lq} for spherical tensors.
####################################################################################################
# Classical spherical tensors:
def vector_to_spherical_tensor(vector: list) -> dict:
    """
    Convert a Cartesian vector to a classical spherical tensor of rank 1.

    Parameters
    ----------
    vector : list
        Cartesian vector components in the form ``[x, y, z]``.

    Returns
    -------
    dict
        Dictionary of the form ``{(l, q): T_lq}`` with rank ``l = 1`` and
        projections ``q = -1, 0, 1``.
    """
    T_m1 = (vector[0] - smp.I * vector[1]) / smp.sqrt(2)
    T_0 = vector[2]
    T_p1 = -(vector[0] + smp.I * vector[1]) / smp.sqrt(2)
    return {(1, -1): T_m1, (1, 0): T_0, (1, 1): T_p1}

# Spherical tensor operators:
def op_T(
    S: float,
    l: int,
    q: int,
) -> smp.MatrixBase | int:
    """
    Build the spherical tensor operator of spin quantum number S, rank l and
    projection q, obtained by sequential lowering of the maximum-projection
    operator (see the nested `op_T_ll`).

    NOTE: The returned operators are not normalized.

    Parameters
    ----------
    S : float
        Spin quantum number.
    l : int
        Rank of the spherical tensor operator.
    q : int
        Projection of the spherical tensor operator.

    Returns
    -------
    sympy.Matrix or int
        Spherical tensor operator ``T_lq`` for spin quantum number `S`, or 0
        if `l` or `q` are outside their allowed ranges.
    """
    def op_T_ll(
        S: float,
        l: int,
    ) -> smp.MatrixBase | int:
        """
        Build the spherical tensor operator of spin quantum number S, rank l
        and maximum projection q = l.

        Parameters
        ----------
        S : float
            Spin quantum number.
        l : int
            Rank of the spherical tensor operator.

        Returns
        -------
        sympy.Matrix or int
            Spherical tensor operator ``T_ll``, or 0 if `l` exceeds ``2*S``.
        """
        if l > int(2*S):
            return 0
        else:
            return ((-1.)**l * 2.**(-l/2) * (op_Sp(S))**l).applyfunc(lambda x: smp.nsimplify(x))
    if abs(q) > l:
        return 0
    else:
        T_ll = op_T_ll(S, l)
        S_m = op_Sm(S)
        for i in range(l - q):
            comm = commutator(S_m, T_ll)
            N = smp.sqrt((l - q)*(l + q + 1))
            T_ll = (1 / N) * comm
            q += 1
        return T_ll.applyfunc(smp.simplify)

# Coupling of spherical tensor operators.
# NOTE: Used for bilinear contributions in the relaxation superoperator.
def op_T_coupled_lq(
    T1_dict: dict,
    T2_dict: dict,
    l: int,
    q: int,
) -> smp.MatrixBase:
    """
    Build the coupled spherical tensor operator of rank l and projection q,
    obtained by coupling two rank-1 spherical tensors via Clebsch-Gordan coefficients.

    Parameters
    ----------
    T1_dict : dict
        First dictionary of spherical tensor components, of the form ``{(l, q): T_lq}``.
    T2_dict : dict
        Second dictionary of spherical tensor components, of the form ``{(l, q): T_lq}``.
    l : int
        Rank of the coupled spherical tensor operator.
    q : int
        Projection of the coupled spherical tensor operator.

    Returns
    -------
    sympy.Matrix
        Coupled spherical tensor operator of rank `l` and projection `q`.
    """
    T = smp.zeros(T1_dict[1, 1].shape[0], T1_dict[1, 1].shape[0], complex=True)
    for q1 in range(-1, 2):
        for q2 in range(-1, 2):
            T += CG(1, q1, 1, q2, l, q).doit() * T1_dict[1, q1] * T2_dict[1, q2]
    return T

####################################################################################################
# Superoperators for Liouville space.
# NOTE: Function inputs "op" have to be in Hilbert space, apart from "de" functions.
####################################################################################################
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

####################################################################################################
# Spin operator classes.
####################################################################################################
def many_spin_operator(
    S: list[float],
    single_spin_operator: smp.MatrixBase,
    spin_index: int,
) -> smp.MatrixBase:
    """
    Embed a single-spin operator into the full Hilbert space of a many-spin system.

    Parameters
    ----------
    S : list of float
        Spin quantum numbers of the spins in the system.
    single_spin_operator : sympy.Matrix
        Single-spin operator to be embedded.
    spin_index : int
        Index of the spin that `single_spin_operator` acts on.

    Returns
    -------
    sympy.Matrix
        Many-spin system version of `single_spin_operator`, obtained via the
        Kronecker product with unit operators on all other spins.
    """
    op = smp.eye(1)
    for i in range(len(S)):
        if i == spin_index:
            op = Kronecker_product(op, single_spin_operator)
        else:
            op = Kronecker_product(op, smp.eye(int(2*S[i]+1)))
    return op

class SpinOperators:
    """
    General class for the spin operators (Cartesian and spherical tensor) of a spin system.

    NOTE: One of the main classes of Rela2x.

    Parameters
    ----------
    spinsystem : list of str
        Nuclear isotopes (as string labels, see `nmr_isotopes.py`) that define the spin system.

    Attributes
    ----------
    S : list of float
        Spin quantum numbers of the spins.
    N_spins : int
        Number of spins in the system.
    N_states : int
        Number of Hilbert-space states in the system.
    E : list of sympy.Matrix
        Many-spin unit operator for each spin.
    E_symbol : list of sympy.physics.quantum.Operator
        Unit operator symbols for each spin.
    Sx, Sy, Sz, Sp, Sm : list of sympy.Matrix
        Many-spin Cartesian spin operators for each spin.
    Sx_symbol, Sy_symbol, Sz_symbol, Sp_symbol, Sm_symbol : list of sympy.physics.quantum.Operator
        Cartesian spin operator symbols for each spin.
    T : list of dict
        Many-spin spherical tensor operators for each spin, of the form ``{(l, q): T_lq}``.
    T_symbol : list of dict
        Spherical tensor operator symbols for each spin, of the form ``{(l, q): T_lq symbol}``.

    Raises
    ------
    ValueError
        If `spinsystem` is not a list of strings.
    """
    def __init__(
        self,
        spinsystem: list[str],
    ) -> None:
        """
        Initialise the spin system and generate its Cartesian and spherical
        tensor spin operators (and their symbols).

        Parameters
        ----------
        spinsystem : list of str
            Nuclear isotopes (as string labels, see `nmr_isotopes.py`) that define the spin system.
        """

        # Check that the input is a list of strings.
        if not all(isinstance(isotope, str) for isotope in spinsystem):
            raise ValueError("The spinsystem input has to be a list of strings corresponding to "
                             "NMR isotopes (e.g. ['1H', '13C']).")

        self.spinsystem = spinsystem
        self.S = spin_quantum_numbers(spinsystem)

        # Determine the size of the many-spin Hilbert space.
        self.N_spins = len(self.S)
        self.gen_N_states()

        # Generate the Cartesian spin operators and their symbols.
        self.gen_many_spin_cartesian_operators()
        self.gen_cartesian_operator_symbols()

        # Generate the spherical tensor spin operators and their symbols.
        self.gen_many_spin_T_operators()
        self.gen_T_operator_symbols()

    def gen_N_states(self) -> None:
        """
        Generate the number of Hilbert-space states in the spin system and
        store it in `self.N_states`.
        """
        self.N_states = 1
        for i in range(self.N_spins):
            self.N_states *= int(2*self.S[i] + 1)

    # Cartesian spin operators
    def gen_many_spin_cartesian_operators(self) -> None:
        """
        Generate the many-spin Cartesian spin operators and store them in
        `self.E`, `self.Sx`, `self.Sy`, `self.Sz`, `self.Sp` and `self.Sm`.
        """
        self.E = [many_spin_operator(self.S, smp.eye(int(2*S+1)), i) for i, S in enumerate(self.S)]
        self.Sx = [many_spin_operator(self.S, op_Sx(S), i) for i, S in enumerate(self.S)]
        self.Sy = [many_spin_operator(self.S, op_Sy(S), i) for i, S in enumerate(self.S)]
        self.Sz = [many_spin_operator(self.S, op_Sz(S), i) for i, S in enumerate(self.S)]
        self.Sp = [many_spin_operator(self.S, op_Sp(S), i) for i, S in enumerate(self.S)]
        self.Sm = [many_spin_operator(self.S, op_Sm(S), i) for i, S in enumerate(self.S)]

    def gen_cartesian_operator_symbols(self) -> None:
        """
        Generate the Cartesian spin operator symbols and store them in
        `self.E_symbol`, `self.Sx_symbol`, `self.Sy_symbol`, `self.Sz_symbol`,
        `self.Sp_symbol` and `self.Sm_symbol`.
        """
        self.E_symbol = [smpq.Operator(f'\\hat{{E}}^{{({i+1})}}') for i in range(self.N_spins)]
        self.Sx_symbol = [op_S_symbol('x', i+1) for i in range(self.N_spins)]
        self.Sy_symbol = [op_S_symbol('y', i+1) for i in range(self.N_spins)]
        self.Sz_symbol = [op_S_symbol('z', i+1) for i in range(self.N_spins)]
        self.Sp_symbol = [op_S_symbol('+', i+1) for i in range(self.N_spins)]
        self.Sm_symbol = [op_S_symbol('-', i+1) for i in range(self.N_spins)]

    # Spherical tensor operators
    def gen_many_spin_T_operators(self) -> None:
        """
        Generate the many-spin spherical tensor operators and store them in `self.T`.
        """

        def gen_T_operators(S: float) -> dict:
            """
            Generate the single-spin spherical tensor operators for a given
            spin quantum number.

            Parameters
            ----------
            S : float
                Spin quantum number.

            Returns
            -------
            dict
                Single-spin spherical tensor operators, of the form ``{(l, q): T_lq}``.
            """
            return {(l, q): op_T(S, l, q) for l in range(int(2*S)+1) for q in range(-l, l+1)}

        self.T = [gen_T_operators(S) for S in self.S]

        # Overwrite the single-spin operators with their many-spin system versions.
        self.T = [{(l, q): many_spin_operator(self.S, T_lq, i) for (l, q), T_lq in T.items()}
                  for i, T in enumerate(self.T)]

    def gen_T_operator_symbols(self) -> None:
        """
        Generate the spherical tensor operator symbols and store them in `self.T_symbol`.
        """
        self.T_symbol = [{(l, q): op_T_symbol(l, q, i+1) for (l, q), _ in T.items()} for i, T in enumerate(self.T)]

####################################################################################################
# Basis operators for Liouville space.
####################################################################################################
# Product basis construction
# NOTE: The Cartesian and spherical tensor bases share the same combinatorial structure and differ
# only in the single-spin operators used, in the ordering convention, and in how the single-spin
# identity operator is recognised. The two helpers below hold that shared structure.
def _product_basis_from_factors(
    factors: list[list[smp.MatrixBase]],
    reverse: bool,
) -> tuple[list[smp.MatrixBase], list[smp.Expr]]:
    """
    Build a normalized direct product basis from per-spin operator lists.

    Parameters
    ----------
    factors : list of list of sympy.Matrix
        Single-spin operators available for each spin, in a consistent order.
    reverse : bool
        Whether to reverse the order of the generated combinations.

    Returns
    -------
    basis : list of sympy.Matrix
        Normalized product basis operators.
    norms : list of sympy.Expr
        Liouville norms of the (unnormalized) product basis operators.
    """
    N_spins = len(factors)

    # Enumerate one operator index per spin, in every combination.
    index_combinations = all_combinations(N_spins, *[list_indexes(f) for f in factors],
                                          reverse=reverse)

    # Multiply the selected single-spin operators together for each combination.
    basis = []
    for index_combination in index_combinations:
        product = None
        for spin_index, factor_index in enumerate(index_combination):
            operator = factors[spin_index][factor_index]
            product = operator if product is None else product @ operator
        basis.append(product)

    # Norms of the basis operators. Used for normalization and later for basis to observables conversion.
    norms = [Lv_norm(operator) for operator in basis]

    # Normalize the basis.
    basis = [operator / norm for operator, norm in zip(basis, norms)]

    return basis, norms

def _product_basis_symbols_from_factors(
    symbol_factors: list[list[smp.Expr]],
    identity_marker: str,
    reverse: bool,
) -> list[smp.Expr]:
    """
    Build the direct product basis symbols from per-spin symbol lists.

    Parameters
    ----------
    symbol_factors : list of list of sympy.Expr
        Single-spin operator symbols available for each spin, ordered consistently
        with the operators they label.
    identity_marker : str
        Substring identifying the single-spin identity operator in a printed symbol.
    reverse : bool
        Whether to reverse the order of the generated combinations.

    Returns
    -------
    symbols : list of sympy.Expr
        Product basis symbols.
    """
    N_spins = len(symbol_factors)

    # Enumerate one symbol index per spin, in every combination.
    index_combinations = all_combinations(N_spins, *[list_indexes(f) for f in symbol_factors],
                                          reverse=reverse)

    # Multiply the selected single-spin symbols together for each combination.
    symbols = []
    for index_combination in index_combinations:
        product_symbol = 1
        for spin_index, factor_index in enumerate(index_combination):
            symbol = symbol_factors[spin_index][factor_index]

            # Leave single-spin identity operators out of the product symbol.
            if identity_marker not in str(symbol):
                product_symbol *= symbol

        # Denote a product consisting only of identity operators by a single identity symbol.
        if product_symbol == 1:
            product_symbol = smpq.Operator('\\hat{E}')

        symbols.append(product_symbol)

    return symbols

def _Cartesian_factors(
    spin_operators: SpinOperators,
    symbolic: bool,
) -> list[list]:
    """
    Collect the single-spin Cartesian operators, or their symbols, for every spin.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system whose operators are collected.
    symbolic : bool
        Whether to collect the operator symbols rather than the matrix representations.

    Returns
    -------
    list of list
        The E, Sx, Sy and Sz entries of each spin, in that order.
    """
    if symbolic:
        sets = (spin_operators.E_symbol, spin_operators.Sx_symbol,
                spin_operators.Sy_symbol, spin_operators.Sz_symbol)
    else:
        sets = (spin_operators.E, spin_operators.Sx, spin_operators.Sy, spin_operators.Sz)

    return [[entries[i] for entries in sets] for i in range(spin_operators.N_spins)]

def _T_factors(
    spin_operators: SpinOperators,
    symbolic: bool,
) -> list[list]:
    """
    Collect the single-spin spherical tensor operators, or their symbols, for every spin.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system whose operators are collected.
    symbolic : bool
        Whether to collect the operator symbols rather than the matrix representations.

    Returns
    -------
    list of list
        The T_lq entries of each spin, ordered by rank and then by projection.
    """
    S = spin_operators.S
    entries = spin_operators.T_symbol if symbolic else spin_operators.T

    return [[entries[i][(l, q)] for l in range(int(2*S[i]) + 1) for q in range(-l, l + 1)]
            for i in range(spin_operators.N_spins)]

# Product basis of Cartesian spin operators
def Cartesian_product_basis(spin_operators: SpinOperators) -> tuple[list[smp.MatrixBase], list[smp.Expr]]:
    """
    Generate the direct product basis of Cartesian spin operators.

    Each of the E, Sx, Sy, Sz operators of every spin is multiplied with the
    E, Sx, Sy, Sz operators of every other spin.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the product basis.

    Returns
    -------
    basis : list of sympy.Matrix
        Normalized product basis operators.
    norms : list of sympy.Expr
        Liouville norms of the (unnormalized) product basis operators.
    """
    return _product_basis_from_factors(_Cartesian_factors(spin_operators, symbolic=False),
                                       reverse=False)

def Cartesian_product_basis_symbols(spin_operators: SpinOperators) -> list[smp.Expr]:
    """
    Generate the direct product basis symbols of Cartesian spin operators.

    Each of the E, Sx, Sy, Sz symbols of every spin is multiplied with the
    E, Sx, Sy, Sz symbols of every other spin.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the product basis symbols.

    Returns
    -------
    list of sympy.Expr
        Product basis symbols.
    """
    return _product_basis_symbols_from_factors(_Cartesian_factors(spin_operators, symbolic=True),
                                               identity_marker='E', reverse=False)

def Cartesian_product_basis_and_symbols(
        spin_operators: SpinOperators
) -> tuple[list[smp.MatrixBase], list[smp.Expr], list[smp.Expr], list[tuple]]:
    """
    Generate the direct product basis of Cartesian spin operators together with
    their symbols and indices.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the product basis.

    Returns
    -------
    basis : list of sympy.Matrix
        Normalized product basis operators.
    symbols : list of sympy.Expr
        Product basis symbols.
    norms : list of sympy.Expr
        Liouville norms of the (unnormalized) product basis operators.
    indices : list of tuple
        Basis operator indices.
    """
    basis, norms = Cartesian_product_basis(spin_operators)
    symbols = Cartesian_product_basis_symbols(spin_operators)
    indices = Cartesian_product_basis_indices(spin_operators)

    return basis, symbols, norms, indices

# Product basis of spherical tensor operators
def T_product_basis(spin_operators: SpinOperators) -> tuple[list[smp.MatrixBase], list[smp.Expr]]:
    """
    Generate the direct product basis of spherical tensor operators.

    Each T_lq operator of every spin is multiplied with the T_lq operators of every other spin.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the product basis.

    Returns
    -------
    basis : list of sympy.Matrix
        Normalized product basis operators.
    norms : list of sympy.Expr
        Liouville norms of the (unnormalized) product basis operators.
    """
    return _product_basis_from_factors(_T_factors(spin_operators, symbolic=False), reverse=True)

def T_product_basis_symbols(spin_operators: SpinOperators) -> list[smp.Expr]:
    """
    Generate the direct product basis symbols of spherical tensor operators.

    Each T_lq symbol of every spin is multiplied with the T_lq symbols of every other spin.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the product basis symbols.

    Returns
    -------
    list of sympy.Expr
        Product basis symbols.
    """
    return _product_basis_symbols_from_factors(_T_factors(spin_operators, symbolic=True),
                                               identity_marker='_{00}', reverse=True)

def Cartesian_product_basis_indices(spin_operators: SpinOperators) -> list[tuple]:
    """
    Generate the basis operator indices of the Cartesian product basis.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the indices.

    Returns
    -------
    list of tuple
        One entry per basis operator, each a tuple of (spin index, direction)
        pairs for the spins carrying a non-identity operator.
    """
    N_spins = spin_operators.N_spins
    directions = [['E', 'x', 'y', 'z'] for _ in range(N_spins)]

    # Enumerate one direction index per spin, in the same order as the operators.
    index_combinations = all_combinations(N_spins, *[list_indexes(d) for d in directions],
                                          reverse=False)

    # Record the spins carrying a non-identity operator.
    indices = []
    for index_combination in index_combinations:
        entry = []
        for spin_index, factor_index in enumerate(index_combination):
            direction = directions[spin_index][factor_index]
            if direction != 'E':
                entry.append((spin_index + 1, direction))
        indices.append(tuple(entry))

    return indices

def T_product_basis_indices(spin_operators: SpinOperators) -> list[tuple]:
    """
    Generate the basis operator indices of the spherical tensor product basis.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the indices.

    Returns
    -------
    list of tuple
        One entry per basis operator, each a tuple of (spin index, l, q) triples
        for the spins carrying a non-identity operator.
    """
    S = spin_operators.S
    N_spins = spin_operators.N_spins
    lqs = [[(l, q) for l in range(int(2*S[i]) + 1) for q in range(-l, l + 1)]
           for i in range(N_spins)]

    # Enumerate one (l, q) index per spin, in the same order as the operators.
    index_combinations = all_combinations(N_spins, *[list_indexes(lq) for lq in lqs],
                                          reverse=True)

    # Record the spins carrying a non-identity operator.
    indices = []
    for index_combination in index_combinations:
        entry = []
        for spin_index, factor_index in enumerate(index_combination):
            l, q = lqs[spin_index][factor_index]
            if (l, q) != (0, 0):
                entry.append((spin_index + 1, l, q))
        indices.append(tuple(entry))

    return indices

# Basis set sorting
# NOTE: A sorting scheme is expressed as an ordered list of sort keys, from the most significant
# to the least. Each key is evaluated for every basis operator and the basis is ordered by the
# resulting tuples, so the hierarchy is explicit rather than implied by a sequence of passes.
# There are multiple sensible ways to order the basis; the two schemes below are just possibilities.
def T_index_coherence_sort_key(T_index: tuple) -> tuple:
    """
    Sort key ordering basis operators by coherence order.

    Coherences are grouped by magnitude first, so that a coherence order and its
    negative stay adjacent.

    Parameters
    ----------
    T_index : tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.

    Returns
    -------
    tuple
        Sort key of the form (absolute coherence order, coherence order).
    """
    coherence_order = T_index_coherence_order(T_index)

    return (abs(coherence_order), coherence_order)

def T_index_projection_sort_key(T_index: tuple, N_spins: int) -> tuple:
    """
    Sort key ordering basis operators by the projections carried by each spin.

    The spins are taken in reverse order, so that the last spin is the most
    significant, matching the order in which the projections are applied.

    Parameters
    ----------
    T_index : tuple
        Basis operator index, as a tuple of (spin index, l, q) triples.
    N_spins : int
        Number of spins in the system.

    Returns
    -------
    tuple
        Projections of the spins, from the last spin to the first.
    """
    return tuple(T_index_spin_projection(T_index, spin_index)
                 for spin_index in range(N_spins, 0, -1))

def T_basis_sort_keys(sorting: str, N_spins: int) -> list:
    """
    Build the ordered list of sort keys defining a sorting scheme.

    Version ``'v1'`` orders by coherence order, then by spin order, then by type.
    Version ``'v2'`` places the identity operator first, then orders by coherence
    order, then by the projections carried by each spin.

    Parameters
    ----------
    sorting : str
        Sorting scheme, ``'v1'`` or ``'v2'``.
    N_spins : int
        Number of spins in the system.

    Returns
    -------
    list of callable
        Sort keys, from the most significant to the least, each taking a basis
        operator index and returning a comparable value.

    Raises
    ------
    ValueError
        If `sorting` is not a recognised scheme.
    """
    if sorting == 'v1':
        return [T_index_coherence_sort_key,
                T_index_spin_order,
                T_index_type]

    if sorting == 'v2':
        return [lambda T_index: 0 if T_index_spin_order(T_index) == 0 else 1,
                T_index_coherence_sort_key,
                lambda T_index: T_index_projection_sort_key(T_index, N_spins)]

    raise ValueError("Invalid sorting scheme. Choose 'v1' or 'v2'.")

def sort_T_product_basis(
    T_product_basis: list[smp.MatrixBase],
    T_product_basis_symbols: list[smp.Expr],
    T_product_basis_norms: list[smp.Expr],
    T_product_basis_indices: list[tuple],
    sort_keys: list,
) -> tuple[list[smp.MatrixBase], list[smp.Expr], list[smp.Expr], list[tuple]]:
    """
    Sort the product basis of spherical tensor operators by a list of sort keys.

    Parameters
    ----------
    T_product_basis : list of sympy.Matrix
        Basis of spherical tensor operators.
    T_product_basis_symbols : list of sympy.Expr
        Basis of spherical tensor operator symbols.
    T_product_basis_norms : list of sympy.Expr
        Norms of the basis operators.
    T_product_basis_indices : list of tuple
        Basis operator indices.
    sort_keys : list of callable
        Sort keys, from the most significant to the least.

    Returns
    -------
    tuple
        Sorted ``(T_product_basis, T_product_basis_symbols, T_product_basis_norms,
        T_product_basis_indices)``.
    """
    # Order the basis by the combined key, keeping equally ranked operators in their original order.
    order = sorted(list_indexes(T_product_basis_indices),
                   key=lambda i: tuple(key(T_product_basis_indices[i]) for key in sort_keys))

    return ([T_product_basis[i] for i in order],
            [T_product_basis_symbols[i] for i in order],
            [T_product_basis_norms[i] for i in order],
            [T_product_basis_indices[i] for i in order])

def T_product_basis_and_symbols(
        spin_operators: SpinOperators,
        sorting: str | None='v1'
) -> tuple[list[smp.MatrixBase], list[smp.Expr], list[smp.Expr], list[tuple]]:
    """
    Generate and sort the product basis of spherical tensor operators, together
    with their symbols and indices.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to generate the product basis.
    sorting : str or None, optional
        Sorting scheme, ``'v1'`` or ``'v2'``, or None to skip sorting. Default is ``'v1'``.

    Returns
    -------
    basis : list of sympy.Matrix
        Sorted basis of spherical tensor operators.
    symbols : list of sympy.Expr
        Sorted basis of spherical tensor operator symbols.
    norms : list of sympy.Expr
        Norms of the (unnormalized) basis operators, sorted correspondingly.
    indices : list of tuple
        Basis operator indices, sorted correspondingly.
    """
    basis, norms = T_product_basis(spin_operators)
    symbols = T_product_basis_symbols(spin_operators)
    indices = T_product_basis_indices(spin_operators)

    if sorting is None:
        return basis, symbols, norms, indices

    sort_keys = T_basis_sort_keys(sorting, spin_operators.N_spins)

    return sort_T_product_basis(basis, symbols, norms, indices, sort_keys)

####################################################################################################
# General operator classes.
####################################################################################################
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
    """
    def __init__(
        self,
        sop: smp.MatrixBase,
    ) -> None:
        """
        Initialise the superoperator and extract its symbols and functions.

        Parameters
        ----------
        sop : sympy.Matrix
            Matrix representation of the superoperator.
        """
        super().__init__(sop)

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
        print('\nChanging basis...')
        self.op = op_change_of_basis(self.op, basis_vectorized)
        print('Basis changed.')

####################################################################################################
# Spectral density functions and relaxation constants.
####################################################################################################
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

    NOTE: ``beta = hbar / (k_B * T)``, defined in `constants_and_variables.py`.

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
        for the quantum mechanical theory, depending on `settings.RELAXATION_THEORY`.
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

    if settings.RELAXATION_THEORY == 'sc':
        return expr
    elif settings.RELAXATION_THEORY == 'qm':
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
    
####################################################################################################
# Relaxation superoperators.
# NOTE: alpha is single-spin interaction and beta is two-spin interaction.
# See https://doi.org/10.1016/j.jmr.2024.107828
####################################################################################################
def sop_R_term(
    op_T_left: smp.MatrixBase,
    J_w: smp.Expr,
    op_T_right: smp.MatrixBase,
) -> smp.MatrixBase:
    """
    Build a single term in the sum defining the relaxation superoperator,
    schematically T * J(w) * T^dagger.

    NOTE: The "left"/"right" naming of the operators is for bookkeeping
    purposes only.

    Parameters
    ----------
    op_T_left : sympy.Matrix
        Left spherical tensor operator.
    J_w : sympy.Expr
        Spectral density function.
    op_T_right : sympy.Matrix
        Right spherical tensor operator.

    Returns
    -------
    sympy.Matrix
        Term in the relaxation superoperator, built from the semiclassical
        double-commutator or the Lindbladian dissipation superoperator,
        depending on `settings.RELAXATION_THEORY`.
    """
    if settings.RELAXATION_THEORY == 'sc':
        return smp.Rational(1, 2) * J_w * sop_double_commutator(op_T_left.H, op_T_right)
    elif settings.RELAXATION_THEORY == 'qm':
        return -J_w * sop_D(op_T_left.H, op_T_right)

# NOTE: The functions below have the same input and return structure as sop_R_term_alpha_alpha.
def sop_R_term_alpha_alpha(
    l: int,
    q: int,
    alpha1: str,
    alpha2: str,
    alpha1_spin_name: str,
    alpha2_spin_name: str,
    op_T_left: smp.MatrixBase,
    op_T_right: smp.MatrixBase,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Build the term in the relaxation superoperator between two single-spin interactions.

    Parameters
    ----------
    l : int
        Rank of the spherical tensor operator.
    q : int
        Projection of the spherical tensor operator.
    alpha1 : str
        Name of the first single-spin interaction.
    alpha2 : str
        Name of the second single-spin interaction.
    alpha1_spin_name : str
        Name of the spin associated with the first single-spin interaction.
    alpha2_spin_name : str
        Name of the spin associated with the second single-spin interaction.
    op_T_left : sympy.Matrix
        Left spherical tensor operator.
    op_T_right : sympy.Matrix
        Right spherical tensor operator.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    sympy.Matrix
        Term in the relaxation superoperator, or a zero matrix of matching
        shape if the term is excluded by the secular approximation.
    """
    w_s1 = smp.Symbol(f'\\omega_{{{alpha1_spin_name}}}', real=True)
    w_s2 = smp.Symbol(f'\\omega_{{{alpha2_spin_name}}}', real=True)

    # Dirac delta function argument for the secular approximation
    delta_sec = q*(w_s1 - w_s2)

    # Check secular approximation
    if delta_sec != 0:

        # Keep non-secular terms if specified (and semiclassical relaxation theory is used)
        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            # Spectral density function with argument defined by the second interaction
            argument = q*w_s2
            J = J_w(alpha1, alpha2, l, argument)

            return sop_R_term(op_T_left, J, op_T_right)

        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        # Spectral density function with argument defined by the second interaction
        argument = q*w_s2
        J = J_w(alpha1, alpha2, l, argument)

        return sop_R_term(op_T_left, J, op_T_right)

def sop_R_term_alpha_beta(
    l: int,
    q1: int,
    q2: int,
    alpha: str,
    beta: str,
    alpha_spin_name: str,
    beta_spin_name1: str,
    beta_spin_name2: str,
    op_T_left: smp.MatrixBase,
    op_T_right: smp.MatrixBase,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Build the term in the relaxation superoperator between a single-spin
    interaction and a two-spin interaction.

    Parameters
    ----------
    l : int
        Rank of the spherical tensor operator.
    q1 : int
        Projection of the coupled single-spin tensor component.
    q2 : int
        Projection of the second rank-1 component coupled into the two-spin interaction.
    alpha : str
        Name of the single-spin interaction.
    beta : str
        Name of the two-spin interaction.
    alpha_spin_name : str
        Name of the spin associated with the single-spin interaction.
    beta_spin_name1 : str
        Name of the first spin associated with the two-spin interaction.
    beta_spin_name2 : str
        Name of the second spin associated with the two-spin interaction.
    op_T_left : sympy.Matrix
        Left spherical tensor operator.
    op_T_right : sympy.Matrix
        Right spherical tensor operator.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    sympy.Matrix
        Term in the relaxation superoperator, or a zero matrix of matching
        shape if the term is excluded by the secular approximation.
    """
    w_s = smp.Symbol(f'\\omega_{{{alpha_spin_name}}}', real=True)
    w_t1 = smp.Symbol(f'\\omega_{{{beta_spin_name1}}}', real=True)
    w_t2 = smp.Symbol(f'\\omega_{{{beta_spin_name2}}}', real=True)
    
    delta_sec = (q1+q2)*w_s - q1*w_t1 - q2*w_t2

    if delta_sec != 0:

        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            argument = q1*w_t1 + q2*w_t2
            J = J_w(alpha, beta, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        argument = q1*w_t1 + q2*w_t2
        J = J_w(alpha, beta, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()

def sop_R_term_beta_alpha(
    l: int,
    q1: int,
    q2: int,
    beta: str,
    alpha: str,
    beta_spin_name1: str,
    beta_spin_name2: str,
    alpha_spin_name: str,
    op_T_left: smp.MatrixBase,
    op_T_right: smp.MatrixBase,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Build the term in the relaxation superoperator between a two-spin
    interaction and a single-spin interaction.

    See `sop_R_term_alpha_beta` for the parameter and return structure
    (with `beta`/`alpha` interaction order swapped accordingly).
    """
    w_t1 = smp.Symbol(f'\\omega_{{{beta_spin_name1}}}', real=True)
    w_t2 = smp.Symbol(f'\\omega_{{{beta_spin_name2}}}', real=True)
    w_s = smp.Symbol(f'\\omega_{{{alpha_spin_name}}}', real=True)

    delta_sec = q1*w_t1 + q2*w_t2 - (q1+q2)*w_s

    if delta_sec != 0:

        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            argument = (q1+q2)*w_s
            J = J_w(beta, alpha, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        argument = (q1+q2)*w_s
        J = J_w(beta, alpha, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()

def sop_R_term_beta_beta(
    l: int,
    q1_t1: int,
    q2_t1: int,
    q1_t2: int,
    q2_t2: int,
    beta1: str,
    beta2: str,
    beta1_spin_name1: str,
    beta1_spin_name2: str,
    beta2_spin_name1: str,
    beta2_spin_name2: str,
    op_T_left: smp.MatrixBase,
    op_T_right: smp.MatrixBase,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Build the term in the relaxation superoperator between two two-spin interactions.

    Parameters
    ----------
    l : int
        Rank of the spherical tensor operator.
    q1_t1 : int
        Projection of the first rank-1 component coupled into the first two-spin interaction.
    q2_t1 : int
        Projection of the second rank-1 component coupled into the first two-spin interaction.
    q1_t2 : int
        Projection of the first rank-1 component coupled into the second two-spin interaction.
    q2_t2 : int
        Projection of the second rank-1 component coupled into the second two-spin interaction.
    beta1 : str
        Name of the first two-spin interaction.
    beta2 : str
        Name of the second two-spin interaction.
    beta1_spin_name1 : str
        Name of the first spin associated with the first two-spin interaction.
    beta1_spin_name2 : str
        Name of the second spin associated with the first two-spin interaction.
    beta2_spin_name1 : str
        Name of the first spin associated with the second two-spin interaction.
    beta2_spin_name2 : str
        Name of the second spin associated with the second two-spin interaction.
    op_T_left : sympy.Matrix
        Left spherical tensor operator.
    op_T_right : sympy.Matrix
        Right spherical tensor operator.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    sympy.Matrix
        Term in the relaxation superoperator, or a zero matrix of matching
        shape if the term is excluded by the secular approximation.
    """
    w1_t1 = smp.Symbol(f'\\omega_{{{beta1_spin_name1}}}', real=True)
    w1_t2 = smp.Symbol(f'\\omega_{{{beta1_spin_name2}}}', real=True)
    w2_t1 = smp.Symbol(f'\\omega_{{{beta2_spin_name1}}}', real=True)
    w2_t2 = smp.Symbol(f'\\omega_{{{beta2_spin_name2}}}', real=True)

    delta_sec = q1_t1*w1_t1 + q2_t1*w1_t2 - q1_t2*w2_t1 - q2_t2*w2_t2
    
    if delta_sec != 0:

        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            argument = q1_t1*w1_t1 + q2_t1*w1_t2
            J = J_w(beta1, beta2, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_t1, 1, q2_t1, l, (q1_t1+q2_t1)).doit()\
                                                        * CG(1, q1_t2, 1, q2_t2, l, (q1_t2+q2_t2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        argument = q1_t1*w1_t1 + q2_t1*w1_t2
        J = J_w(beta1, beta2, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_t1, 1, q2_t1, l, (q1_t1+q2_t1)).doit()\
                                                    * CG(1, q1_t2, 1, q2_t2, l, (q1_t2+q2_t2)).doit()

def sop_R(
    spin_operators: SpinOperators,
    INCOHERENT_INTERACTIONS: dict,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Compute the matrix representation of the relaxation superoperator in Liouville space.

    NOTE: This is the implementation of the main equations in
    https://doi.org/10.1016/j.jmr.2024.107828, summing the alpha-alpha,
    alpha-beta, beta-alpha and beta-beta contributions (see `sop_R_term_alpha_alpha`,
    `sop_R_term_alpha_beta`, `sop_R_term_beta_alpha` and `sop_R_term_beta_beta`)
    over every pair of mechanisms in `INCOHERENT_INTERACTIONS`.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to compute the relaxation superoperator.
    INCOHERENT_INTERACTIONS : dict
        Dictionary of incoherent interactions (see README.md for the required format).
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    R_final : sympy.Matrix
        Matrix representation of the relaxation superoperator in Liouville space.

    Raises
    ------
    ValueError
        If `INCOHERENT_INTERACTIONS` specifies an interaction dictionary that
        is not of the format described in README.md.
    """
    # Initialize the relaxation superoperator.
    R_final = smp.zeros(spin_operators.N_states**2, spin_operators.N_states**2, complex=True)

    # Prepare coupling vector for linear interactions
    # NOTE: assumed always along z-axis (the magnetic field direction) 
    T_vector = vector_to_spherical_tensor([0, 0, 1])

    print('\nComputing R for interaction pairs...')

    # Loop over all mechanism pairs
    for mechanism1, properties1 in INCOHERENT_INTERACTIONS.items():
        for mechanism2, properties2 in INCOHERENT_INTERACTIONS.items():

            # Single-spin single-spin mechanism pair
            if properties1[0][0] == '1' and properties2[0][0] == '1':

                # Lists of coupling strengths, list indices stand for the spin indices
                # NOTE: Coupling strength is always 0 or 1 for the symbolic relaxation superoperator
                coupling_strengths1 = properties1[1]
                coupling_strengths2 = properties2[1]

                # Lists of ranks
                ranks1 = properties1[2]
                ranks2 = properties2[2]

                # Commom rank, Hubbard's approximation
                ls = list(set(ranks1) & set(ranks2))

                # Loop over all interactions pairs
                for spin_1_index, coupling_strength_1 in enumerate(coupling_strengths1):
                    if coupling_strength_1 != 0:

                        for spin_2_index, coupling_strength_2 in enumerate(coupling_strengths2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index + 1)
                                intr_name2 = mechanism2 + str(spin_2_index + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name = spin_operators.spinsystem[spin_1_index]
                                spin_2_name = spin_operators.spinsystem[spin_2_index]

                                # Loop over all common ranks and components
                                for l in ls:
                                    for q in range(-l, l+1):

                                        # Check if linear or quadratic interaction
                                        if properties1[0][1] == 'L':
                                            T_left = op_T_coupled_lq(spin_operators.T[spin_1_index], T_vector, l, q)
                                        elif properties1[0][1] == 'Q':
                                            T_left = spin_operators.T[spin_1_index][l, q]

                                        if properties2[0][1] == 'L':
                                            T_right = op_T_coupled_lq(spin_operators.T[spin_2_index], T_vector, l, q)
                                        elif properties2[0][1] == 'Q':
                                            T_right = spin_operators.T[spin_2_index][l, q]

                                        R_term = sop_R_term_alpha_alpha(
                                            l, q, intr_name1, intr_name2,
                                            spin_1_name, spin_2_name, T_left, T_right,
                                            keep_non_secular=keep_non_secular)

                                        # Add the relaxation superoperator term to the relaxation superoperator
                                        R_final += R_term

            # Single-spin two-spin mechanism pair
            elif properties1[0][0] == '1' and properties2[0][0] == '2':
                
                # List of coupling strengths and coupling-strength matrix
                coupling_strengths1 = properties1[1]
                coupling_strengths_matrix2 = properties2[1]

                ranks1 = properties1[2]
                ranks2 = properties2[2]

                ls = list(set(ranks1) & set(ranks2))

                # Loop over all interactions pairs (now including the coupling-strength matrix)
                for spin_1_index, coupling_strength_1 in enumerate(coupling_strengths1):
                    if coupling_strength_1 != 0:

                        for (spin_2_index_i, spin_2_index_j), coupling_strength_2 \
                                in np.ndenumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index + 1)
                                intr_name2 = mechanism2 + str(spin_2_index_i + 1) + str(spin_2_index_j + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name = spin_operators.spinsystem[spin_1_index]
                                spin_2_name_i = spin_operators.spinsystem[spin_2_index_i]
                                spin_2_name_j = spin_operators.spinsystem[spin_2_index_j]

                                for l in ls:
                                    # Loop over q1 and q2 values in the symbolic case
                                    for q1 in range(-1, 2):
                                        for q2 in range(-1, 2):

                                            # Clebsch-Gordan restriction
                                            if np.abs(q1 + q2) <= l:

                                                if properties1[0][1] == 'L':
                                                    T_left = op_T_coupled_lq(
                                                        spin_operators.T[spin_1_index],
                                                        T_vector, l, (q1 + q2))
                                                elif properties1[0][1] == 'Q':
                                                    T_left = spin_operators.T[spin_1_index][l, (q1 + q2)]

                                                # Interaction 2 is always bilinear if type is '2'
                                                T_right_i = spin_operators.T[spin_2_index_i]
                                                T_right_j = spin_operators.T[spin_2_index_j]
                                                T_right = T_right_i[1, q1] @ T_right_j[1, q2]

                                                R_term = sop_R_term_alpha_beta(
                                                    l, q1, q2, intr_name1, intr_name2,
                                                    spin_1_name, spin_2_name_i, spin_2_name_j,
                                                    T_left, T_right,
                                                    keep_non_secular=keep_non_secular)
                                                R_final += R_term

            # Double-spin single-spin mechanism pair
            elif properties1[0][0] == '2' and properties2[0][0] == '1':
                    
                coupling_strengths_matrix1 = properties1[1]
                coupling_strengths2 = properties2[1]

                ranks1 = properties1[2]
                ranks2 = properties2[2]

                ls = list(set(ranks1) & set(ranks2))

                for (spin_1_index_i, spin_1_index_j), coupling_strength_1 in np.ndenumerate(coupling_strengths_matrix1):
                    if coupling_strength_1 != 0:

                        for spin_2_index, coupling_strength_2 in enumerate(coupling_strengths2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index_i + 1) + str(spin_1_index_j + 1)
                                intr_name2 = mechanism2 + str(spin_2_index + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name_i = spin_operators.spinsystem[spin_1_index_i]
                                spin_1_name_j = spin_operators.spinsystem[spin_1_index_j]
                                spin_2_name = spin_operators.spinsystem[spin_2_index]

                                for l in ls:
                                    for q1 in range(-1, 2):
                                        for q2 in range(-1, 2):

                                            if np.abs(q1 + q2) <= l:
                                            
                                                T_left_i = spin_operators.T[spin_1_index_i]
                                                T_left_j = spin_operators.T[spin_1_index_j]
                                                T_left = T_left_i[1, q1] @ T_left_j[1, q2]

                                                if properties2[0][1] == 'L':
                                                    T_right = op_T_coupled_lq(
                                                        spin_operators.T[spin_2_index],
                                                        T_vector, l, (q1 + q2))
                                                elif properties2[0][1] == 'Q':
                                                    T_right = spin_operators.T[spin_2_index][l, (q1 + q2)]

                                                R_term = sop_R_term_beta_alpha(
                                                    l, q1, q2, intr_name1, intr_name2,
                                                    spin_1_name_i, spin_1_name_j, spin_2_name,
                                                    T_left, T_right,
                                                    keep_non_secular=keep_non_secular)
                                                R_final += R_term

            # Double-spin two-spin mechanism pair
            elif properties1[0][0] == '2' and properties2[0][0] == '2':
                
                coupling_strengths_matrix1 = properties1[1]
                coupling_strengths_matrix2 = properties2[1]

                ranks1 = properties1[2]
                ranks2 = properties2[2]

                ls = list(set(ranks1) & set(ranks2))

                for (spin_1_index_i, spin_1_index_j), coupling_strength_1 in np.ndenumerate(coupling_strengths_matrix1):
                    if coupling_strength_1 != 0:

                        for (spin_2_index_i, spin_2_index_j), coupling_strength_2 \
                                in np.ndenumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index_i + 1) + str(spin_1_index_j + 1)
                                intr_name2 = mechanism2 + str(spin_2_index_i + 1) + str(spin_2_index_j + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name_i = spin_operators.spinsystem[spin_1_index_i]
                                spin_1_name_j = spin_operators.spinsystem[spin_1_index_j]
                                spin_2_name_i = spin_operators.spinsystem[spin_2_index_i]
                                spin_2_name_j = spin_operators.spinsystem[spin_2_index_j]

                                for l in ls:
                                    for q1_d1 in range(-1, 2):
                                        for q2_d1 in range(-1, 2):
                                            for q1_d2 in range(-1, 2):
                                                for q2_d2 in range(-1, 2):

                                                    # Clebsch-Gordan restriction
                                                    if np.abs(q1_d1 + q2_d1) <= l \
                                                            and (q1_d1 + q2_d1) == (q1_d2 + q2_d2):

                                                        T_left_i = spin_operators.T[spin_1_index_i]
                                                        T_left_j = spin_operators.T[spin_1_index_j]
                                                        T_left = T_left_i[1, q1_d1] @ T_left_j[1, q2_d1]

                                                        T_right_i = spin_operators.T[spin_2_index_i]
                                                        T_right_j = spin_operators.T[spin_2_index_j]
                                                        T_right = T_right_i[1, q1_d2] @ T_right_j[1, q2_d2]

                                                        R_term = sop_R_term_beta_beta(
                                                            l, q1_d1, q2_d1, q1_d2, q2_d2,
                                                            intr_name1, intr_name2,
                                                            spin_1_name_i, spin_1_name_j,
                                                            spin_2_name_i, spin_2_name_j,
                                                            T_left, T_right,
                                                            keep_non_secular=keep_non_secular)
                                                        R_final += R_term
      
            else:
                raise ValueError('Invalid interaction dictionary. See README.md for details.')
            
    print('R computed.')
    return R_final

####################################################################################################
# Relaxation superoperator class.
####################################################################################################
class RelaxationSuperoperator(Superoperator):
    """
    General class for the relaxation superoperator of a spin system.
    Inherits from `Superoperator`.

    See `Superoperator` and `Operator` for more information.

    NOTE: The main class of Rela2x.

    Parameters
    ----------
    sop_R : sympy.Matrix
        Relaxation superoperator matrix representation.
    basis_symbols : list of sympy.Symbol
        Basis operator symbols.
    basis_norms : list of sympy.Expr
        Liouville norms of the (unnormalized) basis operators.

    Attributes
    ----------
    basis_symbols : list of sympy.Symbol
        Basis operator symbols.
    basis_norms : list of sympy.Expr
        Liouville norms of the (unnormalized) basis operators.
    """
    def __init__(
        self,
        sop_R: smp.MatrixBase,
        basis_symbols: list[smp.Expr],
        basis_norms: list[smp.Expr],
        basis_indices: list[tuple],
    ) -> None:
        """
        Initialise the relaxation superoperator.

        Parameters
        ----------
        sop_R : sympy.Matrix
            Relaxation superoperator matrix representation.
        basis_symbols : list of sympy.Symbol
            Basis operator symbols.
        basis_norms : list of sympy.Expr
            Liouville norms of the (unnormalized) basis operators.
        basis_indices : list of tuple
            Basis operator indices, used for relaxation rate lookups and filtering.
        """
        Superoperator.__init__(self, sop_R)
        self.basis_symbols = basis_symbols
        self.basis_norms = basis_norms
        self.basis_indices = basis_indices

    def to_observables(self) -> None:
        """
        Fix the basis operator normalization in `self.op`, in order to obtain
        the correct relaxation rates and equations of motion of observables.
        """
        op = self.op
        norms = self.basis_norms

        # Rescale every matrix element by the norms of its row and column basis operators.
        # NOTE: A new matrix is constructed, so that the method also accepts an immutable
        # self.op, as returned by, e.g., simplify().
        rescaled = smp.Matrix(op.shape[0], op.shape[1],
                              lambda i, j: op[i, j] * (norms[i] / norms[j]))

        # Simplify the relaxation superoperator.
        self.op = smp.simplify(rescaled)

    def rate(
        self,
        spin_index_op_index_1: str,
        spin_index_op_index_2: str | None=None,
    ) -> smp.Expr | None:
        """
        Get the relaxation rate between two basis operators.

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
            the same format as `spin_index_op_index_1`. If None, the
            auto-relaxation rate of `spin_index_op_index_1` is returned. Default is None.

        Returns
        -------
        sympy.Expr or None
            Relaxation rate between the two operators, or None if either
            operator is not found in the basis.
        """
        # Interpret the specifications as Cartesian or as spherical tensor operators.
        if spin_index_op_index_1[-1] in 'xyz':
            to_index = S_index_from_string
        else:
            to_index = T_index_from_string

        # Locate the requested operators in the basis.
        index_1 = basis_index_list_index(self.basis_indices, to_index(spin_index_op_index_1))
        if spin_index_op_index_2 is None:
            index_2 = index_1
        else:
            index_2 = basis_index_list_index(self.basis_indices,
                                             to_index(spin_index_op_index_2))

        # Return nothing if either operator was not found in the basis.
        # NOTE: The lookup function above reports the failure and returns None.
        if index_1 is None or index_2 is None:
            return None

        return self.op[index_1, index_2]

    def to_isotropic_rotational_diffusion(
        self,
        fast_motion_limit: bool=False,
        slow_motion_limit: bool=False,
    ) -> None:
        """
        Substitute every J(w) function in the relaxation superoperator with
        the isotropic rotational diffusion spectral density function.

        Parameters
        ----------
        fast_motion_limit : bool, optional
            Whether to use the fast-motion limit. Default is False.
        slow_motion_limit : bool, optional
            Whether to use the slow-motion limit. Default is False.
        """
        J_w_functions = [function for function in self.functions_in if 'J' in str(function)]
        subst_dict = {}

        # Build the substitution for every anisotropic (l > 0) spectral density function.
        for J_w in J_w_functions:
            # See extract_J_w_symbols_and_args and J_w_isotropic_rotational_diffusion.
            intrs, lq, arg = extract_J_w_symbols_and_args(J_w)
            if lq[0] > 0:
                subst_dict[J_w] = J_w_isotropic_rotational_diffusion(*intrs, lq[0], arg,
                                    fast_motion_limit=fast_motion_limit, slow_motion_limit=slow_motion_limit)
        self.substitute(subst_dict)

    def neglect_cross_correlated_terms(
        self,
        mechanism1: str | None=None,
        mechanism2: str | None=None,
    ) -> None:
        """
        Neglect the cross-correlated terms between `mechanism1` and
        `mechanism2` in the relaxation superoperator.

        NOTE: If `mechanism1` and `mechanism2` are both None, all
        cross-correlated terms are neglected.

        Parameters
        ----------
        mechanism1 : str, optional
            Name of the first mechanism. Default is None.
        mechanism2 : str, optional
            Name of the second mechanism. If None, `mechanism1` is used. Default is None.
        """
        J_w_functions = [function for function in self.functions_in if 'J' in str(function) or 'G' in str(function)]

        # Neglect all cross-correlated terms if mechanism1 and mechanism2 are both None.
        if mechanism1 is None and mechanism2 is None:
            for J_w in J_w_functions:
                # See extract_J_w_symbols_and_args.
                intrs, _, _ = extract_J_w_symbols_and_args(J_w)
                if ',' in intrs[0]:
                    self.substitute({J_w: 0})

        # Otherwise, neglect cross-correlated terms between mechanism1 and mechanism2 only.
        else:
            if mechanism2 is None:
                mechanism2 = mechanism1
            for J_w in J_w_functions:
                # See extract_J_w_symbols_and_args.
                intrs, _, _ = extract_J_w_symbols_and_args(J_w)
                if ',' in intrs[0]:
                    if mechanism1 == mechanism2:
                        if str(J_w).count(mechanism1) > 1:
                            self.substitute({J_w: 0})
                    else:
                        if mechanism1 in str(J_w) and mechanism2 in str(J_w):
                            self.substitute({J_w: 0})

    def neglect_cross_relaxation(self) -> None:
        """
        Neglect all cross-relaxation terms (off-diagonal elements) in the relaxation superoperator.
        """
        op = self.op

        # Retain the diagonal auto-relaxation elements and zero all off-diagonal ones.
        # NOTE: A new matrix is constructed, so that the method also accepts an immutable
        # self.op, as returned by, e.g., simplify().
        self.op = smp.Matrix(op.shape[0], op.shape[1],
                             lambda i, j: op[i, j] if i == j else 0)

    def filter(
        self,
        filter_name: str,
        filter_value: int | list[int],
    ) -> None:
        """
        Filter out regions of the relaxation superoperator based on given criteria.

        See `coherence_order_filter`, `spin_order_filter` and `type_filter`
        for more information.

        NOTE: Works only for the irreducible spherical tensor basis.

        Parameters
        ----------
        filter_name : {'c', 's', 't'}
            Filter criterion: ``'c'`` for coherence order, ``'s'`` for spin
            order, ``'t'`` for type.
        filter_value : int or list of int
            Filter value(s) to retain (see `coherence_order_filter`,
            `spin_order_filter` and `type_filter`).
        """
        if filter_name == 'c':
            self.op, self.basis_symbols, self.basis_indices = coherence_order_filter(
                self.op, self.basis_symbols, self.basis_indices, filter_value)
        elif filter_name == 's':
            self.op, self.basis_symbols, self.basis_indices = spin_order_filter(
                self.op, self.basis_symbols, self.basis_indices, filter_value)
        elif filter_name == 't':
            self.op, self.basis_symbols, self.basis_indices = type_filter(
                self.op, self.basis_symbols, self.basis_indices, filter_value)

        # Update symbols_in and functions_in to reflect the filtering.
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

####################################################################################################
# Master equations.
####################################################################################################
def equations_of_motion(
    R: smp.MatrixBase,
    basis_op_symbols: list[smp.Expr],
    expectation_values: bool=True,
    included_operators: list[int] | None=None,
) -> smp.Eq:
    """
    Build the system of differential equations resulting from the master
    equation of the relaxation theory set in `settings.RELAXATION_THEORY`.

    Parameters
    ----------
    R : sympy.Matrix
        Relaxation superoperator matrix representation.
    basis_op_symbols : list of sympy.Symbol
        Basis operator symbols.
    expectation_values : bool, optional
        Whether to display the equations in terms of expectation values. Default is True.
    included_operators : list of int, optional
        Indices selecting a subset of basis operators to include. If None,
        all basis operators are included. Default is None.

    Returns
    -------
    sympy.Eq
        System of differential equations for the observables.
    """
    # Include only a subset of operators if desired.
    if included_operators is not None:
        R = pick_from_matrix(R, included_operators)
        basis_op_symbols = pick_from_list(basis_op_symbols, included_operators)

    # Compute the left-hand side of the differential equations, optionally as expectation values.
    if expectation_values:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(f_expectation_value_t(x), t))
    else:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(x, t))

    # Build the right-hand side according to the relaxation theory: deviations
    # from thermal equilibrium for the semiclassical theory, or the
    # operators themselves for the quantum mechanical (Lindbladian) theory.
    if settings.RELAXATION_THEORY == 'sc':
        rhs = smp.Matrix([smp.Symbol(f'\\Delta {symbol}'.replace('*', '')) for symbol in basis_op_symbols])
    elif settings.RELAXATION_THEORY == 'qm':
        rhs = smp.Matrix([symbol for symbol in basis_op_symbols])

    # Compute the right-hand side of the differential equations.
    if expectation_values:
        rhs = rhs.applyfunc(lambda x: f_expectation_value_t(x))

    rhs = -R * rhs
    return smp.Eq(lhs, rhs, evaluate=False)

def equations_of_motion_to_latex(
    eqs: smp.Eq,
    savename: str,
) -> None:
    """
    Convert a system of equations of motion to LaTeX and save it to file.

    NOTE: Saves the LaTeX source to a file in the current working directory.

    Parameters
    ----------
    eqs : sympy.Eq
        System of differential equations, as returned by `equations_of_motion`.
    savename : str
        Name used to construct the saved file name, ``EOMs_{savename}.txt``.
    """
    diff_eqs = ''
    diff_eqs += '\\begin{cases}\n'

    for lhs_i, rhs_i in zip(eqs.lhs, eqs.rhs):
        eq_latex = smp.latex(lhs_i) + '=' + smp.latex(rhs_i)
        eq_latex = eq_latex.replace('\\partial', 'd').replace('\\left|', '')\
                  .replace('\\right|', '').replace('*', '')
        diff_eqs += eq_latex + '\\\\\n'

    diff_eqs += '\\end{cases}'

    with open(f'EOMs_{savename}.txt', 'w') as file:
        file.write(diff_eqs)

####################################################################################################
# Combined functions.
####################################################################################################
def R_object_in_product_operator_basis(
    spinsystem: list[str],
    INCOHERENT_INTERACTIONS: dict,
    basis: str='T',
    sorting: str | None='v1',
    keep_non_secular: bool=False,
) -> RelaxationSuperoperator:
    """
    Compute the relaxation superoperator, converted to a product operator
    basis, and return it as a `RelaxationSuperoperator` object.

    NOTE: This is the main combined, user-facing entry point of Rela2x,
    chaining spin-operator generation (`SpinOperators`), computation of the
    relaxation superoperator (`sop_R`), and conversion to the requested
    product operator basis.

    Parameters
    ----------
    spinsystem : list of str
        Nuclear isotopes (as string labels, see `nmr_isotopes.py`) that define the spin system.
    INCOHERENT_INTERACTIONS : dict
        Dictionary of incoherent interactions (see README.md for the required format).
    basis : {'T', 'C'}, optional
        Product operator basis: ``'T'`` for spherical tensor operators, ``'C'``
        for Cartesian spin operators. Default is ``'T'``.
    sorting : str, optional
        Sorting of the basis operators, ``'v1'`` or ``'v2'``. NOTE: Only
        applicable to the spherical tensor operator basis. Default is ``'v1'``.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    R : RelaxationSuperoperator
        Relaxation superoperator object, expressed in the requested product
        operator basis and normalized to correspond to observables.

    Raises
    ------
    ValueError
        If `basis` is ``'C'`` for a spin system that is not spin-1/2
        throughout, if `keep_non_secular` is requested together with the
        quantum mechanical relaxation theory, or if `basis` is neither
        ``'T'`` nor ``'C'``.
    """
    # Create the SpinOperators object for the given spin system.
    Sops = SpinOperators(spinsystem)

    # Check that the spin system is a spin-1/2 system if the Cartesian basis is requested.
    if basis == 'C' and not all(Sops.S[i] == 1/2 for i in range(Sops.N_spins)):
        raise ValueError('Cartesian basis is only available for spin-1/2 systems.')

    # Check that the relaxation theory is semiclassical if non-secular terms are to be kept.
    if keep_non_secular and settings.RELAXATION_THEORY == 'qm':
        raise ValueError('Non-secular version of the quantum mechanical relaxation theory is not defined.')

    # Compute the matrix representation of the relaxation superoperator.
    R = sop_R(Sops, INCOHERENT_INTERACTIONS, keep_non_secular=keep_non_secular)

    if basis == 'C':
        # Compute the direct product basis of the Cartesian spin operators.
        basis_ops, symbols, norms, indices = Cartesian_product_basis_and_symbols(Sops)
    elif basis == 'T':
        # Compute the direct product basis of the spherical tensor operators.
        basis_ops, symbols, norms, indices = T_product_basis_and_symbols(Sops, sorting=sorting)
    else:
        raise ValueError("Invalid basis type. Choose 'T' for spherical tensor operators "
                         "or 'C' for Cartesian spin operators.")

    # Create the relaxation superoperator and convert it to the product basis.
    R = RelaxationSuperoperator(R, symbols, norms, indices)
    R.to_basis(basis_ops)

    print('\nFinal clean-ups...')
    # Convert the relaxation rates to correspond to observables.
    R.to_observables()

    # Expand matrix elements to simplify the expressions.
    R.op = R.op.expand()
    print('Done.')

    return R
