"""
Basis operators for Liouville space.

The module constructs the Cartesian and spherical tensor product operator
bases, their symbols and their indices, and provides the sorting and filtering
machinery acting on those indices.

NOTE: Each basis operator is described by a tuple of (spin index, l, q) triples,
one per spin carrying a non-identity operator, ordered by spin index. The
identity operator of the whole system is described by an empty tuple. Spin
indices are 1-based, matching the operator symbols. The Cartesian basis is
described analogously by (spin index, direction) pairs.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp
import sympy.physics.quantum as smpq

from rela2x._core._la import Lv_norm
from rela2x._core._operators import SpinOperators
from rela2x._core._utils import all_combinations, cut_list, cut_matrix, list_indexes


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
