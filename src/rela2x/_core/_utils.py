"""
General-purpose helpers used across Rela2x.

The module contains string hashing for reproducible sorting, interaction-label
ordering, and the list and matrix operations used when picking out or cutting
away parts of the relaxation superoperator.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import hashlib
import itertools

import sympy as smp


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
