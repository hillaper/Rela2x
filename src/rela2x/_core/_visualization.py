"""
Visualization of operators and superoperators.

The routines here render the non-zero pattern of a matrix representation,
optionally annotating the rows and columns with their basis operator symbols.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import sympy as smp


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
