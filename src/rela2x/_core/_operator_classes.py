"""
General operator and superoperator classes.

`Operator` wraps a matrix representation together with the basis symbols that
label it, and `Superoperator` extends it to Liouville space.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp

from rela2x._core._la import op_change_of_basis
from rela2x._core._superoperators import vectorize_all
from rela2x._core._visualization import visualize_operator


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
