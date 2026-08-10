"""
The relaxation superoperator class.

`RelaxationSuperoperator` couples the matrix representation of the relaxation
superoperator to the basis it is expressed in, and provides the analysis,
filtering and approximation tools acting on it.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp

from rela2x._core._basis import (
    S_index_from_string,
    T_index_from_string,
    basis_index_list_index,
    coherence_order_filter,
    spin_order_filter,
    type_filter,
)
from rela2x._core._operator_classes import Superoperator
from rela2x._core._spectral_density import (
    J_w_isotropic_rotational_diffusion,
    extract_J_w_symbols_and_args,
)


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
