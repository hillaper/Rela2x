"""
The Liouvillian superoperator class.

`LiouvillianSuperoperator` combines the coherent and incoherent parts of the dynamics into
the generator of the full equation of motion, holding the two parts separately
so that the analysis and approximation tools of each remain available.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp

from rela2x._core._hamiltonian import HamiltonianSuperoperator
from rela2x._core._relaxation import RelaxationSuperoperator
from rela2x._core._superoperators import Superoperator


class LiouvillianSuperoperator(Superoperator):
    """
    General class for the Liouvillian superoperator of a spin system.
    Inherits from `Superoperator`.

    See `Superoperator` and `Operator` for more information.

    NOTE: The main class of Rela2x for the full dynamics. The coherent and
    incoherent parts are held separately, as `self.H` and `self.R`, and are
    combined as ``L = -i [H, .] - R``. Every operation acting on the
    Liouvillian is applied to the parts and the combination is rebuilt from
    them, so that `self.op` and the parts can never disagree.

    NOTE: Combining the parts after their basis operator normalization has
    been fixed is exact, because that rescaling is a diagonal similarity
    transform and therefore distributes over the sum
    (see `Superoperator.to_observables`).

    NOTE: Both parts rest on the same high-field assumption. The relaxation
    superoperator is derived in the interaction frame of the Zeeman
    Hamiltonian, and its secular approximation is made with Zeeman frequency
    differences; the coherent part applies that same approximation, against the
    same Larmor frequency symbols, so that the J-coupling of a heteronuclear
    pair keeps only its longitudinal term (see `op_H_J`). The `keep_non_secular`
    argument switches the approximation off in both parts at once.

    NOTE: A consequence of applying the approximation consistently is that the
    coherent part commutes with the Zeeman Hamiltonian, and therefore with the
    high-temperature equilibrium state. For the semiclassical theory the
    equations of motion are written for the deviation from thermal
    equilibrium, and that separation is exact within the high-field limit.
    Retaining the non-secular terms breaks it, to the order of the ratio of the
    J-coupling to the Larmor frequency.

    Parameters
    ----------
    H : HamiltonianSuperoperator
        Coherent part of the dynamics.
    R : RelaxationSuperoperator
        Incoherent part of the dynamics.

    Attributes
    ----------
    H : HamiltonianSuperoperator
        Coherent part of the dynamics.
    R : RelaxationSuperoperator
        Incoherent part of the dynamics.
    """
    def __init__(
        self,
        H: HamiltonianSuperoperator,
        R: RelaxationSuperoperator,
    ) -> None:
        """
        Initialise the Liouvillian from its coherent and incoherent parts.

        NOTE: The two parts are assumed to be expressed in the same basis, as
        produced by `liouvillian_superoperator`. The basis attributes are taken from `R`.

        Parameters
        ----------
        H : HamiltonianSuperoperator
            Coherent part of the dynamics.
        R : RelaxationSuperoperator
            Incoherent part of the dynamics.
        """
        self.H = H
        self.R = R
        Superoperator.__init__(self, -smp.I * H.op - R.op,
                               R.basis_symbols, R.basis_norms, R.basis_indices)

    def _recombine(self) -> None:
        """
        Rebuild `self.op` and the basis attributes from the current parts.

        NOTE: Called after every operation that acts on `self.H` or `self.R`,
        so that the combination always reflects the parts it was built from.
        """
        self.op = -smp.I * self.H.op - self.R.op

        # Take the basis attributes from the relaxation part, which the filtering updates.
        self.basis_symbols = self.R.basis_symbols
        self.basis_norms = self.R.basis_norms
        self.basis_indices = self.R.basis_indices

        # Update symbols_in and functions_in to reflect the new combination.
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

    # Operations acting on both parts
    def to_basis(
        self,
        basis: list[smp.MatrixBase],
    ) -> None:
        """
        Convert both parts of the Liouvillian to a different basis.

        Parameters
        ----------
        basis : list of sympy.Matrix
            New basis operators (in Hilbert space; vectorized internally
            before the Liouville-space change of basis).
        """
        self.H.to_basis(basis)
        self.R.to_basis(basis)
        self._recombine()

    def to_observables(self) -> None:
        """
        Fix the basis operator normalization in both parts of the Liouvillian.
        """
        self.H.to_observables()
        self.R.to_observables()
        self._recombine()

    def substitute(
        self,
        substitutions_dict: dict,
    ) -> None:
        """
        Substitute symbols and functions in both parts of the Liouvillian.

        Parameters
        ----------
        substitutions_dict : dict
            Substitutions of the form ``{symbol: value}``.
        """
        self.H.substitute(substitutions_dict)
        self.R.substitute(substitutions_dict)
        self._recombine()

    def filter(
        self,
        filter_name: str,
        filter_value: int | list[int],
    ) -> None:
        """
        Filter out regions of both parts of the Liouvillian based on given criteria.

        See `Superoperator.filter` for more information.

        Parameters
        ----------
        filter_name : {'c', 's', 't'}
            Filter criterion: ``'c'`` for coherence order, ``'s'`` for spin
            order, ``'t'`` for type.
        filter_value : int or list of int
            Filter value(s) to retain.
        """
        self.H.filter(filter_name, filter_value)
        self.R.filter(filter_name, filter_value)
        self._recombine()

    # Operations acting on the relaxation part only
    def to_isotropic_rotational_diffusion(
        self,
        fast_motion_limit: bool=False,
        slow_motion_limit: bool=False,
    ) -> None:
        """
        Apply the isotropic rotational diffusion model to the relaxation part.

        See `RelaxationSuperoperator.to_isotropic_rotational_diffusion` for
        more information.

        Parameters
        ----------
        fast_motion_limit : bool, optional
            Whether to use the fast-motion limit. Default is False.
        slow_motion_limit : bool, optional
            Whether to use the slow-motion limit. Default is False.
        """
        self.R.to_isotropic_rotational_diffusion(fast_motion_limit=fast_motion_limit,
                                                 slow_motion_limit=slow_motion_limit)
        self._recombine()

    def neglect_cross_correlated_terms(
        self,
        mechanism1: str | None=None,
        mechanism2: str | None=None,
    ) -> None:
        """
        Neglect the cross-correlated terms in the relaxation part.

        See `RelaxationSuperoperator.neglect_cross_correlated_terms` for more
        information.

        Parameters
        ----------
        mechanism1 : str, optional
            Name of the first mechanism. Default is None.
        mechanism2 : str, optional
            Name of the second mechanism. If None, `mechanism1` is used. Default is None.
        """
        self.R.neglect_cross_correlated_terms(mechanism1, mechanism2)
        self._recombine()

    def neglect_cross_relaxation(self) -> None:
        """
        Neglect all cross-relaxation terms in the relaxation part.

        NOTE: Only the relaxation part is affected, so the off-diagonal
        elements of the coherent part are retained.
        """
        self.R.neglect_cross_relaxation()
        self._recombine()
