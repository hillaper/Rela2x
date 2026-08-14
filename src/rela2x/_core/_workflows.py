"""
High-level entry points combining the rest of the package.

The functions here take a spin system and its interactions and carry out the
whole construction of the relaxation superoperator, the Hamiltonian
superoperator, or the Liouvillian combining the two, in one call.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp
import time

from rela2x._core import _settings
from rela2x._core._basis import Cartesian_product_basis_and_symbols, T_product_basis_and_symbols
from rela2x._core._hamiltonian import HamiltonianSuperoperator, sop_H
from rela2x._core._liouvillian import LiouvillianSuperoperator
from rela2x._core._operators import SpinOperators
from rela2x._core._relaxation import RelaxationSuperoperator, sop_R
from rela2x._core._status import status, status_section
from rela2x._core._superoperators import Superoperator


def _product_operator_basis(
    spin_operators: SpinOperators,
    basis: str='T',
    sorting: str | None='v1',
) -> tuple[list[smp.MatrixBase], list[smp.Expr], list[smp.Expr], list[tuple]]:
    """
    Build the requested product operator basis together with its symbols,
    norms and indices.

    NOTE: Internal helper of the entry points below, holding the basis
    selection and its validity checks in one place, so that a Liouvillian can
    be built against a single basis shared by both of its parts.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to build the basis.
    basis : {'T', 'C'}, optional
        Product operator basis: ``'T'`` for spherical tensor operators, ``'C'``
        for Cartesian spin operators. Default is ``'T'``.
    sorting : str, optional
        Sorting of the basis operators, ``'v1'`` or ``'v2'``. NOTE: Only
        applicable to the spherical tensor operator basis. Default is ``'v1'``.

    Returns
    -------
    tuple
        Basis operators, their symbols, their Liouville norms and their indices.

    Raises
    ------
    ValueError
        If `basis` is ``'C'`` for a spin system that is not spin-1/2
        throughout, or if `basis` is neither ``'T'`` nor ``'C'``.
    """
    # Check that the spin system is a spin-1/2 system if the Cartesian basis is requested.
    if basis == 'C' and not all(spin_operators.S[i] == 1/2 for i in range(spin_operators.N_spins)):
        raise ValueError('Cartesian basis is only available for spin-1/2 systems.')

    if basis == 'C':
        # Compute the direct product basis of the Cartesian spin operators.
        return Cartesian_product_basis_and_symbols(spin_operators)
    elif basis == 'T':
        # Compute the direct product basis of the spherical tensor operators.
        return T_product_basis_and_symbols(spin_operators, sorting=sorting)
    else:
        raise ValueError("Invalid basis type. Choose 'T' for spherical tensor operators "
                         "or 'C' for Cartesian spin operators.")


def _to_product_operator_basis(
    superoperator: Superoperator,
    basis_ops: list[smp.MatrixBase],
) -> None:
    """
    Convert a superoperator to a product operator basis and fix its basis
    operator normalization, in place.

    NOTE: Internal helper of the entry points below, holding the clean-up
    chain they share: change of basis, normalization to observables, and
    expansion of the matrix elements.

    Parameters
    ----------
    superoperator : Superoperator
        Superoperator to convert.
    basis_ops : list of sympy.Matrix
        Product operator basis to convert to.
    """
    superoperator.to_basis(basis_ops)

    # Record the start time for status reporting.
    time_start = time.time()
    status('Final clean-ups...')

    # Convert the matrix elements to correspond to observables.
    superoperator.to_observables()

    # Expand matrix elements to simplify the expressions.
    superoperator.op = superoperator.op.expand()

    status(f'Done in {time.time() - time_start:.2f} seconds.\n')


def _relaxation_superoperator_in_basis(
    spin_operators: SpinOperators,
    incoherent_interactions: dict,
    basis_ops: list[smp.MatrixBase],
    basis_symbols: list[smp.Expr],
    basis_norms: list[smp.Expr],
    basis_indices: list[tuple],
    keep_non_secular: bool=False,
) -> RelaxationSuperoperator:
    """
    Compute the relaxation superoperator in an already-built product operator basis.

    NOTE: Internal worker of `relaxation_superoperator` and `liouvillian_superoperator`,
    taking the basis as an argument so that it is built only once when both
    parts of a Liouvillian are needed.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to compute the relaxation superoperator.
    incoherent_interactions : dict
        Dictionary of incoherent interactions (see README.md for the required format).
    basis_ops : list of sympy.Matrix
        Product operator basis.
    basis_symbols : list of sympy.Symbol
        Basis operator symbols.
    basis_norms : list of sympy.Expr
        Liouville norms of the (unnormalized) basis operators.
    basis_indices : list of tuple
        Basis operator indices.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    RelaxationSuperoperator
        Relaxation superoperator object in the given product operator basis.

    Raises
    ------
    ValueError
        If `keep_non_secular` is requested together with the quantum
        mechanical relaxation theory.
    """
    # Check that the relaxation theory is semiclassical if non-secular terms are to be kept.
    if keep_non_secular and _settings.RELAXATION_THEORY == 'qm':
        raise ValueError('Non-secular version of the quantum mechanical relaxation theory is not defined.')

    status_section('Relaxation superoperator (R)')

    # Compute the matrix representation of the relaxation superoperator.
    R = sop_R(spin_operators, incoherent_interactions, keep_non_secular=keep_non_secular)

    # Create the relaxation superoperator and convert it to the product basis.
    R = RelaxationSuperoperator(R, basis_symbols, basis_norms, basis_indices)
    _to_product_operator_basis(R, basis_ops)

    return R


def _hamiltonian_superoperator_in_basis(
    spin_operators: SpinOperators,
    coherent_interactions: dict,
    basis_ops: list[smp.MatrixBase],
    basis_symbols: list[smp.Expr],
    basis_norms: list[smp.Expr],
    basis_indices: list[tuple],
    keep_non_secular: bool=False,
) -> HamiltonianSuperoperator:
    """
    Compute the Hamiltonian superoperator in an already-built product operator basis.

    NOTE: Internal worker of `hamiltonian_superoperator` and `liouvillian_superoperator`,
    taking the basis as an argument so that it is built only once when both
    parts of a Liouvillian are needed.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to compute the Hamiltonian superoperator.
    coherent_interactions : dict
        Dictionary of coherent interactions (see README.md for the required format).
    basis_ops : list of sympy.Matrix
        Product operator basis.
    basis_symbols : list of sympy.Symbol
        Basis operator symbols.
    basis_norms : list of sympy.Expr
        Liouville norms of the (unnormalized) basis operators.
    basis_indices : list of tuple
        Basis operator indices.
    keep_non_secular : bool, optional
        Whether to keep the non-secular terms of the coherent interactions. Default is False.

    Returns
    -------
    HamiltonianSuperoperator
        Hamiltonian superoperator object in the given product operator basis.
    """
    status_section('Hamiltonian superoperator (H)')

    # Compute the matrix representation of the Hamiltonian commutation superoperator.
    H = sop_H(spin_operators, coherent_interactions, keep_non_secular=keep_non_secular)

    # Create the Hamiltonian superoperator and convert it to the product basis.
    H = HamiltonianSuperoperator(H, basis_symbols, basis_norms, basis_indices)
    _to_product_operator_basis(H, basis_ops)

    return H


def relaxation_superoperator(
    spin_system: list[str],
    incoherent_interactions: dict,
    basis: str='T',
    sorting: str | None='v1',
    keep_non_secular: bool=False,
) -> RelaxationSuperoperator:
    """
    Compute the relaxation superoperator, converted to a product operator
    basis, and return it as a `RelaxationSuperoperator` object.

    NOTE: One of the main user-facing entry points of Rela2x, chaining
    spin-operator generation (`SpinOperators`), computation of the relaxation
    superoperator (`sop_R`), and conversion to the requested product operator
    basis. See `hamiltonian_superoperator` for the coherent counterpart and
    `liouvillian_superoperator` for the two combined.

    Parameters
    ----------
    spin_system : list of str
        Nuclear isotopes (as string labels, see `_nmr_isotopes.py`) that define the spin system.
    incoherent_interactions : dict
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
    RelaxationSuperoperator
        Relaxation superoperator object, expressed in the requested product
        operator basis and normalized to correspond to observables.
    """
    # Create the SpinOperators object for the given spin system.
    Sops = SpinOperators(spin_system)

    # Build the requested product operator basis.
    basis_ops, symbols, norms, indices = _product_operator_basis(Sops, basis=basis, sorting=sorting)

    return _relaxation_superoperator_in_basis(Sops, incoherent_interactions, basis_ops, symbols,
                                              norms, indices, keep_non_secular=keep_non_secular)


def hamiltonian_superoperator(
    spin_system: list[str],
    coherent_interactions: dict,
    basis: str='T',
    sorting: str | None='v1',
    keep_non_secular: bool=False,
) -> HamiltonianSuperoperator:
    """
    Compute the Hamiltonian commutation superoperator, converted to a product
    operator basis, and return it as a `HamiltonianSuperoperator` object.

    NOTE: One of the main user-facing entry points of Rela2x, chaining
    spin-operator generation (`SpinOperators`), computation of the Hamiltonian
    superoperator (`sop_H`), and conversion to the requested product operator
    basis. See `relaxation_superoperator` for the incoherent counterpart and
    `liouvillian_superoperator` for the two combined.

    Parameters
    ----------
    spin_system : list of str
        Nuclear isotopes (as string labels, see `_nmr_isotopes.py`) that define the spin system.
    coherent_interactions : dict
        Dictionary of coherent interactions (see README.md for the required format).
    basis : {'T', 'C'}, optional
        Product operator basis: ``'T'`` for spherical tensor operators, ``'C'``
        for Cartesian spin operators. Default is ``'T'``.
    sorting : str, optional
        Sorting of the basis operators, ``'v1'`` or ``'v2'``. NOTE: Only
        applicable to the spherical tensor operator basis. Default is ``'v1'``.
    keep_non_secular : bool, optional
        Whether to keep the non-secular terms of the coherent interactions,
        i.e. the flip-flop terms of heteronuclear J-coupled pairs. Default is False.

    Returns
    -------
    HamiltonianSuperoperator
        Hamiltonian superoperator object, expressed in the requested product
        operator basis and normalized to correspond to observables.
    """
    # Create the SpinOperators object for the given spin system.
    Sops = SpinOperators(spin_system)

    # Build the requested product operator basis.
    basis_ops, symbols, norms, indices = _product_operator_basis(Sops, basis=basis, sorting=sorting)

    return _hamiltonian_superoperator_in_basis(Sops, coherent_interactions, basis_ops, symbols,
                                               norms, indices, keep_non_secular=keep_non_secular)


def liouvillian_superoperator(
    spin_system: list[str],
    coherent_interactions: dict,
    incoherent_interactions: dict,
    basis: str='T',
    sorting: str | None='v1',
    keep_non_secular: bool=False,
) -> LiouvillianSuperoperator:
    """
    Compute the Liouvillian superoperator, converted to a product operator
    basis, and return it as a `LiouvillianSuperoperator` object.

    NOTE: The main user-facing entry point of Rela2x for the full dynamics,
    building the coherent and incoherent parts against a single shared basis
    and combining them as ``L = -i [H, .] - R``. Both parts remain available
    as `L.H` and `L.R`.

    Parameters
    ----------
    spin_system : list of str
        Nuclear isotopes (as string labels, see `_nmr_isotopes.py`) that define the spin system.
    coherent_interactions : dict
        Dictionary of coherent interactions (see README.md for the required format).
    incoherent_interactions : dict
        Dictionary of incoherent interactions (see README.md for the required format).
    basis : {'T', 'C'}, optional
        Product operator basis: ``'T'`` for spherical tensor operators, ``'C'``
        for Cartesian spin operators. Default is ``'T'``.
    sorting : str, optional
        Sorting of the basis operators, ``'v1'`` or ``'v2'``. NOTE: Only
        applicable to the spherical tensor operator basis. Default is ``'v1'``.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in both the coherent and the
        incoherent part, i.e. the flip-flop terms of heteronuclear J-coupled
        pairs and the non-secular terms of the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    LiouvillianSuperoperator
        Liouvillian superoperator object, expressed in the requested product
        operator basis and normalized to correspond to observables.
    """
    # Create the SpinOperators object for the given spin system.
    Sops = SpinOperators(spin_system)

    # Build the requested product operator basis, shared by both parts.
    basis_ops, symbols, norms, indices = _product_operator_basis(Sops, basis=basis, sorting=sorting)

    # Compute the coherent and incoherent parts against that basis.
    H = _hamiltonian_superoperator_in_basis(Sops, coherent_interactions, basis_ops, symbols,
                                            norms, indices, keep_non_secular=keep_non_secular)
    R = _relaxation_superoperator_in_basis(Sops, incoherent_interactions, basis_ops, symbols,
                                           norms, indices, keep_non_secular=keep_non_secular)

    return LiouvillianSuperoperator(H, R)
