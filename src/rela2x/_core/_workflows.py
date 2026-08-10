"""
High-level entry points combining the rest of the package.

The functions here take a spin system and its interactions and carry out the
whole construction of the relaxation superoperator in one call.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

from rela2x._core import _settings
from rela2x._core._basis import Cartesian_product_basis_and_symbols, T_product_basis_and_symbols
from rela2x._core._operators import SpinOperators
from rela2x._core._relaxation import sop_R
from rela2x._core._relaxation_superoperator import RelaxationSuperoperator


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
    if keep_non_secular and _settings.RELAXATION_THEORY == 'qm':
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
