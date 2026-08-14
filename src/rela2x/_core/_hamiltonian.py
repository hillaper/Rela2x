"""
Construction of the coherent Hamiltonian, its commutation superoperator, and
the `HamiltonianSuperoperator` class wrapping the latter.

The individual coherent interactions are built here as Hilbert-space operators
and assembled into the complete Hamiltonian by `op_H`, which `sop_H` then turns
into the Liouville-space commutation superoperator. `HamiltonianSuperoperator`
couples that matrix representation to the basis it is expressed in, and
provides the coherent counterpart of the analysis tools of
`RelaxationSuperoperator`.

NOTE: The Hamiltonian is expressed in the laboratory frame and in angular
frequency units, so that its Zeeman part is written directly in terms of the
Larmor frequency symbols that the relaxation superoperator uses for its secular
approximation (see `w_symbol`).

NOTE: The high-field secular approximation is applied here as well as in the
relaxation superoperator, and against the same Larmor frequency symbols, so
that the coherent and incoherent parts of the dynamics rest on the same
assumption. The `keep_non_secular` argument switches it off, exactly as it does
for the relaxation superoperator.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp
import time

from rela2x._core._operators import SpinOperators
from rela2x._core._status import status
from rela2x._core._superoperators import Superoperator, sop_commutator
from rela2x._core._symbols import J_coupling_symbol, w_symbol


def op_H_Z(
    spin_operators: SpinOperators,
    coupling_strengths: list[int],
) -> smp.MatrixBase:
    """
    Build the Zeeman Hamiltonian, including the chemical shift.

    NOTE: The chemical shift is carried by the Larmor frequency symbol itself
    rather than by a separate shielding or shift symbol. Chemically
    inequivalent nuclei of the same isotope are therefore distinguished by
    giving them distinct isotope labels (see `w_symbol`).

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to build the Zeeman Hamiltonian.
    coupling_strengths : list of int
        Values 1 or 0 defining which spins are included in the interaction.

    Returns
    -------
    sympy.Matrix
        Zeeman Hamiltonian ``sum_i w_i S_z^(i)`` over the included spins.
    """
    # Initialize the Zeeman Hamiltonian.
    H_Z = smp.zeros(spin_operators.N_states, spin_operators.N_states, complex=True)

    # Accumulate the single-spin Zeeman terms over the included spins.
    for spin_index, coupling_strength in enumerate(coupling_strengths):
        if coupling_strength:
            w = w_symbol(spin_operators.spin_system[spin_index])
            H_Z += w * spin_operators.Sz[spin_index]

    return H_Z


def op_H_J(
    spin_operators: SpinOperators,
    coupling_strengths_matrix: list[list[int]],
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Build the isotropic J-coupling Hamiltonian.

    NOTE: The secular approximation is applied in the high-field limit, exactly
    as it is for the relaxation superoperator. The longitudinal part
    ``S_z^i S_z^j`` commutes with the Zeeman Hamiltonian and is always
    retained, whereas the flip-flop part ``S_x^i S_x^j + S_y^i S_y^j``
    oscillates at the difference of the two Larmor frequencies and is retained
    only when that difference vanishes (or is sufficiently small) or when `keep_non_secular` is True.
    Homonuclear pairs therefore keep the
    full isotropic dot product (strong coupling), and heteronuclear pairs
    reduce to the longitudinal term alone (weak coupling).

    NOTE: The secular test is made on the same Larmor frequency symbols that
    the relaxation superoperator uses (see `w_symbol`), so that the two are
    guaranteed to make the same approximation. Spins sharing an isotope label
    are degenerate; spins carrying distinct labels are not.

    NOTE: The J-coupling constants are ordinary frequencies (in Hz) and are
    converted to angular frequency by the explicit factor of 2*pi.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to build the J-coupling Hamiltonian.
    coupling_strengths_matrix : list of list of int
        Coupling matrix whose values 1 define which spins are coupled. Only
        the upper triangle is read, so each pair contributes exactly once.
    keep_non_secular : bool, optional
        Whether to keep the non-secular flip-flop terms of heteronuclear
        pairs, giving the full isotropic dot product for every coupled pair.
        Default is False.

    Returns
    -------
    sympy.Matrix
        J-coupling Hamiltonian, ``sum_{i<j} 2*pi*J_ij S_z^i S_z^j`` for
        heteronuclear pairs and
        ``sum_{i<j} 2*pi*J_ij (S_x^i S_x^j + S_y^i S_y^j + S_z^i S_z^j)`` for
        homonuclear ones.
    """
    # Initialize the J-coupling Hamiltonian.
    H_J = smp.zeros(spin_operators.N_states, spin_operators.N_states, complex=True)

    # Accumulate the bilinear terms over the coupled spin pairs.
    # NOTE: Only the upper triangle is read, so that a coupling matrix given in
    # symmetric form contributes each pair once rather than twice.
    for spin_1_index in range(spin_operators.N_spins):
        for spin_2_index in range(spin_1_index + 1, spin_operators.N_spins):
            if coupling_strengths_matrix[spin_1_index][spin_2_index]:
                J = J_coupling_symbol(spin_1_index + 1, spin_2_index + 1)

                # The longitudinal term commutes with the Zeeman Hamiltonian and is always secular.
                dot_product = spin_operators.Sz[spin_1_index] @ spin_operators.Sz[spin_2_index]

                # Frequency at which the flip-flop term oscillates in the interaction frame.
                delta_sec = w_symbol(spin_operators.spin_system[spin_1_index])\
                            - w_symbol(spin_operators.spin_system[spin_2_index])

                # Add the flip-flop term for degenerate spins, or if non-secular terms are kept.
                if delta_sec == 0 or keep_non_secular:
                    dot_product += spin_operators.Sx[spin_1_index] @ spin_operators.Sx[spin_2_index]\
                                   + spin_operators.Sy[spin_1_index] @ spin_operators.Sy[spin_2_index]

                H_J += 2*smp.pi * J * dot_product

    return H_J


def op_H(
    spin_operators: SpinOperators,
    coherent_interactions: dict,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Build the total coherent Hamiltonian of a spin system.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to build the Hamiltonian.
    coherent_interactions : dict
        Dictionary of coherent interactions (see README.md for the required format).
    keep_non_secular : bool, optional
        Whether to keep the non-secular terms of the coherent interactions. Default is False.

    Returns
    -------
    sympy.Matrix
        Matrix representation of the coherent Hamiltonian in Hilbert space.

    Raises
    ------
    ValueError
        If `coherent_interactions` contains a mechanism other than ``'Z'`` or ``'J'``.
    """
    # Initialize the coherent Hamiltonian.
    H_final = smp.zeros(spin_operators.N_states, spin_operators.N_states, complex=True)

    # Accumulate the contribution of every requested coherent mechanism.
    for mechanism, coupling_strengths in coherent_interactions.items():

        if mechanism == 'Z':
            H_final += op_H_Z(spin_operators, coupling_strengths)

        elif mechanism == 'J':
            H_final += op_H_J(spin_operators, coupling_strengths,
                              keep_non_secular=keep_non_secular)

        else:
            raise ValueError(f"Invalid coherent interaction '{mechanism}'. Choose 'Z' for the Zeeman "
                             "interaction (including the chemical shift) or 'J' for the J-coupling "
                             "interaction.")

    return H_final


def sop_H(
    spin_operators: SpinOperators,
    coherent_interactions: dict,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Compute the matrix representation of the Hamiltonian commutation
    superoperator in Liouville space.

    NOTE: The commutation superoperator ``[H, .]`` is returned without the
    factor of -i appearing in the Liouville-von Neumann equation, so that its
    matrix elements are frequencies. That factor is applied when the coherent
    and incoherent parts are combined into the Liouvillian (see `LiouvillianSuperoperator`).

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to compute the Hamiltonian superoperator.
    coherent_interactions : dict
        Dictionary of coherent interactions (see README.md for the required format).
    keep_non_secular : bool, optional
        Whether to keep the non-secular terms of the coherent interactions. Default is False.

    Returns
    -------
    sympy.Matrix
        Matrix representation of the Hamiltonian commutation superoperator in
        Liouville space.
    """
    # Record the start time for status reporting.
    time_start = time.time()
    status('Computing H for the coherent interactions...')

    # Build the Hilbert-space Hamiltonian and turn it into its commutation superoperator.
    H = op_H(spin_operators, coherent_interactions, keep_non_secular=keep_non_secular)
    H_final = sop_commutator(H)

    status(f'H computed in {time.time() - time_start:.2f} seconds.\n')

    return H_final


class HamiltonianSuperoperator(Superoperator):
    """
    General class for the Hamiltonian commutation superoperator of a spin system.
    Inherits from `Superoperator`.

    See `Superoperator` and `Operator` for more information, including the
    basis attributes and the matrix element lookup and filtering tools that
    are shared with the other superoperators.

    NOTE: `self.op` holds the commutation superoperator ``[H, .]`` itself,
    without the factor of -i appearing in the Liouville-von Neumann equation,
    so that its matrix elements are frequencies. That factor is carried by
    `self.generator`.

    NOTE: Holds the coherent part of the dynamics. The incoherent part is held
    by `RelaxationSuperoperator`, and the two are combined by `LiouvillianSuperoperator`.

    Parameters
    ----------
    sop_H : sympy.Matrix
        Hamiltonian commutation superoperator matrix representation.
    basis_symbols : list of sympy.Symbol
        Basis operator symbols.
    basis_norms : list of sympy.Expr
        Liouville norms of the (unnormalized) basis operators.
    basis_indices : list of tuple
        Basis operator indices, used for frequency lookups and filtering.
    """
    def __init__(
        self,
        sop_H: smp.MatrixBase,
        basis_symbols: list[smp.Expr],
        basis_norms: list[smp.Expr],
        basis_indices: list[tuple],
    ) -> None:
        """
        Initialise the Hamiltonian superoperator.

        Parameters
        ----------
        sop_H : sympy.Matrix
            Hamiltonian commutation superoperator matrix representation.
        basis_symbols : list of sympy.Symbol
            Basis operator symbols.
        basis_norms : list of sympy.Expr
            Liouville norms of the (unnormalized) basis operators.
        basis_indices : list of tuple
            Basis operator indices, used for frequency lookups and filtering.
        """
        Superoperator.__init__(self, sop_H, basis_symbols, basis_norms, basis_indices)

    @property
    def generator(self) -> smp.MatrixBase:
        """
        Generator of the coherent equation of motion, ``-i [H, .]``.

        Returns
        -------
        sympy.Matrix
            Matrix representation of the generator.
        """
        return -smp.I * self.op

    def frequency(
        self,
        spin_index_op_index_1: str,
        spin_index_op_index_2: str | None=None,
    ) -> smp.Expr | None:
        """
        Get the coherent frequency between two basis operators.

        NOTE: The coherent counterpart of `RelaxationSuperoperator.rate`. See
        `Superoperator.element` for the format of the arguments.

        Parameters
        ----------
        spin_index_op_index_1 : str
            Spin index and operator index of the first basis operator.
        spin_index_op_index_2 : str, optional
            Spin index and operator index of the second basis operator. If
            None, the frequency of `spin_index_op_index_1` itself is returned.
            Default is None.

        Returns
        -------
        sympy.Expr or None
            Coherent frequency between the two operators, or None if either
            operator is not found in the basis.
        """
        return self.element(spin_index_op_index_1, spin_index_op_index_2)
