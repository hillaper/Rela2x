"""
Construction of the relaxation superoperator and the `RelaxationSuperoperator`
class wrapping it.

The individual terms of the relaxation superoperator are built here, separated
by whether the two interactions involved are single-spin (alpha) or two-spin
(beta), and assembled into the complete superoperator by `sop_R`.
`RelaxationSuperoperator` then couples that matrix representation to the basis
it is expressed in, and provides the analysis, filtering and approximation
tools acting on it.

NOTE: alpha is a single-spin interaction and beta is a two-spin interaction.
See https://doi.org/10.1016/j.jmr.2024.107828
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import product
from sympy.core.function import AppliedUndef
from sympy.physics.quantum.cg import CG
import numpy as np
import sympy as smp
import time

from rela2x._core import _settings
from rela2x._core._operators import SpinOperators, op_T_coupled_lq, vector_to_spherical_tensor
from rela2x._core._product_basis import ProductBasis, sparse_add, sparse_matmul
from rela2x._core._spectral_density import (
    G0_from_amplitudes,
    G0_from_tensors,
    J_w,
    J_w_isotropic_rotational_diffusion,
    extract_J_w_symbols_and_args,
    interaction_amplitude,
)
from rela2x._core._status import status
from rela2x._core._symbols import theta_symbol, w_symbol
from rela2x._core._superoperators import Superoperator, sop_D, sop_double_commutator
from rela2x._core._utils import interaction_labels, upper_triangle_enumerate


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
        depending on `_settings.RELAXATION_THEORY`.
    """
    if _settings.RELAXATION_THEORY == 'sc':
        return smp.Rational(1, 2) * J_w * sop_double_commutator(op_T_left.H, op_T_right)
    elif _settings.RELAXATION_THEORY == 'qm':
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
    w_s1 = w_symbol(alpha1_spin_name)
    w_s2 = w_symbol(alpha2_spin_name)

    # Dirac delta function argument for the secular approximation
    delta_sec = q*(w_s1 - w_s2)

    # Check secular approximation
    if delta_sec != 0:

        # Keep non-secular terms if specified (and semiclassical relaxation theory is used)
        if keep_non_secular and _settings.RELAXATION_THEORY == 'sc':

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
    w_s = w_symbol(alpha_spin_name)
    w_t1 = w_symbol(beta_spin_name1)
    w_t2 = w_symbol(beta_spin_name2)
    
    delta_sec = (q1+q2)*w_s - q1*w_t1 - q2*w_t2

    if delta_sec != 0:

        if keep_non_secular and _settings.RELAXATION_THEORY == 'sc':

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
    w_t1 = w_symbol(beta_spin_name1)
    w_t2 = w_symbol(beta_spin_name2)
    w_s = w_symbol(alpha_spin_name)

    delta_sec = q1*w_t1 + q2*w_t2 - (q1+q2)*w_s

    if delta_sec != 0:

        if keep_non_secular and _settings.RELAXATION_THEORY == 'sc':

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
    w1_t1 = w_symbol(beta1_spin_name1)
    w1_t2 = w_symbol(beta1_spin_name2)
    w2_t1 = w_symbol(beta2_spin_name1)
    w2_t2 = w_symbol(beta2_spin_name2)

    delta_sec = q1_t1*w1_t1 + q2_t1*w1_t2 - q1_t2*w2_t1 - q2_t2*w2_t2
    
    if delta_sec != 0:

        if keep_non_secular and _settings.RELAXATION_THEORY == 'sc':

            # Spectral density function with argument defined by the second interaction
            argument = q1_t2*w2_t1 + q2_t2*w2_t2
            J = J_w(beta1, beta2, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_t1, 1, q2_t1, l, (q1_t1+q2_t1)).doit()\
                                                        * CG(1, q1_t2, 1, q2_t2, l, (q1_t2+q2_t2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        # Spectral density function with argument defined by the second interaction
        argument = q1_t2*w2_t1 + q2_t2*w2_t2
        J = J_w(beta1, beta2, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_t1, 1, q2_t1, l, (q1_t1+q2_t1)).doit()\
                                                    * CG(1, q1_t2, 1, q2_t2, l, (q1_t2+q2_t2)).doit()


def sop_R(
    spin_operators: SpinOperators,
    incoherent_interactions: dict,
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Compute the matrix representation of the relaxation superoperator in Liouville space.

    NOTE: This is the implementation of the main equations in
    https://doi.org/10.1016/j.jmr.2024.107828, summing the alpha-alpha,
    alpha-beta, beta-alpha and beta-beta contributions (see `sop_R_term_alpha_alpha`,
    `sop_R_term_alpha_beta`, `sop_R_term_beta_alpha` and `sop_R_term_beta_beta`)
    over every pair of mechanisms in `incoherent_interactions`.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to compute the relaxation superoperator.
    incoherent_interactions : dict
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
        If `incoherent_interactions` specifies an interaction dictionary that
        is not of the format described in README.md.
    """
    # Record the start time for status reporting.
    time_start = time.time()

    # Initialize the relaxation superoperator.
    R_final = smp.zeros(spin_operators.N_states**2, spin_operators.N_states**2, complex=True)

    # Prepare coupling vector for linear interactions
    # NOTE: assumed always along z-axis (the magnetic field direction)
    T_vector = vector_to_spherical_tensor([0, 0, 1])

    status('Computing R for incoherent interaction pairs...')

    # Loop over all mechanism pairs
    for mechanism1, properties1 in incoherent_interactions.items():
        for mechanism2, properties2 in incoherent_interactions.items():

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
                                status(f'  {intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name = spin_operators.spin_system[spin_1_index]
                                spin_2_name = spin_operators.spin_system[spin_2_index]

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
                                in upper_triangle_enumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index + 1)
                                intr_name2 = mechanism2 + str(spin_2_index_i + 1) + str(spin_2_index_j + 1)
                                status(f'  {intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name = spin_operators.spin_system[spin_1_index]
                                spin_2_name_i = spin_operators.spin_system[spin_2_index_i]
                                spin_2_name_j = spin_operators.spin_system[spin_2_index_j]

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

                for (spin_1_index_i, spin_1_index_j), coupling_strength_1 \
                        in upper_triangle_enumerate(coupling_strengths_matrix1):
                    if coupling_strength_1 != 0:

                        for spin_2_index, coupling_strength_2 in enumerate(coupling_strengths2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index_i + 1) + str(spin_1_index_j + 1)
                                intr_name2 = mechanism2 + str(spin_2_index + 1)
                                status(f'  {intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name_i = spin_operators.spin_system[spin_1_index_i]
                                spin_1_name_j = spin_operators.spin_system[spin_1_index_j]
                                spin_2_name = spin_operators.spin_system[spin_2_index]

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

                for (spin_1_index_i, spin_1_index_j), coupling_strength_1 \
                        in upper_triangle_enumerate(coupling_strengths_matrix1):
                    if coupling_strength_1 != 0:

                        for (spin_2_index_i, spin_2_index_j), coupling_strength_2 \
                                in upper_triangle_enumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index_i + 1) + str(spin_1_index_j + 1)
                                intr_name2 = mechanism2 + str(spin_2_index_i + 1) + str(spin_2_index_j + 1)
                                status(f'  {intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name_i = spin_operators.spin_system[spin_1_index_i]
                                spin_1_name_j = spin_operators.spin_system[spin_1_index_j]
                                spin_2_name_i = spin_operators.spin_system[spin_2_index_i]
                                spin_2_name_j = spin_operators.spin_system[spin_2_index_j]

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
            
    status(f'R computed in {time.time() - time_start:.2f} seconds.\n')
    return R_final


@lru_cache(maxsize=None)
def _clebsch_gordan(
    q1: int,
    q2: int,
    l: int,
) -> smp.Expr:
    """
    Clebsch-Gordan coefficient <1 q1 1 q2 | l q1+q2>.

    Parameters
    ----------
    q1 : int
        Projection of the first rank-1 component.
    q2 : int
        Projection of the second rank-1 component.
    l : int
        Coupled rank.

    Returns
    -------
    sympy.Expr
        Value of the coefficient.
    """
    return CG(1, q1, 1, q2, l, q1 + q2).doit()


@lru_cache(maxsize=4096)
def _combine_exponentials(expression: smp.Expr) -> smp.Expr:
    """
    Combine a sum of exponential functions into hyperbolic functions.

    The terms of exponents x and -x are paired, a*exp(x) + b*exp(-x), which
    gives 2a*sinh(x) for b = -a and 2a*cosh(x) for b = a. This is the form
    `sympy.simplify` gives for the quantum correction factors of the Lindbladian
    theory, obtained here much faster.

    Parameters
    ----------
    expression : sympy.Expr
        Sum of terms, each a constant or a constant times one exponential function.

    Returns
    -------
    sympy.Expr
        Expression of equal value.
    """
    # Coefficient of every distinct exponent
    coefficients = defaultdict(lambda: smp.S.Zero)
    for term in smp.Add.make_args(expression):
        coefficient, exponential = term.as_independent(smp.exp, as_Add=False)
        if isinstance(exponential, smp.exp):
            coefficients[smp.expand(exponential.args[0])] += coefficient
        else:
            coefficients[smp.S.Zero] += term

    combined = []
    while coefficients:
        x, a = coefficients.popitem()

        # Constant term
        if x == 0:
            combined.append(a)
            continue

        # Pair with the term of the opposite exponent, taking x to be the one without a leading minus sign
        b = coefficients.pop(-x, smp.S.Zero)
        if x.could_extract_minus_sign():
            x, a, b = -x, b, a
        argument = smp.factor_terms(x)

        if smp.expand(a + b) == 0:
            combined.append(2*a*smp.sinh(argument))
        elif smp.expand(a - b) == 0:
            combined.append(2*a*smp.cosh(argument))
        else:
            combined.append(a*smp.exp(argument) + b*smp.exp(-argument))

    return smp.Add(*combined)


def _interaction_components(
    spin_system: list[str],
    incoherent_interactions: dict,
) -> list[dict]:
    """
    Expand every interaction into the product operators of its spherical tensor
    components.

    Parameters
    ----------
    spin_system : list of str
        Nuclear isotopes of the spin system.
    incoherent_interactions : dict
        Dictionary of incoherent interactions (see README.md for the required format).

    Returns
    -------
    list of dict
        One entry per interaction, with its `name` and its
        `components`, ``{(l, q): [(coefficient, operator, frequency), ...]}``,
        where `operator` is in the format of `ProductBasis`.

    Raises
    ------
    ValueError
        If the type of an interaction is not of the format described in README.md.
    """
    interactions = []
    for mechanism, (kind, coupling_strengths, ranks) in incoherent_interactions.items():

        # Spins taking part in each interaction of the mechanism
        if kind in ('1L', '1Q'):
            targets = [(i,) for i, strength in enumerate(coupling_strengths) if strength != 0]
        elif kind[0] == '2':
            targets = [pair for pair, strength in upper_triangle_enumerate(coupling_strengths) if strength != 0]
        else:
            raise ValueError('Invalid interaction dictionary. See README.md for details.')

        for spins in targets:
            w = [w_symbol(spin_system[spin]) for spin in spins]
            components = defaultdict(list)
            for l in set(ranks):

                # Single-spin quadratic interaction: the spherical tensor operator itself
                if kind == '1Q':
                    for q in range(-l, l+1):
                        components[l, q].append((smp.S.One, ((spins[0], l, q, False),), q*w[0]))

                # Single-spin linear interaction: coupled with the field along the z axis
                elif kind == '1L':
                    for q in range(-l, l+1):
                        if abs(q) <= 1 and _clebsch_gordan(q, 0, l) != 0:
                            components[l, q].append((_clebsch_gordan(q, 0, l), ((spins[0], 1, q, False),), q*w[0]))

                # Two-spin interaction: every pair of rank-1 components coupled to (l, q1+q2)
                else:
                    for q1, q2 in product(range(-1, 2), repeat=2):
                        if abs(q1 + q2) <= l and _clebsch_gordan(q1, q2, l) != 0:
                            operator = ((spins[0], 1, q1, False), (spins[1], 1, q2, False))
                            components[l, q1 + q2].append((_clebsch_gordan(q1, q2, l), operator, q1*w[0] + q2*w[1]))

            name = mechanism + ''.join(str(spin + 1) for spin in spins)
            interactions.append({'name': name, 'components': components})

    return interactions


def _sop_R_term_in_basis(
    product_basis: ProductBasis,
    left: tuple,
    right: tuple,
) -> dict:
    """
    Counterpart of `sop_R_term` for a pair of product operators X and Y,
    without the spectral density: [X^dagger, [Y, .]] / 2 in the semiclassical
    theory and -D(X^dagger, Y) in the Lindbladian one.

    Parameters
    ----------
    product_basis : ProductBasis
        Product operator basis.
    left : tuple
        Product operator X.
    right : tuple
        Product operator Y.

    Returns
    -------
    dict
        Columns of the term as ``{column: {row: value}}``.
    """
    # Conjugate the left operator, whose factors act on distinct spins and therefore commute
    X_dagger = tuple((spin, l, q, not conjugate) for spin, l, q, conjugate in left)

    L_X = product_basis.multiplication(X_dagger, 'left')
    R_X = product_basis.multiplication(X_dagger, 'right')
    L_Y = product_basis.multiplication(right, 'left')
    R_Y = product_basis.multiplication(right, 'right')

    if _settings.RELAXATION_THEORY == 'sc':
        commutators = sparse_matmul(sparse_add(L_X, R_X, -1), sparse_add(L_Y, R_Y, -1))
        return sparse_add({}, commutators, smp.Rational(1, 2))

    elif _settings.RELAXATION_THEORY == 'qm':
        anticommutator = sparse_add(sparse_matmul(L_Y, L_X), sparse_matmul(R_X, R_Y), 1)
        return sparse_add(sparse_add({}, sparse_matmul(L_X, R_Y), -1), anticommutator, smp.Rational(1, 2))


def sop_R_in_basis(
    spin_operators: SpinOperators,
    incoherent_interactions: dict,
    basis: str,
    basis_indices: list[tuple],
    keep_non_secular: bool=False,
) -> smp.MatrixBase:
    """
    Compute the relaxation superoperator directly in a product operator basis.

    NOTE: Gives the matrix of `sop_R` after the change to the product operator
    basis and the normalisation to observables, but much faster, because only
    the nonzero single-spin matrix elements are combined.

    Parameters
    ----------
    spin_operators : SpinOperators
        Spin system for which to compute the relaxation superoperator.
    incoherent_interactions : dict
        Dictionary of incoherent interactions (see README.md for the required format).
    basis : {'T', 'C'}
        Product operator basis.
    basis_indices : list of tuple
        Basis operator indices, in the order of the basis.
    keep_non_secular : bool, optional
        Whether to keep non-secular terms in the relaxation superoperator.
        NOTE: Only applicable for the semiclassical relaxation theory. Default is False.

    Returns
    -------
    R : sympy.Matrix
        Relaxation superoperator in the product operator basis.
    """
    # Record the start time for status reporting.
    time_start = time.time()

    product_basis = ProductBasis(spin_operators.S, basis, basis_indices)
    interactions = _interaction_components(spin_operators.spin_system, incoherent_interactions)
    keep = keep_non_secular and _settings.RELAXATION_THEORY == 'sc'

    # Matrix elements collected as {(row, column): {spectral density: coefficient}}
    elements = defaultdict(lambda: defaultdict(lambda: smp.S.Zero))
    terms = {}

    status('Computing R for incoherent interaction pairs...')

    for interaction_a, interaction_b in product(interactions, repeat=2):
        status(f"  {interaction_a['name']} * {interaction_b['name']}")

        # Commom rank, Hubbard's approximation, and equal projection
        for l, q in interaction_a['components'].keys() & interaction_b['components'].keys():
            pairs = product(interaction_a['components'][l, q], interaction_b['components'][l, q])
            for (coefficient_a, left, w_a), (coefficient_b, right, w_b) in pairs:

                # Secular approximation
                if w_a - w_b != 0 and not keep:
                    continue

                # Spectral density function with argument defined by the second interaction
                J = J_w(interaction_a['name'], interaction_b['name'], l, w_b)

                if (left, right) not in terms:
                    terms[left, right] = _sop_R_term_in_basis(product_basis, left, right)
                for column, values in terms[left, right].items():
                    for row, value in values.items():
                        elements[row, column][J] += coefficient_a * coefficient_b * value

    # Assemble the matrix
    R = smp.zeros(product_basis.dim, product_basis.dim)
    for (row, column), densities in elements.items():
        if _settings.RELAXATION_THEORY == 'sc':
            R[row, column] = smp.expand(smp.Add(*[J * c for J, c in densities.items()]))

        elif _settings.RELAXATION_THEORY == 'qm':
            # NOTE: Each J carries the quantum correction factor of its argument. The factors multiplying the same
            # spectral density function are collected and combined into hyperbolic functions.
            factors = defaultdict(lambda: smp.S.Zero)
            for J, c in densities.items():
                function, = J.atoms(AppliedUndef)
                factors[function] += smp.expand(c) * (J / function)
            R[row, column] = smp.Add(*[smp.expand(function * _combine_exponentials(factor))
                                       for function, factor in factors.items()])

    status(f'R computed in {time.time() - time_start:.2f} seconds.\n')
    return R


class RelaxationSuperoperator(Superoperator):
    """
    General class for the relaxation superoperator of a spin system.
    Inherits from `Superoperator`.

    See `Superoperator` and `Operator` for more information, including the
    basis attributes and the matrix element lookup and filtering tools that
    are shared with the other superoperators.

    NOTE: Holds the incoherent part of the dynamics. The coherent part is held
    by `HamiltonianSuperoperator`, and the two are combined by `LiouvillianSuperoperator`.

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
    spin_system : list of str, optional
        Nuclear isotopes of the spin system, used by `substitute_G0`. Default is None.
    incoherent_interactions : dict, optional
        Dictionary of incoherent interactions the superoperator was built from,
        used by `substitute_G0`. Default is None.
    """
    def __init__(
        self,
        sop_R: smp.MatrixBase,
        basis_symbols: list[smp.Expr],
        basis_norms: list[smp.Expr],
        basis_indices: list[tuple],
        spin_system: list[str] | None=None,
        incoherent_interactions: dict | None=None,
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
        spin_system : list of str, optional
            Nuclear isotopes of the spin system. Default is None.
        incoherent_interactions : dict, optional
            Dictionary of incoherent interactions. Default is None.
        """
        Superoperator.__init__(self, sop_R, basis_symbols, basis_norms, basis_indices)
        self.spin_system = spin_system
        self.incoherent_interactions = incoherent_interactions

    @property
    def generator(self) -> smp.MatrixBase:
        """
        Generator of the relaxation equation of motion, ``-R``.

        NOTE: The relaxation superoperator is positive-definite and enters the
        equations of motion with a minus sign, hence the sign here.

        Returns
        -------
        sympy.Matrix
            Matrix representation of the generator.
        """
        return -self.op

    def rate(
        self,
        spin_index_op_index_1: str,
        spin_index_op_index_2: str | None=None,
    ) -> smp.Expr | None:
        """
        Get the relaxation rate between two basis operators.

        NOTE: The incoherent counterpart of `HamiltonianSuperoperator.frequency`. See
        `Superoperator.element` for the format of the arguments.

        Parameters
        ----------
        spin_index_op_index_1 : str
            Spin index and operator index of the first basis operator.
        spin_index_op_index_2 : str, optional
            Spin index and operator index of the second basis operator. If
            None, the auto-relaxation rate of `spin_index_op_index_1` is
            returned. Default is None.

        Returns
        -------
        sympy.Expr or None
            Relaxation rate between the two operators, or None if either
            operator is not found in the basis.
        """
        return self.element(spin_index_op_index_1, spin_index_op_index_2)

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
        J_w_functions = [function for function in self.functions_in if str(function.func).startswith('J^')]
        subst_dict = {}

        # Build the substitution for every anisotropic (l > 0) spectral density function.
        for J_w in J_w_functions:
            # See extract_J_w_symbols_and_args and J_w_isotropic_rotational_diffusion.
            intrs, lq, arg = extract_J_w_symbols_and_args(J_w)
            if lq[0] > 0:
                subst_dict[J_w] = J_w_isotropic_rotational_diffusion(*intrs, lq[0], arg,
                                    fast_motion_limit=fast_motion_limit, slow_motion_limit=slow_motion_limit)
        self._replace(subst_dict)

    def substitute_G0(
        self,
        tensors: dict | None=None,
    ) -> None:
        """
        Substitute every G(0) in the relaxation superoperator with its expression
        in terms of the interaction parameters.

        NOTE: To be applied after `to_isotropic_rotational_diffusion`, which is
        what introduces the G(0) functions. Holds for a rigid molecule in an
        isotropic liquid.

        NOTE: By default every interaction is taken to be axially symmetric, and
        what it is follows from the mechanism name (see `interaction_amplitude`).
        A cross-correlation then depends on the angle between the two symmetry
        axes (see `theta_symbol`). The angles of different pairs are independent
        symbols, although the molecular geometry links them.

        NOTE: Interactions given in `tensors` use their full coupling tensor
        instead, with any asymmetry and orientation (see `G0_from_tensors`).
        Both interactions of a cross-correlation then have to be given, in the
        same molecular frame. `axial_tensor` builds the tensor of an axially
        symmetric one.

        Parameters
        ----------
        tensors : dict, optional
            Dictionary from interaction label (mechanism name followed by the
            1-based spin indices, e.g. ``'CSA1'`` or ``'DD12'``) to the 3x3
            Cartesian coupling tensor, in angular frequency units. Default is None.

        Raises
        ------
        ValueError
            If `tensors` contains an unknown label, or only one of the two
            interactions of a cross-correlation.
        """
        tensors = {} if tensors is None else tensors
        interactions = interaction_labels(self.incoherent_interactions)

        # A misspelt label would silently fall back to the default expression.
        unknown = set(tensors) - set(interactions)
        if unknown:
            raise ValueError(f'Unknown interaction labels {sorted(unknown)}. Choose from {list(interactions)}.')

        substitutions = {}
        taken_as = {}
        for G in [function for function in self.functions_in if str(function.func).startswith('G^')]:

            # Read the interaction labels and the rank from the name of the function.
            intrs, lq, _ = extract_J_w_symbols_and_args(G)
            names = intrs[0].split(', ')
            label_1, label_2, l = names[0], names[-1], lq[0]

            # Use the full coupling tensors if they are given.
            if label_1 in tensors and label_2 in tensors:
                substitutions[G] = G0_from_tensors(tensors[label_1], tensors[label_2], l)
                taken_as[label_1] = taken_as[label_2] = 'coupling tensor given'

            elif label_1 in tensors or label_2 in tensors:
                raise ValueError(f'The cross-correlation of {label_1} and {label_2} needs both coupling tensors.')

            # Otherwise build the axially symmetric expression from the mechanism names.
            else:
                amplitudes = []
                for label in (label_1, label_2):
                    mechanism, interaction_type, spin_indices = interactions[label]
                    amplitude, taken_as[label] = interaction_amplitude(
                        mechanism, interaction_type, spin_indices, self.spin_system, l)
                    amplitudes.append(amplitude)
                theta = 0 if label_1 == label_2 else theta_symbol(label_1, label_2, l)
                substitutions[G] = G0_from_amplitudes(*amplitudes, theta=theta, l=l)

        # Report what each interaction was taken to be.
        status('Substituting G(0)...')
        for label in sorted(taken_as):
            status(f'  {label}: {taken_as[label]}')
        status('')

        self._replace(substitutions)

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

        NOTE: Acts on the J(w) and G(0) functions, and therefore has to be
        applied before `substitute_G0`.

        Parameters
        ----------
        mechanism1 : str, optional
            Name of the first mechanism. Default is None.
        mechanism2 : str, optional
            Name of the second mechanism. If None, `mechanism1` is used. Default is None.
        """
        J_w_functions = [function for function in self.functions_in if str(function.func).startswith(('J^', 'G^'))]

        neglected = {}

        # Neglect all cross-correlated terms if mechanism1 and mechanism2 are both None.
        if mechanism1 is None and mechanism2 is None:
            for J_w in J_w_functions:
                # See extract_J_w_symbols_and_args.
                intrs, _, _ = extract_J_w_symbols_and_args(J_w)
                if ',' in intrs[0]:
                    neglected[J_w] = smp.S.Zero

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
                            neglected[J_w] = smp.S.Zero
                    else:
                        if mechanism1 in str(J_w) and mechanism2 in str(J_w):
                            neglected[J_w] = smp.S.Zero

        self._replace(neglected)

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
