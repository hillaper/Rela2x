"""
Internal core namespace of Rela2x.

This module centralises the re-export of the internal core functionality of
Rela2x. In normal use, the public package namespace should be preferred::

    from rela2x import *

Direct imports from :mod:`rela2x._core` remain possible, but they are intended
primarily for internal use, testing, and advanced development workflows.
"""

from ._settings import set_relaxation_theory, set_verbose
from ._constants import (
    hbar,
    k_B,
    mu_0,
    y_0,
    w_0,
    B,
    temperature,
    beta,
    t,
    tau,
    tau_c,
)
from ._nmr_isotopes import (
    ISOTOPES,
    spin_quantum_numbers,
)
from ._la import (
    Kronecker_product,
    commutator,
    Liouville_bracket,
    Liouville_norm,
    Liouville_amplitude,
    op_change_of_basis,
    op_decomposition,
)
from ._symbols import (
    op_S_symbol,
    product_op_S_symbol,
    op_T_symbol,
    product_op_T_symbol,
    expectation_value,
    f_expectation_value_t,
    J_coupling_symbol,
    D_coupling_symbol,
    w_symbol,
    w_Q_symbol,
)
from ._operators import (
    op_Sx,
    op_Sy,
    op_Sz,
    op_Sp,
    op_Sm,
    op_Svec,
    op_T,
    SpinOperators,
)
from ._superoperators import (
    vectorize,
    vectorize_all,
    sop_right_mul,
    sop_left_mul,
    sop_commutator,
    sop_double_commutator,
    sop_D,
    Operator,
    Superoperator,
)
from ._basis import (
    Cartesian_product_basis_and_symbols,
    T_product_basis_and_symbols,
)
from ._visualization import (
    matrix_nonzeros,
    visualize_operator,
    visualize_many_operators,
)
from ._spectral_density import (
    Lorentzian,
    Schofield_theta,
    J_w,
    J_w_isotropic_rotational_diffusion,
)
from ._relaxation import sop_R, RelaxationSuperoperator
from ._hamiltonian import (
    op_H_Z,
    op_H_J,
    op_H_RDC,
    op_H_RQC,
    op_H,
    sop_H,
    HamiltonianSuperoperator,
)
from ._liouvillian import LiouvillianSuperoperator
from ._master_equations import (
    equations_of_motion,
    equations_of_motion_to_latex,
)
from ._workflows import (
    relaxation_superoperator,
    hamiltonian_superoperator,
    liouvillian_superoperator,
)

# Explicit internal interface, mirroring the public package namespace.
__all__ = [
    "set_relaxation_theory",
    "set_verbose",
    "hbar",
    "k_B",
    "mu_0",
    "y_0",
    "w_0",
    "B",
    "temperature",
    "beta",
    "t",
    "tau",
    "tau_c",
    "ISOTOPES",
    "spin_quantum_numbers",
    "Kronecker_product",
    "commutator",
    "Liouville_bracket",
    "Liouville_norm",
    "Liouville_amplitude",
    "op_change_of_basis",
    "op_decomposition",
    "op_S_symbol",
    "product_op_S_symbol",
    "op_T_symbol",
    "product_op_T_symbol",
    "expectation_value",
    "f_expectation_value_t",
    "J_coupling_symbol",
    "D_coupling_symbol",
    "w_symbol",
    "w_Q_symbol",
    "op_Sx",
    "op_Sy",
    "op_Sz",
    "op_Sp",
    "op_Sm",
    "op_Svec",
    "op_T",
    "SpinOperators",
    "vectorize",
    "vectorize_all",
    "sop_right_mul",
    "sop_left_mul",
    "sop_commutator",
    "sop_double_commutator",
    "sop_D",
    "Cartesian_product_basis_and_symbols",
    "T_product_basis_and_symbols",
    "Operator",
    "Superoperator",
    "matrix_nonzeros",
    "visualize_operator",
    "visualize_many_operators",
    "Lorentzian",
    "Schofield_theta",
    "J_w",
    "J_w_isotropic_rotational_diffusion",
    "sop_R",
    "RelaxationSuperoperator",
    "op_H_Z",
    "op_H_J",
    "op_H_RDC",
    "op_H_RQC",
    "op_H",
    "sop_H",
    "HamiltonianSuperoperator",
    "LiouvillianSuperoperator",
    "equations_of_motion",
    "equations_of_motion_to_latex",
    "relaxation_superoperator",
    "hamiltonian_superoperator",
    "liouvillian_superoperator",
]
