"""
Public package namespace for Rela²x.

Rela²x provides analytic and automatic high-field liquid-state NMR theory,
building the relaxation superoperator, the Hamiltonian superoperator and the
Liouvillian combining the two. The package is intended to be imported as::

    from rela2x import *

so that the functionality is available directly, as in the example notebooks.
The relaxation theory is described in::

    P. Hilla, J. Vaara, J. Magn. Reson., 2025.
    https://doi.org/10.1016/j.jmr.2024.107828
"""

# Re-export the public core functionality under the package namespace.
from rela2x._core._settings import set_relaxation_theory, set_verbose
from rela2x._core._constants import (
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
from rela2x._core._nmr_isotopes import (
    ISOTOPES,
    spin_quantum_numbers,
)
from rela2x._core._la import (
    Kronecker_product,
    commutator,
    Liouville_bracket,
    Liouville_norm,
    Liouville_amplitude,
    op_change_of_basis,
    op_decomposition,
)
from rela2x._core._symbols import (
    op_S_symbol,
    product_op_S_symbol,
    op_T_symbol,
    product_op_T_symbol,
    expectation_value,
    f_expectation_value_t,
    J_coupling_symbol,
    w_symbol,
)
from rela2x._core._operators import (
    op_Sx,
    op_Sy,
    op_Sz,
    op_Sp,
    op_Sm,
    op_Svec,
    op_T,
    SpinOperators,
)
from rela2x._core._superoperators import (
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
from rela2x._core._basis import (
    Cartesian_product_basis_and_symbols,
    T_product_basis_and_symbols,
)
from rela2x._core._visualization import (
    matrix_nonzeros,
    visualize_operator,
    visualize_many_operators,
)
from rela2x._core._spectral_density import (
    Lorentzian,
    Schofield_theta,
    J_w,
    J_w_isotropic_rotational_diffusion,
)
from rela2x._core._relaxation import sop_R, RelaxationSuperoperator
from rela2x._core._hamiltonian import (
    op_H_Z,
    op_H_J,
    op_H,
    sop_H,
    HamiltonianSuperoperator,
)
from rela2x._core._liouvillian import LiouvillianSuperoperator
from rela2x._core._master_equations import (
    equations_of_motion,
    equations_of_motion_to_latex,
)
from rela2x._core._workflows import (
    relaxation_superoperator,
    hamiltonian_superoperator,
    liouvillian_superoperator,
)

# Explicit public interface, so that "from rela2x import *" exposes the Rela2x
# names only, and not the third-party modules imported by the core.
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
    "w_symbol",
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
