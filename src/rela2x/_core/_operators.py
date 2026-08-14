"""
Spin operators in Hilbert space.

The module provides the single-spin Cartesian and ladder operators in the
Zeeman basis, the classical spherical tensors and spherical tensor operators,
and the machinery for assembling many-spin product operators. The
`SpinOperators` class collects the operators of a complete spin system.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

from sympy.physics.quantum.cg import CG
import numpy as np
import sympy as smp
import sympy.physics.quantum as smpq

from rela2x._core._la import Kronecker_product, commutator
from rela2x._core._nmr_isotopes import spin_quantum_numbers
from rela2x._core._symbols import op_S_symbol, op_T_symbol


# Matrix representations:
def op_Sx(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum operator in the x-direction.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum operator ``Sx`` for spin quantum number `S`.
    """
    m = np.arange(-S, S+1)
    Sx = smp.zeros(len(m), len(m), complex=True)

    for i in range(len(m)):
        for j in range(len(m)):
            if i == j+1:
                Sx[i, j] = smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])
            elif i == j-1:
                Sx[i, j] = smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])

    return smp.Matrix(Sx.T, complex=True).applyfunc(smp.nsimplify)


def op_Sy(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum operator in the y-direction.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum operator ``Sy`` for spin quantum number `S`.
    """
    m = np.arange(-S, S+1)
    Sy = smp.zeros(len(m), len(m), complex=True)

    for i in range(len(m)):
        for j in range(len(m)):
            if i == j+1:
                Sy[i, j] = -smp.I * smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])
            elif i == j-1:
                Sy[i, j] = smp.I * smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])

    return smp.Matrix(Sy.T, complex=True).applyfunc(smp.nsimplify)


def op_Sz(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum operator in the z-direction.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum operator ``Sz`` for spin quantum number `S`.
    """
    m = np.arange(-S, S+1)
    m = np.flip(m)

    Sz = smp.zeros(len(m), len(m), complex=True)
    for i in range(len(m)):
        Sz[i, i] = m[i]

    return smp.Matrix(Sz.T, complex=True).applyfunc(smp.nsimplify)


def op_Sp(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum raising operator.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum raising operator for spin quantum number `S`.
    """
    return op_Sx(S) + smp.I * op_Sy(S)


def op_Sm(S: float) -> smp.Matrix:
    """
    Build the spin angular momentum lowering operator.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    sympy.Matrix
        Spin angular momentum lowering operator for spin quantum number `S`.
    """
    return op_Sx(S) - smp.I * op_Sy(S)


def op_Svec(S: float) -> list[smp.Matrix]:
    """
    Build the Cartesian spin angular momentum vector operator.

    Parameters
    ----------
    S : float
        Spin quantum number.

    Returns
    -------
    list of sympy.Matrix
        Cartesian spin operators ``[Sx, Sy, Sz]`` for spin quantum number `S`.
    """
    return [op_Sx(S), op_Sy(S), op_Sz(S)]


# Classical spherical tensors:
def vector_to_spherical_tensor(vector: list) -> dict:
    """
    Convert a Cartesian vector to a spherical tensor of rank 1.

    Parameters
    ----------
    vector : list
        Cartesian vector components in the form ``[x, y, z]``.

    Returns
    -------
    dict
        Dictionary of the form ``{(l, q): T_lq}`` with rank ``l = 1`` and
        projections ``q = -1, 0, 1``.
    """
    T_m1 = (vector[0] - smp.I * vector[1]) / smp.sqrt(2)
    T_0 = vector[2]
    T_p1 = -(vector[0] + smp.I * vector[1]) / smp.sqrt(2)
    return {(1, -1): T_m1, (1, 0): T_0, (1, 1): T_p1}


# Spherical tensor operators:
def op_T(
    S: float,
    l: int,
    q: int,
) -> smp.MatrixBase | int:
    """
    Build the spherical tensor operator of spin quantum number S, rank l and
    projection q, obtained by sequential lowering of the maximum-projection
    operator (see the nested `op_T_ll`).

    NOTE: The returned operators are not normalized.

    Parameters
    ----------
    S : float
        Spin quantum number.
    l : int
        Rank of the spherical tensor operator.
    q : int
        Projection of the spherical tensor operator.

    Returns
    -------
    sympy.Matrix or int
        Spherical tensor operator ``T_lq`` for spin quantum number `S`, or 0
        if `l` or `q` are outside their allowed ranges.
    """
    def op_T_ll(
        S: float,
        l: int,
    ) -> smp.MatrixBase | int:
        """
        Build the spherical tensor operator of spin quantum number S, rank l
        and maximum projection q = l.

        Parameters
        ----------
        S : float
            Spin quantum number.
        l : int
            Rank of the spherical tensor operator.

        Returns
        -------
        sympy.Matrix or int
            Spherical tensor operator ``T_ll``, or 0 if `l` exceeds ``2*S``.
        """
        if l > int(2*S):
            return 0
        else:
            return ((-1.)**l * 2.**(-l/2) * (op_Sp(S))**l).applyfunc(lambda x: smp.nsimplify(x))
    if abs(q) > l:
        return 0
    else:
        T_ll = op_T_ll(S, l)
        S_m = op_Sm(S)
        for i in range(l - q):
            comm = commutator(S_m, T_ll)
            N = smp.sqrt((l - q)*(l + q + 1))
            T_ll = (1 / N) * comm
            q += 1
        return T_ll.applyfunc(smp.simplify)


# Coupling of spherical tensor operators.
# NOTE: Used for bilinear contributions in the relaxation superoperator.
def op_T_coupled_lq(
    T1_dict: dict,
    T2_dict: dict,
    l: int,
    q: int,
) -> smp.MatrixBase:
    """
    Build the coupled spherical tensor operator of rank l and projection q,
    obtained by coupling two rank-1 spherical tensors via Clebsch-Gordan coefficients.

    Parameters
    ----------
    T1_dict : dict
        First dictionary of spherical tensor components, of the form ``{(l, q): T_lq}``.
    T2_dict : dict
        Second dictionary of spherical tensor components, of the form ``{(l, q): T_lq}``.
    l : int
        Rank of the coupled spherical tensor operator.
    q : int
        Projection of the coupled spherical tensor operator.

    Returns
    -------
    sympy.Matrix
        Coupled spherical tensor operator of rank `l` and projection `q`.
    """
    T = smp.zeros(T1_dict[1, 1].shape[0], T1_dict[1, 1].shape[0], complex=True)
    for q1 in range(-1, 2):
        for q2 in range(-1, 2):
            T += CG(1, q1, 1, q2, l, q).doit() * T1_dict[1, q1] * T2_dict[1, q2]
    return T


def many_spin_operator(
    S: list[float],
    single_spin_operator: smp.MatrixBase,
    spin_index: int,
) -> smp.MatrixBase:
    """
    Embed a single-spin operator into the full Hilbert space of a many-spin system.

    Parameters
    ----------
    S : list of float
        Spin quantum numbers of the spins in the system.
    single_spin_operator : sympy.Matrix
        Single-spin operator to be embedded.
    spin_index : int
        Index of the spin that `single_spin_operator` acts on.

    Returns
    -------
    sympy.Matrix
        Many-spin system version of `single_spin_operator`, obtained via the
        Kronecker product with unit operators on all other spins.
    """
    op = smp.eye(1)
    for i in range(len(S)):
        if i == spin_index:
            op = Kronecker_product(op, single_spin_operator)
        else:
            op = Kronecker_product(op, smp.eye(int(2*S[i]+1)))
    return op


class SpinOperators:
    """
    General class for the spin operators (Cartesian and spherical tensor) of a spin system.

    Parameters
    ----------
    spin_system : list of str
        Nuclear isotopes (as string labels, see `_nmr_isotopes.py`) that define the spin system.

    Attributes
    ----------
    S : list of float
        Spin quantum numbers of the spins.
    N_spins : int
        Number of spins in the system.
    N_states : int
        Number of Hilbert-space states in the system.
    E : list of sympy.Matrix
        Many-spin unit operator for each spin.
    E_symbol : list of sympy.physics.quantum.Operator
        Unit operator symbols for each spin.
    Sx, Sy, Sz, Sp, Sm : list of sympy.Matrix
        Many-spin Cartesian spin operators for each spin.
    Sx_symbol, Sy_symbol, Sz_symbol, Sp_symbol, Sm_symbol : list of sympy.physics.quantum.Operator
        Cartesian spin operator symbols for each spin.
    T : list of dict
        Many-spin spherical tensor operators for each spin, of the form ``{(l, q): T_lq}``.
    T_symbol : list of dict
        Spherical tensor operator symbols for each spin, of the form ``{(l, q): T_lq symbol}``.

    Raises
    ------
    ValueError
        If `spin_system` is not a list of strings.
    """
    def __init__(
        self,
        spin_system: list[str],
    ) -> None:
        """
        Initialise the spin system and generate its Cartesian and spherical
        tensor spin operators (and their symbols).

        Parameters
        ----------
        spin_system : list of str
            Nuclear isotopes (as string labels, see `_nmr_isotopes.py`) that define the spin system.
        """

        # Check that the input is a list of strings.
        if not all(isinstance(isotope, str) for isotope in spin_system):
            raise ValueError("The spin_system input has to be a list of strings corresponding to "
                             "NMR isotopes (e.g. ['1H', '13C']).")

        self.spin_system = spin_system
        self.S = spin_quantum_numbers(spin_system)

        # Determine the size of the many-spin Hilbert space.
        self.N_spins = len(self.S)
        self._gen_N_states()

        # Generate the Cartesian spin operators and their symbols.
        self._gen_many_spin_cartesian_operators()
        self._gen_cartesian_operator_symbols()

        # Generate the spherical tensor spin operators and their symbols.
        self._gen_many_spin_T_operators()
        self._gen_T_operator_symbols()

    def _gen_N_states(self) -> None:
        """
        Generate the number of Hilbert-space states in the spin system and
        store it in `self.N_states`.
        """
        self.N_states = 1
        for i in range(self.N_spins):
            self.N_states *= int(2*self.S[i] + 1)

    # Cartesian spin operators
    def _gen_many_spin_cartesian_operators(self) -> None:
        """
        Generate the many-spin Cartesian spin operators and store them in
        `self.E`, `self.Sx`, `self.Sy`, `self.Sz`, `self.Sp` and `self.Sm`.
        """
        self.E = [many_spin_operator(self.S, smp.eye(int(2*S+1)), i) for i, S in enumerate(self.S)]
        self.Sx = [many_spin_operator(self.S, op_Sx(S), i) for i, S in enumerate(self.S)]
        self.Sy = [many_spin_operator(self.S, op_Sy(S), i) for i, S in enumerate(self.S)]
        self.Sz = [many_spin_operator(self.S, op_Sz(S), i) for i, S in enumerate(self.S)]
        self.Sp = [many_spin_operator(self.S, op_Sp(S), i) for i, S in enumerate(self.S)]
        self.Sm = [many_spin_operator(self.S, op_Sm(S), i) for i, S in enumerate(self.S)]

    def _gen_cartesian_operator_symbols(self) -> None:
        """
        Generate the Cartesian spin operator symbols and store them in
        `self.E_symbol`, `self.Sx_symbol`, `self.Sy_symbol`, `self.Sz_symbol`,
        `self.Sp_symbol` and `self.Sm_symbol`.
        """
        self.E_symbol = [smpq.Operator(f'\\hat{{E}}^{{({i+1})}}') for i in range(self.N_spins)]
        self.Sx_symbol = [op_S_symbol('x', i+1) for i in range(self.N_spins)]
        self.Sy_symbol = [op_S_symbol('y', i+1) for i in range(self.N_spins)]
        self.Sz_symbol = [op_S_symbol('z', i+1) for i in range(self.N_spins)]
        self.Sp_symbol = [op_S_symbol('+', i+1) for i in range(self.N_spins)]
        self.Sm_symbol = [op_S_symbol('-', i+1) for i in range(self.N_spins)]

    # Spherical tensor operators
    def _gen_many_spin_T_operators(self) -> None:
        """
        Generate the many-spin spherical tensor operators and store them in `self.T`.
        """

        def gen_T_operators(S: float) -> dict:
            """
            Generate the single-spin spherical tensor operators for a given
            spin quantum number.

            Parameters
            ----------
            S : float
                Spin quantum number.

            Returns
            -------
            dict
                Single-spin spherical tensor operators, of the form ``{(l, q): T_lq}``.
            """
            return {(l, q): op_T(S, l, q) for l in range(int(2*S)+1) for q in range(-l, l+1)}

        self.T = [gen_T_operators(S) for S in self.S]

        # Overwrite the single-spin operators with their many-spin system versions.
        self.T = [{(l, q): many_spin_operator(self.S, T_lq, i) for (l, q), T_lq in T.items()}
                  for i, T in enumerate(self.T)]

    def _gen_T_operator_symbols(self) -> None:
        """
        Generate the spherical tensor operator symbols and store them in `self.T_symbol`.
        """
        self.T_symbol = [{(l, q): op_T_symbol(l, q, i+1) for (l, q), _ in T.items()} for i, T in enumerate(self.T)]
