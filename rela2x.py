"""
The main spin quantum mechanics module for the Rela2x package.

Authors: 
    Perttu Hilla.
    perttu.hilla@oulu.fi
"""

####################################################################################################
# Imports.
####################################################################################################
# General
import re
import hashlib
import itertools

import numpy as np

import matplotlib.pyplot as plt

import sympy as smp
import sympy.physics.quantum as smpq
from sympy.physics.quantum.cg import CG

# Rela2x specific
import settings
from constants_and_variables import *

####################################################################################################
# Settings and modes of the program.
####################################################################################################
def set_relaxation_theory(theory):
    """
    Set the level of theory for the relaxation superoperator.
    
    Input:
        - theory: 'sc' for semiclassical, 'qm' for quantum mechanical.
    """
    settings.RELAXATION_THEORY = theory

def set_frame(frame):
    """
    Set the frame of reference.
    
    Input:
        - frame: 'lab' for laboratory frame, 'rot' for rotating frame.
    """
    settings.FRAME = frame

def set_secular(Boolean):
    """
    Set the secular approximation.
    
    Input:
        - Boolean: True for secular approximation, False for no secular approximation.
    """
    settings.SECULAR = Boolean

####################################################################################################
# Mathematical tools.
# NOTE: General tools defined here, more specific functionalities in classes defined later.
####################################################################################################
def KroneckerProduct(*m):
    """
    Symbolic Kronecker product of multiple matrices.

    Input:
        - m: Matrices to be Kronecker producted.
    """
    result = m[0]
    for i in range(1, len(m)):
        result = smp.Matrix(result.shape[0]*m[i].shape[0], result.shape[1]*m[i].shape[1],\
                 lambda p, q: result[p//m[i].shape[0], q//m[i].shape[1]] * m[i][p%m[i].shape[0], q%m[i].shape[1]])
    return result

# NOTE: ops refer to SymPy matrices.
def commutator(op1, op2):
    """Symbolic commutator of two operators."""
    return op1 * op2 - op2 * op1

# Liouville bracket and norm
# NOTE: In mathematical terms these are the Hilbert-Schmidt/Frobenius inner products.
def Lv_bracket(op1, op2):
    """Symbolic Liouville bracket of two operators."""
    return smp.trace(op1.H * op2)

def Lv_norm(op):
    """Symbolic Liouville norm of an operator."""
    return smp.sqrt(Lv_bracket(op, op))

def op_change_of_basis(op, basis):
    """
    Symbolic change of basis of an operator.
    
    Input:
        - op: Operator to be changed.
        - basis: Basis set (list of operators/matrix representations).
    """
    op_new = smp.zeros(op.shape[0], op.shape[1], complex=True)
    for i in range(op.shape[0]):
        for j in range(op.shape[1]):
            op_new[i, j] = basis[i].H * op * basis[j]
            op_new[i, j] = smp.expand(op_new[i, j])
    return op_new

####################################################################################################
# Miscellaneous tools.
# NOTE: General tools defined here, more specific functionalities in classes.
####################################################################################################
# Information extraction from spherical tensor operator symbols (the basis set that Rela2x mainly uses)
def T_symbol_spin_order(T_symbol):
    """
    Spin order of a spherical tensor operator symbol, i.e. the number of operators in the product.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).
    """
    return str(T_symbol).count('T')

def T_symbol_coherence_order(T_symbol):
    """
    Coherence order of a spherical tensor operator symbol.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).
    """
    s = str(T_symbol)
    numbers = re.findall(r'(\-?\d)}\^\{\(\d+\)\}', s)
    return sum([int(num) for num in numbers])

def T_symbol_type(T_symbol):
    """
    Type (population or coherence) of a spherical tensor operator symbol.
    NOTE: Explicitly written and kind of hard-coded.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).
    """
    s = str(T_symbol)
     # Up to rank 6 should be enough for the current purposes
    numbers = ['11', '1-1',
               '22', '21', '2-1', '2-2',
               '33', '32', '31', '3-1', '3-2', '3-3',
               '44', '43', '42', '41', '4-1', '4-2', '4-3', '4-4',
               '55', '54', '53', '52', '51', '5-1', '5-2', '5-3', '5-4', '5-5',
               '66', '65', '64', '63', '62', '61', '6-1', '6-2', '6-3', '6-4', '6-5', '6-6']
    N = sum([s.count(num) for num in numbers])
     # 0 for population, 1 for coherence
    return 1 if N > 0 else 0

def T_symbol_Nth_spin_projection(T_symbol, N):
    """
    Nth spin projection of a spherical tensor operator symbol (q-value).
    NOTE: N starts from 1.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).
    """
    s = str(T_symbol)
    pattern = r'\\hat\{T\}\_{([^}]*)}\^\{\('+str(N)+'\)\}'
    match = re.search(pattern, s)
    if match:
        return match.group(1)[1:]
    else:
        return 0

# Relaxation theory
def sort_interactions(intr1, intr2):
    """Sort an interaction pair. Used for cosmetic purposes."""
    def string_to_number(string):
        return int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16)

    if str(intr1) == str(intr2):
        return str(intr1)
    else:
        # Hash the strings and return the sorted pair, see tools.string_to_number for details.
        return sorted([str(intr1), str(intr2)], key=string_to_number)

# Basis set filtering
def pick_from_list(lst, kept_indices):
    """Select elements from a list."""
    return [lst[i] for i in kept_indices]

def pick_from_matrix(matrix, kept_indices):
    """Select rows and columns from a matrix."""
    return matrix[kept_indices, :][:, kept_indices]

def cut_list(lst, removed_indices):
    """Cut a list to a smaller size."""
    lst_loc = lst.copy()
    for index in sorted(removed_indices, reverse=True):
        lst_loc.pop(index)
    return lst_loc

def cut_matrix(matrix, removed_indices):
    """Cut a matrix to a smaller size."""
    for index in sorted(removed_indices, reverse=True):
        matrix.row_del(index)
        matrix.col_del(index)
    return matrix

def coherence_order_filter(operator, basis_states, basis_state_symbols, allowed_coherences):
    """
    Filter an operator, basis states and basis state symbols based on allowed coherences.
    Input:
        - operator: Operator to be filtered.
        - basis_states: List of basis states.
        - basis_state_symbols: List of basis state symbols.
        - allowed_coherences: List of allowed coherences.
    """
    basis_coherences = [T_symbol_coherence_order(T_symbol) for T_symbol in basis_state_symbols]
    indexes_to_delete = [i for i, coherence in enumerate(basis_coherences) if coherence not in allowed_coherences]
    unique_indexes_to_delete = list(set(indexes_to_delete))
    return cut_matrix(operator, unique_indexes_to_delete), cut_list(basis_states, unique_indexes_to_delete), cut_list(basis_state_symbols, unique_indexes_to_delete)

def spin_order_filter(operator, basis_states, basis_state_symbols, allowed_spin_orders):
    """
    Filter an operator, basis states and basis state symbols based on allowed spin orders.
    Input:
        - operator: Operator to be filtered.
        - basis_states: List of basis states.
        - basis_state_symbols: List of basis state symbols.
        - allowed_spin_orders: List of allowed spin orders.
    """
    basis_spin_orders = [T_symbol_spin_order(T_symbol) for T_symbol in basis_state_symbols]
    indexes_to_delete = [i for i, spin_order in enumerate(basis_spin_orders) if spin_order not in allowed_spin_orders]
    unique_indexes_to_delete = list(set(indexes_to_delete))
    return cut_matrix(operator, unique_indexes_to_delete), cut_list(basis_states, unique_indexes_to_delete), cut_list(basis_state_symbols, unique_indexes_to_delete)

def type_filter(operator, basis_states, basis_state_symbols, allowed_type):
    """
    Filter an operator, basis states and basis state symbols based on allowed type.
    Input:
        - operator: Operator to be filtered.
        - basis_states: List of basis states.
        - basis_state_symbols: List of basis state symbols.
        - allowed_type: Allowed type, 0 for population, 1 for coherence.
    """
    basis_types = [T_symbol_type(T_symbol) for T_symbol in basis_state_symbols]
    indexes_to_delete = [i for i, type in enumerate(basis_types) if type != allowed_type]
    unique_indexes_to_delete = list(set(indexes_to_delete))
    return cut_matrix(operator, unique_indexes_to_delete), cut_list(basis_states, unique_indexes_to_delete), cut_list(basis_state_symbols, unique_indexes_to_delete)

# Lists
def list_indexes(lst):
    """Return the indexes of a list."""
    return list(range(len(lst)))

def all_combinations(N, *args, reverse=False):
    """Generate all combinations of N lists."""
    list_combinations = list(itertools.combinations(args, N))

    # For each combination of lists, generate all combinations of one element from each list
    all_combinations = []
    for lists in list_combinations:
        all_combinations.extend(itertools.product(*lists))

    # Reverse the order of the combinations (convenient for basis sorting)
    if reverse:
        all_combinations = list(reversed(all_combinations))

    return all_combinations

####################################################################################################
# Visualization tools.
####################################################################################################
def matrix_nonzeros(matrix):
    """Nonzero elements of a matrix."""
    return matrix.applyfunc(lambda x: 1 if x != 0 else 0)

def visualize_operator(operator, rows_start=0, rows_end=None, basis_symbols=None, fontsize=None):
    """
    Visualize a given operator.

    Input:
        - operator: Operator to be visualized.
        - rows_start: Starting row index for the visualization.
        - rows_end: Ending row index for the visualization.
        - basis_symbols: Basis symbols for the visualization (effectively a legend for the basis states)
        - fontsize: Font size for the basis symbols.

    NOTE: Apart from basic plotting, mostly just for pretty visualization purposes.
    """
    operator = operator[rows_start:rows_end, rows_start:rows_end]
    operator_nonzeros = np.array(matrix_nonzeros(operator), dtype=np.float32)

    if operator_nonzeros.shape[0] <= 16:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150)
    else:
        _, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(operator_nonzeros, cmap='Blues', alpha=0.9)

    # Shift the grid
    ax.set_xticks(np.arange(-.5, operator_nonzeros.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, operator_nonzeros.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.2)

    # Move x-axis ticks to the top
    ax.xaxis.tick_top()

    # Set major ticks to start from 1
    if operator_nonzeros.shape[0] <= 16:
        ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
        ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
        ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
        ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1))
    else:
        ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 2))
        ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 2))
        ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1, 2))
        ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 2))

    # Show the basis symbols if given
    if basis_symbols is not None:
        basis_symbols = basis_symbols[rows_start:rows_end]
        basis_symbols = [f'${symbol}$'.replace('*', '').replace(' ', '') for symbol in basis_symbols]

        legend_text = '\n'.join(f'({i+1}): {label}' for i, label in enumerate(basis_symbols))

        if fontsize is None:
            ax.text(1.05, 0.5, legend_text, transform=ax.transAxes, verticalalignment='center')
        else:
            ax.text(1.05, 0.5, legend_text, transform=ax.transAxes, verticalalignment='center', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

####################################################################################################
# Symbolic operators and expectation values.
####################################################################################################
# Spin operators
def op_S_symbol(direction, index):
    """
    Symbolic spin operator
    
    Input:
        - direction: Direction of the spin operator ('x', 'y', 'z', etc.).
        - index: Spin index.
    """
    return smpq.Operator(f'\\hat{{S}}_{direction}^{{({index})}}')

def product_op_S_symbol(directions, indices):
    """
    Symbolic product of spin operators
    
    Input:
        - directions: List of directions of the spin operators.
        - indices: List of indices of the spin operators.
    """
    product_op_S = 1
    for direction, index in zip(directions, indices):
        product_op_S *= op_S_symbol(direction, index)
    return product_op_S

# Spherical tensor operators
def op_T_symbol(l, q, index):
    """
    Symbolic spherical tensor operator
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
        - index: Index of the spherical tensor operator.
    """
    return smpq.Operator(f'\\hat{{T}}_{{{l}{q}}}^{{({index})}}')

def product_op_T_symbol(ls, qs, indices):
    """
    Symbolic product of spherical tensor operators
    
    Input:
        - ls: List of ranks of the spherical tensor operators.
        - qs: List of projections of the spherical tensor operators.
        - indices: List of indices of the spherical tensor operators.
    """
    product_op_T = 1
    for l, q, index in zip(ls, qs, indices):
        product_op_T *= op_T_symbol(l, q, index)
    return product_op_T

def expectation_value(op_symbol):
    """Symbolic expectation value of an operator."""
    return smp.Symbol('\\langle ' + str(op_symbol) + '\\rangle')

def f_expectation_value_t(op_symbol):
    """Symbolic time-dependent expectation value of an operator."""
    return smp.Function(expectation_value(op_symbol))(t)

####################################################################################################
# Spin operators (Zeeeman basis).
####################################################################################################
# Matrix representations:
def op_Sx(S):
    """Spin angular momentum operator of quantum number S in x-direction."""
    m = np.arange(-S, S+1)
    Sx = smp.zeros(len(m), len(m), complex=True)

    for i in range(len(m)):
        for j in range(len(m)):
            if i == j+1:
                Sx[i, j] = smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])
            elif i == j-1:
                Sx[i, j] = smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])

    return smp.Matrix(Sx.T, complex=True).applyfunc(smp.nsimplify)

def op_Sy(S):
    """Spin angular momentum operator of quantum number S in y-direction."""
    m = np.arange(-S, S+1)
    Sy = smp.zeros(len(m), len(m), complex=True)

    for i in range(len(m)):
        for j in range(len(m)):
            if i == j+1:
                Sy[i, j] = -smp.I * smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])
            elif i == j-1:
                Sy[i, j] = smp.I * smp.Rational(1, 2) * smp.sqrt(S*(S+1) - m[i]*m[j])

    return smp.Matrix(Sy.T, complex=True).applyfunc(smp.nsimplify)

def op_Sz(S):
    """Spin angular momentum operator of quantum number S in z-direction."""
    m = np.arange(-S, S+1)
    m = np.flip(m)

    Sz = smp.zeros(len(m), len(m), complex=True)
    for i in range(len(m)):
        Sz[i, i] = m[i]

    return smp.Matrix(Sz.T, complex=True).applyfunc(smp.nsimplify)

def op_Sp(S):
    """Spin angular momentum raising operator of quantum number S."""
    return op_Sx(S) + smp.I * op_Sy(S)

def op_Sm(S):
    """Spin angular momentum lowering operator of quantum number S."""
    return op_Sx(S) - smp.I * op_Sy(S)

def op_Svec(S):
    """Spin angular momentum vector operator of quantum number S."""
    return [op_Sx(S), op_Sy(S), op_Sz(S)]

####################################################################################################
# Spherical tensors and spherical tensor operators.
# NOTE: These functions use dictionaries with keys (l, q) for spherical tensor components.
####################################################################################################
# Classical spherical tensors:
def vector_to_spherical_tensor(vector):
    """
    Convert a vector to a spherical tensor of rank 1.

    Input:
        - vector: Vector in the form [x, y, z].
    """
    T_m1 = (vector[0] - smp.I * vector[1]) / smp.sqrt(2)
    T_0 = vector[2]
    T_p1 = -(vector[0] + smp.I * vector[1]) / smp.sqrt(2)
    return {(1, -1): T_m1, (1, 0): T_0, (1, 1): T_p1}

# Spherical tensor operators:
def op_T(S, l, q):
    """
    Spherical tensor operator of spin quantum number S, rank l and projection q.
    Obtained by sequential lowering of the maximum projection operator below.

    NOTE: These are not normalized.
    """
    def op_T_ll(S, l):
        """Spherical tensor operator of spin quantum number S, rank l and maximum projection l."""
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

# Coupling of spherical tensor operators
def op_T_coupled_lq(T1_dict, T2_dict, l, q):
    """
    Coupled spherical tensor operator of rank l and projection q from two spherical tensor operators of rank 1.

    NOTE: Some ambiguity with the normalization of the scalar product (0,0) term. This is without the -sqrt(3) factor,
    which is sometimes used in the literature.

    Input:
        - T1_dict: Dictionary of spherical tensor components for the first spin.
        - T2_dict: Dictionary of spherical tensor components for the second spin.
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
    """
    T = smp.zeros(T1_dict[1, 1].shape[0], T1_dict[1, 1].shape[0], complex=True)
    for q1 in range(-1, 2):
        for q2 in range(-1, 2):
            T += CG(1, q1, 1, q2, l, q).doit() * T1_dict[1, q1] * T2_dict[1, q2]
    return T

####################################################################################################
# Superoperators for Liouville space.
# NOTE: Function inputs "op" have to be in Hilbert space, apart from "de" functions.
####################################################################################################
# Vectorizations:
def vectorize(op):
    """Vectorize a matrix."""
    return smp.Matrix(np.array(op).flatten(order='F'))

def vectorize_all(ops):
    """Vectorize a list of matrices."""
    return [vectorize(op) for op in ops]

# Liouville-space matrix representations:
def sop_rmul(op):
    """Right-multiplication superoperator."""
    return KroneckerProduct(smp.eye(op.shape[0]), op.T)

def sop_lmul(op):
    """Left-multiplication superoperator."""
    return KroneckerProduct(op, smp.eye(op.shape[0]))

def sop_commutator(op):
    """Commutator superoperator."""
    return sop_lmul(op) - sop_rmul(op)

def sop_double_commutator(op1, op2):
    """Double commutator superoperator."""
    return sop_commutator(op1) @ sop_commutator(op2)

# Dissipation superoperators (Lindbladian formalism):
def sop_D(op1, op2):
    """Lindbladian dissipation superoperator."""
    return sop_lmul(op1) @ sop_rmul(op2)\
           - smp.Rational(1, 2) * (sop_lmul(op2 @ op1) + sop_rmul(op2 @ op1))

####################################################################################################
# Spin operator classes.
####################################################################################################
def many_spin_operator(S, single_spin_operator, spin_index):
    """
    Many-spin system version of single-spin operators.
    
    Input:
        - S: List of spin quantum numbers of the spins.
        - single_spin_operator: Single-spin operator.
        - spin_index: Index of the spin.
    """
    op = smp.eye(1)
    for i in range(len(S)):
        if i == spin_index:
            op = KroneckerProduct(op, single_spin_operator)
        else:
            op = KroneckerProduct(op, smp.eye(int(2*S[i]+1)))
    return op
    
class SpinOperators:
    """
    General class for the spin operators of a spin system.
    NOTE: One of the main classes used in the Rela2x package.
    
    Input:
        - S: List of spin quantum numbers of the spins.
        - Cartesian_operators: Whether to generate the Cartesian spin operators (default = True)
        - spherical_tensors: Whether to generate the spherical tensor operators (default = True)
        
    Attributes:
        - S: Spin quantum numbers of the spins.    
        - N_spins: Number of spins in the system.
        - N_states: Number of states in the system.

        If Cartesian_operators = True:
            - Sx, Sy, Sz, Sp, Sm, Svec: Single-spin operators for each spin.
            - Sx_symbol, Sy_symbol, Sz_symbol, Sp_symbol, Sm_symbol, Svec_symbol: Single-spin operator symbols for each spin.

        If spherical_tensors = True:
            - T: Spherical tensor operators for each spin.
            - T_symbol: Spherical tensor operator symbols for each spin.
    """
    def __init__(self, S, 
                 Cartesian_operators=True,
                 spherical_tensors=True):
        self.S = S

        # Automatically generated attributes
        self.N_spins = len(S)
        self.gen_N_states()

        if Cartesian_operators:
            self.gen_many_spin_operators()
            self.gen_spin_operator_symbols()
        
        if spherical_tensors:
            self.gen_many_spin_T_operators()
            self.gen_T_operator_symbols()

    # Functions called in __init__:
    def gen_N_states(self):
        """Generate the number of states in the spin system."""
        self.N_states = 1
        for i in range(self.N_spins):
            self.N_states *= int(2*self.S[i] + 1)

    # Cartesian spin operators
    def gen_many_spin_operators(self):
        """Generate the many-spin operators"""
        def gen_spin_operators():
            """Single-spin operators for each spin if they were in a single-spin system
            (overwritten below for many-spin operators)."""
            self.Sx = [op_Sx(S) for S in self.S]
            self.Sy = [op_Sy(S) for S in self.S]
            self.Sz = [op_Sz(S) for S in self.S]
            self.Sp = [op_Sp(S) for S in self.S]
            self.Sm = [op_Sm(S) for S in self.S]

        gen_spin_operators()

        # Overwrite the operators corresponding to a single-spin system with many-spin system operators
        self.Sx = [many_spin_operator(self.S, op, i) for i, op in enumerate(self.Sx)]
        self.Sy = [many_spin_operator(self.S, op, i) for i, op in enumerate(self.Sy)]
        self.Sz = [many_spin_operator(self.S, op, i) for i, op in enumerate(self.Sz)]
        self.Sp = [many_spin_operator(self.S, op, i) for i, op in enumerate(self.Sp)]
        self.Sm = [many_spin_operator(self.S, op, i) for i, op in enumerate(self.Sm)]
        self.Svec = [[self.Sx[i], self.Sy[i], self.Sz[i]] for i in range(self.N_spins)]

    def gen_spin_operator_symbols(self):
        """Generate the spin operator symbols."""
        self.Sx_symbol = [op_S_symbol('x', i) for i in range(self.N_spins)]
        self.Sy_symbol = [op_S_symbol('y', i) for i in range(self.N_spins)]
        self.Sz_symbol = [op_S_symbol('z', i) for i in range(self.N_spins)]
        self.Sp_symbol = [op_S_symbol('+', i) for i in range(self.N_spins)]
        self.Sm_symbol = [op_S_symbol('-', i) for i in range(self.N_spins)]
        self.Svec_symbol = [f'\\hat{{S}}_^{{({i})}}' for i in range(self.N_spins)]

    # Spherical tensor operators
    def gen_many_spin_T_operators(self):
        """Generate the many-spin spherical tensor operators."""
        def gen_T_operators(S):
            """Single-spin spherical tensor operators for quantum number S."""
            return {(l, q): op_T(S, l, q) for l in range(int(2*S)+1) for q in range(-l, l+1)}
        
        self.T = [gen_T_operators(S) for S in self.S]
        
        # Overwrite the operators corresponding to a single-spin system with many-spin system operators
        self.T = [{(l, q): many_spin_operator(self.S, T_lq, i) for (l, q), T_lq in T.items()} for i, T in enumerate(self.T)]

    def gen_T_operator_symbols(self):
        """Generate the spherical tensor operator symbols."""
        self.T_symbol = [{(l, q): op_T_symbol(l, q, i+1) for (l, q), _ in T.items()} for i, T in enumerate(self.T)]

####################################################################################################
# Basis operators for Liouville space.
####################################################################################################
# Product basis of spherical tensor operators
def T_product_basis(SpinOperators, normalize=True):
    """
    Generate the product basis of spherical tensor operators.
    Each T operator of each spin is multiplied with the T operators of the other spins.

    NOTE: SpinOperators object has to have been called with spherical_tensors = True.

    Input:
        - SpinOperators: SpinOperators object.
        - normalize: Whether to normalize the basis (default = True).
    """
    S = SpinOperators.S
    N_spins = SpinOperators.N_spins
    T = SpinOperators.T

    ops = [[T[i][(l, q)] for l in range(int(2*S[i])+1) for q in range(-l, l+1)] for i in range(N_spins)]
    op_indexes = [list_indexes(ops[i]) for i in range(N_spins)]
    op_indexes = all_combinations(N_spins, *op_indexes, reverse=True)
    spin_indexes = [tuple(range(N_spins)) for _ in range(len(op_indexes))]

    T_product_basis = []
    for spin_index_tuple, op_index_tuple in zip(spin_indexes, op_indexes):
        T_product = 1
        for i, j in zip(spin_index_tuple, op_index_tuple):
            if isinstance(T_product, int):
                T_product = ops[i][j]
            else:
                T_product = T_product @ ops[i][j]
        T_product_basis.append(T_product)

    # Normalize the basis
    if normalize:
        T_product_basis = [T_product_basis[i] / Lv_norm(T_product_basis[i]) for i in range(len(T_product_basis))]
    return T_product_basis

def T_product_basis_symbols(SpinOperators):
    """
    Generate the product basis symbols of spherical tensor operators.
    Each T operator of each spin is multiplied with the T operators of the other spins.

    NOTE: SpinOperators object has to have been called with spherical_tensors = True.

    Input:
        - SpinOperators: SpinOperators object.
    """
    S = SpinOperators.S
    N_spins = SpinOperators.N_spins
    T_symbol = SpinOperators.T_symbol

    symbols = [[T_symbol[i][(l, q)] for l in range(int(2*S[i])+1) for q in range(-l, l+1)] for i in range(N_spins)]
    symbol_indexes = [list_indexes(symbols[i]) for i in range(N_spins)]
    symbol_indexes = all_combinations(N_spins, *symbol_indexes, reverse=True)
    spin_indexes = [tuple(range(N_spins)) for i in range(len(symbol_indexes))]

    T_product_basis_symbols = []
    for spin_index_tuple, symbol_index_tuple in zip(spin_indexes, symbol_indexes):
        T_product_symbol = 1
        for i, j in zip(spin_index_tuple, symbol_index_tuple):

            # Ignore the identity operator
            if not '_{00}' in str(symbols[i][j]):
                T_product_symbol *= symbols[i][j]

        # Denote the identity operator as E
        if T_product_symbol == 1:
            T_product_symbol = smpq.Operator(f'\\hat{{E}}')

        T_product_basis_symbols.append(T_product_symbol)
    return T_product_basis_symbols

# Basis set sorting
def T_basis_split_to_coherence_orders(T_product_basis, T_product_basis_symbols):
    """
    Split a basis of spherical tensor operators to groups of same coherence order.
    
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
    """
    coherence_orders = [T_symbol_coherence_order(T_symbol) for T_symbol in T_product_basis_symbols]
    T_basis_groups = {}
    T_symbol_groups = {}
    for i, coherence_order in enumerate(coherence_orders):
        if coherence_order in T_basis_groups:
            T_basis_groups[coherence_order].append(T_product_basis[i])
            T_symbol_groups[coherence_order].append(T_product_basis_symbols[i])
        else:
            T_basis_groups[coherence_order] = [T_product_basis[i]]
            T_symbol_groups[coherence_order] = [T_product_basis_symbols[i]]
    return T_basis_groups, T_symbol_groups

def spin_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols):
    """
    Sort the product basis of spherical tensor operators by spin order.
    
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
    """
    spin_orders = [T_symbol_spin_order(op) for op in T_product_basis_symbols]
    sorting = np.argsort(spin_orders)
    return [T_product_basis[i] for i in sorting], [T_product_basis_symbols[i] for i in sorting]

def coherence_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols):
    """
    Sort the product basis of spherical tensor operators by coherence order.
        
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
    """
    coherence_orders = [T_symbol_coherence_order(op) for op in T_product_basis_symbols]
    sorting = np.lexsort((coherence_orders, np.abs(coherence_orders)))
    return [T_product_basis[i] for i in sorting], [T_product_basis_symbols[i] for i in sorting]

def type_sort_T_product_basis(T_product_basis, T_product_basis_symbols):
    """
    Sort the product basis of spherical tensor operators by type.
        
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
    """
    types = [T_symbol_type(op) for op in T_product_basis_symbols]
    sorting = np.argsort(types)
    return [T_product_basis[i] for i in sorting], [T_product_basis_symbols[i] for i in sorting]

def projection_sort_T_product_basis(T_product_basis, T_product_basis_symbols, spin_index):
    """
    Sort the product basis of spherical tensor operators by projection.
        
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
        - spin_index: Spin index for the projection sorting.
    """
    T_product_basis_NEW = []
    T_product_basis_symbols_NEW = []

    T_product_basis_coherence_orders, T_product_basis_coherence_order_symbols\
        = T_basis_split_to_coherence_orders(T_product_basis, T_product_basis_symbols)

    for coherence_order, T_symbols in T_product_basis_coherence_order_symbols.items():
        projections = [T_symbol_Nth_spin_projection(T_symbol, spin_index) for T_symbol in T_symbols]
        sorting = np.argsort(projections)
        T_product_basis_NEW += [T_product_basis_coherence_orders[coherence_order][i] for i in sorting]
        T_product_basis_symbols_NEW += [T_product_basis_coherence_order_symbols[coherence_order][i] for i in sorting]

    return T_product_basis_NEW, T_product_basis_symbols_NEW

def identity_first_sort_T_product_basis(T_product_basis, T_product_basis_symbols):
    """
    Sort the product basis of spherical tensor operators by moving the identity operator first.
    
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
    """
    identity_index = [i for i, T_symbol in enumerate(T_product_basis_symbols) if '{E}' in str(T_symbol)][0]
    T_product_basis_NEW = [T_product_basis[identity_index]] + T_product_basis[:identity_index] + T_product_basis[identity_index+1:]
    T_product_basis_symbols_NEW = [T_product_basis_symbols[identity_index]] + T_product_basis_symbols[:identity_index] + T_product_basis_symbols[identity_index+1:]
    return T_product_basis_NEW, T_product_basis_symbols_NEW

# Quick sorting that combines all the above
# NOTE: Two versions are available
def quick_sort_T_product_basis(T_product_basis, T_product_basis_symbols, sorting='v1'):
    """
    Quickly sort the product basis of spherical tensor operators.
    
    Version 1 sorts by type, spin order and coherence order.
    Version 2 sorts by projection of each spin, coherence order and identity operator first.

    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
        - sorting: Sorting version ('v1' or 'v2').
    """
    if sorting == 'v1':
        T_product_basis, T_product_basis_symbols = type_sort_T_product_basis(T_product_basis, T_product_basis_symbols)
        T_product_basis, T_product_basis_symbols = spin_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols)
        T_product_basis, T_product_basis_symbols = coherence_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols)
    elif sorting == 'v2':
        # Loop over all spins and sort by projection
        for i in range(1, 10): # NOTE: This will always suffice for up to 10 spins
            T_product_basis, T_product_basis_symbols = projection_sort_T_product_basis(T_product_basis, T_product_basis_symbols, i)
        T_product_basis, T_product_basis_symbols = coherence_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols)
        T_product_basis, T_product_basis_symbols = identity_first_sort_T_product_basis(T_product_basis, T_product_basis_symbols)
    return T_product_basis, T_product_basis_symbols

def T_product_basis_and_symbols(SpinOperators, normalize=True, sorting=None):
    """
    Generate and sort the product basis of spherical tensor operators.
    
    Input:
        - SpinOperators: SpinOperators object.
        - normalize: Normalize the basis (default = True).
        - sorting: Sorting version (None, 'v1' or 'v2').
    """
    basis = T_product_basis(SpinOperators, normalize=normalize)
    symbols = T_product_basis_symbols(SpinOperators)
    if sorting == None:
        return basis, symbols
    else:
        return quick_sort_T_product_basis(basis, symbols, sorting=sorting)
    
####################################################################################################
# General operator classes.
####################################################################################################
class Operator:
    """
    General class for operators. 
    
    Takes as input:
        - op: Matrix representation of the operator.
        
    Attributes:
        - op: Matrix representation of the operator.
        - symbols_in: Symbols in the matrix representation of the operator.
        - functions_in: Functions in the matrix representation of the operator.
        
    Functions:
        - to_basis: Converts the operator to a different basis.

        - get_symbols: Get the symbols in the operator.
        - get_functions: Get the functions in the operator.
        - substitute: Substitute symbols and functions in the operator with numerical values.

        - visualize: Visualize the operator.
    """

    def __init__(self, op):
        self.op = op

        # Automatically generated attributes
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

    # Matrix algebra
    def to_basis(self, basis):
        """
        Converts the operator to a different basis.
        
        Input:
            - basis: List of new basis operators.
        """
        self.op = op_change_of_basis(self.op, basis)

    # Symbols, functions and substitutions
    def get_symbols(self):
        """Get the symbols in the operator."""
        return sorted(list(self.op.free_symbols), key=lambda x: str(x))
    
    def get_functions(self):
        """Get the functions in the operator."""
        return sorted(list(self.op.atoms(smp.Function)), key=lambda x: str(x))
    
    def substitute(self, substitutions_dict):
        """
        Substitute symbols and functions in the operator with numerical values.
        
        Input:
            - substitutions_dict: Dictionary of substitutions of the form {symbol_str: value}.
        """
        self.op = self.op.subs(substitutions_dict)

        # Update symbols_in and functions_in
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()
    
    # Visualization
    def visualize(self, rows_start=0, rows_end=None, basis_symbols=None, fontsize=None):
        """
        Visualize the operator.
        
        Input:
            - rows_start: Starting row index for the visualization.
            - rows_end: Ending row index for the visualization.
            - basis_symbols: Basis symbols for the visualization (effectively a legend for the basis states)
            - fontsize: Font size for the basis symbols.
        """
        visualize_operator(self.op, rows_start=rows_start, rows_end=rows_end, basis_symbols=basis_symbols, fontsize=fontsize)

class Superoperator(Operator):
    """
    General class for superoperators. 
    Inherits from Operator.

    See Operator class for more information.
    """
    def __init__(self, sop):
        super().__init__(sop)

    # Change of basis for superoperators
    def to_basis(self, basis):
        """
        Converts the superoperator to a different basis.
        
        Input:
            - basis: List of new basis operators.
        """
        basis_vectorized = vectorize_all(basis)
        self.op = op_change_of_basis(self.op, basis_vectorized)

####################################################################################################
# Spectral density functions and relaxation constants.
####################################################################################################
def Lorentzian(w, tau_c):
    """Lorentzian spectral density function (normalized to tau_c at w = 0)."""
    return tau_c / (1 + (w * tau_c)**2)

def Schofield_theta(w):
    """Schofield thermal correction function f(w, T) in quantum mechanical spectral density function K(w) = J(w) * f(w, T).
    NOTE: beta defined in constants_and_variables.py."""
    return smp.exp(-smp.Rational(1, 2) * beta * w)
        
def J_w(intr1, intr2, l, argument):
    """
    Spectral density function J(w).
    
    Input:
        - intr1: String for the first interaction.
        - intr2: String for the second interaction.
        - l: Rank of the spherical tensor operator.
        - argument: Argument of the spectral density function (combination of angular frequencies).
    """
    intr_sorted = sort_interactions(intr1, intr2)
    if isinstance(intr_sorted, str): # NOTE: See sort_interactions
        expr = smp.Function(f'J^{{{intr_sorted}}}_{{{l, 0}}}')(smp.Abs(argument))
    else:
        expr = smp.Function(f'J^{{{intr_sorted[0]}, {intr_sorted[1]}}}_{{{l, 0}}}')(smp.Abs(argument))

    if settings.RELAXATION_THEORY == 'sc':
        return expr
    elif settings.RELAXATION_THEORY == 'qm':
        f_theta = Schofield_theta(argument)
        return expr * f_theta
    
def J_w_isotropic_rotational_diffusion(intr1, intr2, l, argument, tau_c,
                                       fast_motion_limit=False, slow_motion_limit=False):
    """
    Isotropic rotational diffusion spectral density function, which is a Lorentzian function.
    
    Input:
        - intr1: String for the first interaction.
        - intr2: String for the second interaction.
        - l: Rank of the spherical tensor operator.
        - argument: Argument of the spectral density function (combination of angular frequencies).
        - tau_c: Correlation time of the isotropic rotational diffusion.
        - fast_motion_limit: Whether to use the fast motion limit where (w * tau_c) << 1. Default is False.
        - slow_motion_limit: Whether to use the slow motion limit where (w * tau_c) >> 1. Default is False.
    """
    intr_sorted = sort_interactions(intr1, intr2)
    if isinstance(intr_sorted, str): # NOTE: See sort_interactions
        G = smp.Symbol(f'G^{{{intr_sorted}}}_{{{l, 0}}}', real=True)
    else:
        G = smp.Symbol(f'G^{{{intr_sorted[0]}, {intr_sorted[1]}}}_{{{l, 0}}}', real=True)

    if fast_motion_limit:
        J_w = 2*G * tau_c
    elif slow_motion_limit:
        if argument == 0:
            J_w = 2*G * tau_c
        else:
            J_w = 0
    else:
        J_w = 2*G * Lorentzian(argument, tau_c)

    if settings.RELAXATION_THEORY == 'sc':
        return J_w
    elif settings.RELAXATION_THEORY == 'qm':
        f_theta = Schofield_theta(argument)
        return J_w * f_theta
    
# Helper functions for RelaxationSuperoperator object
def extract_J_w_symbols_and_args(J):
    """
    Extract the symbols and arguments of the spectral density function J_w.
    Used for substitution of symbols and functions in the spectral density function.

    Input:
        - J: Spectral density function symbol.
    """
    J_str = str(J.func)

    intrs = re.findall(r'\^{(.*?)}', J_str)[0]
    lq = re.findall(r'\_\{\((.*?)\)\}', J_str)[0]

    intrs = tuple(intrs.split(', ')) if isinstance(intrs, tuple) else tuple([intrs, intrs])
    lq = tuple(map(int, lq.split(', ')))
    arg = J.args[0]

    return intrs, lq, arg
    
####################################################################################################
# Relaxation superoperators, high-field liquid-state NMR.
# NOTE: sigma is single-spin interaction and delta is double-spin interaction.
####################################################################################################
def sop_R_term(op_T_left, J_w, op_T_right):
    """
    Term in the relaxation superoperator.
    NOTE: The left and right "order" of the operators is just for bookkeeping purposes.

    Input:
        - op_T_left: Left spherical tensor operator.
        - J_w: Spectral density function.
        - op_T_right: Right spherical tensor operator.
    """
    if settings.RELAXATION_THEORY == 'sc':
        return smp.Rational(1, 2) * J_w * sop_double_commutator(op_T_left, op_T_right.H)
    elif settings.RELAXATION_THEORY == 'qm':
        return -J_w * sop_D(op_T_left, op_T_right.H)
    
# Laboratory frame:
def sop_R_term_sigma_sigma_LAB(l, q, sigma1, sigma2, sigma2_spin_name,
                               op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between two single-spin interactions.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
        - sigma1: Name of the first single-spin interaction.
        - sigma2: Name of the second single-spin interaction.
        - sigma2_spin_name: Name of the second spin associated with the second single-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w = smp.Symbol(f'\\omega_{{{sigma2_spin_name}}}', real=True)
    argument = q*w
    J = J_w(sigma1, sigma2, l, argument)
    return sop_R_term(op_T_left, J, op_T_right)

def sop_R_term_sigma_delta_LAB(l, q1, q2, sigma, delta, delta_spin_name1, delta_spin_name2,
                               op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between a single-spin interaction and a double-spin interaction.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q1: Projection 1 of the coupled spherical tensor operator.
        - q2: Projection 2 of the coupled spherical tensor operator.
        - sigma: Name of the single-spin interaction.
        - delta: Name of the double-spin interaction.
        - delta_spin_name1: Name of the first spin associated with the double-spin interaction.
        - delta_spin_name2: Name of the second spin associated with the double-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w1 = smp.Symbol(f'\\omega_{{{delta_spin_name1}}}', real=True)
    w2 = smp.Symbol(f'\\omega_{{{delta_spin_name2}}}', real=True)
    argument = q1*w1 + q2*w2
    J = J_w(sigma, delta, l, argument)
    return (sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit())

def sop_R_term_delta_sigma_LAB(l, q, delta, sigma, sigma_spin_name,
                               op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between a double-spin interaction and a single-spin interaction.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
        - delta: Name of the double-spin interaction.
        - sigma: Name of the single-spin interaction.
        - sigma_spin_name: Name of the spin associated with the single-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w = smp.Symbol(f'\\omega_{{{sigma_spin_name}}}', real=True)
    argument = q*w
    J = J_w(delta, sigma, l, argument)
    return sop_R_term(op_T_left, J, op_T_right)

def sop_R_term_delta_delta_LAB(l, q1, q2, delta1, delta2, delta2_spin_name1, delta2_spin_name2,
                                 op_T_left, op_T_right):
     """
     Term in the relaxation superoperator between two double-spin interactions.
     
     Input:
         - l: Rank of the spherical tensor operator.
         - q1: Projection 1 of the coupled spherical tensor operator.
         - q2: Projection 2 of the coupled spherical tensor operator.
         - delta1: Name of the first double-spin interaction.
         - delta2: Name of the second double-spin interaction.
         - delta2_spin_name1: Name of the first spin associated with the second double-spin interaction.
         - delta2_spin_name2: Name of the second spin associated with the second double-spin interaction.
         - op_T_left: Left spherical tensor operator.
         - op_T_right: Right spherical tensor operator.
        """
     w1 = smp.Symbol(f'\\omega_{{{delta2_spin_name1}}}', real=True)
     w2 = smp.Symbol(f'\\omega_{{{delta2_spin_name2}}}', real=True)
     argument = q1*w1 + q2*w2
     J = J_w(delta1, delta2, l, argument)
     return (sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit())

# Rotating frame:
def sop_R_term_sigma_sigma_ROT(l, q, sigma1, sigma2, sigma1_spin_name, sigma2_spin_name,
                               op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between two single-spin interactions in the rotating frame.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
        - sigma1: Name of the first single-spin interaction.
        - sigma2: Name of the second single-spin interaction.
        - sigma1_spin_name: Name of the first spin associated with the first single-spin interaction.
        - sigma2_spin_name: Name of the second spin associated with the second single-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w2 = smp.Symbol(f'\\omega_{{{sigma2_spin_name}}}', real=True)
    argument = q*w2
    J = J_w(sigma1, sigma2, l, argument)

    w1 = smp.Symbol(f'\\omega_{{{sigma1_spin_name}}}', real=True)
    w = -w1 + w2
    exp = smp.exp(smp.I*q*w*t)
    if settings.SECULAR:
        if w != 0:
            exp = 0

    return sop_R_term(op_T_left, J, op_T_right) * exp

def sop_R_term_sigma_delta_ROT(l, q1, q2, sigma, delta, sigma_spin_name, delta_spin_name1, delta_spin_name2,
                                op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between a single-spin interaction and a double-spin interaction in the rotating frame.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q1: Projection 1 of the coupled spherical tensor operator.
        - q2: Projection 2 of the coupled spherical tensor operator.
        - sigma: Name of the single-spin interaction.
        - delta: Name of the double-spin interaction.
        - sigma_spin_name: Name of the spin associated with the single-spin interaction.
        - delta_spin_name1: Name of the first spin associated with the double-spin interaction.
        - delta_spin_name2: Name of the second spin associated with the double-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w_d1 = smp.Symbol(f'\\omega_{{{delta_spin_name1}}}', real=True)
    w_d2 = smp.Symbol(f'\\omega_{{{delta_spin_name2}}}', real=True)
    argument = q1*w_d1 + q2*w_d2
    J = J_w(sigma, delta, l, argument)

    w_s = smp.Symbol(f'\\omega_{{{sigma_spin_name}}}', real=True)
    w = -(q1+q2)*w_s + q1*w_d1 + q2*w_d2
    exp = smp.exp(smp.I*w*t)
    if settings.SECULAR:
        if w != 0:
            exp = 0

    return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit() * exp

def sop_R_term_delta_sigma_ROT(l, q1, q2, delta, sigma, delta_spin_name1, delta_spin_name2, sigma_spin_name,
                                op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between a double-spin interaction and a single-spin interaction in the rotating frame.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q1: Projection 1 of the coupled spherical tensor operator.
        - q2: Projection 2 of the coupled spherical tensor operator.
        - delta: Name of the double-spin interaction.
        - sigma: Name of the single-spin interaction.
        - delta_spin_name1: Name of the first spin associated with the double-spin interaction.
        - delta_spin_name2: Name of the second spin associated with the double-spin interaction.
        - sigma_spin_name: Name of the spin associated with the single-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w_s = smp.Symbol(f'\\omega_{{{sigma_spin_name}}}', real=True)
    argument = (q1+q2)*w_s
    J = J_w(delta, sigma, l, argument)

    w_d1 = smp.Symbol(f'\\omega_{{{delta_spin_name1}}}', real=True)
    w_d2 = smp.Symbol(f'\\omega_{{{delta_spin_name2}}}', real=True)
    w = -q1*w_d1 - q2*w_d2 + (q1+q2)*w_s
    exp = smp.exp(smp.I*w*t)
    if settings.SECULAR:
        if w != 0:
            exp = 0

    return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit() * exp

def sop_R_term_delta_delta_ROT(l, q1_d1, q2_d1, q1_d2, q2_d2, delta1, delta2, delta1_spin_name1, delta1_spin_name2, delta2_spin_name1, delta2_spin_name2,
                                op_T_left, op_T_right):
    """
    Term in the relaxation superoperator between two double-spin interactions in the rotating frame.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q1_d1: Projection 1 of the first coupled spherical tensor operator.
        - q2_d1: Projection 2 of the first coupled spherical tensor operator.
        - q1_d2: Projection 1 of the second coupled spherical tensor operator.
        - q2_d2: Projection 2 of the second coupled spherical tensor operator.
        - delta1: Name of the first double-spin interaction.
        - delta2: Name of the second double-spin interaction.
        - delta1_spin_name1: Name of the first spin associated with the first double-spin interaction.
        - delta1_spin_name2: Name of the second spin associated with the first double-spin interaction.
        - delta2_spin_name1: Name of the first spin associated with the second double-spin interaction.
        - delta2_spin_name2: Name of the second spin associated with the second double-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.
    """
    w_d1_1 = smp.Symbol(f'\\omega_{{{delta1_spin_name1}}}', real=True)
    w_d1_2 = smp.Symbol(f'\\omega_{{{delta1_spin_name2}}}', real=True)
    argument = q1_d1*w_d1_1 + q2_d1*w_d1_2
    J = J_w(delta1, delta2, l, argument)

    w_d2_1 = smp.Symbol(f'\\omega_{{{delta2_spin_name1}}}', real=True)
    w_d2_2 = smp.Symbol(f'\\omega_{{{delta2_spin_name2}}}', real=True)
    w = -q1_d1*w_d1_1 - q2_d1*w_d1_2 + q1_d2*w_d2_1 + q2_d2*w_d2_2
    exp = smp.exp(smp.I*w*t)
    if settings.SECULAR:
        if w != 0:
            exp = 0

    return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_d1, 1, q2_d1, l, (q1_d1+q2_d1)).doit() * CG(1, q1_d2, 1, q2_d2, l, (q1_d2+q2_d2)).doit() * exp

def sop_R(SpinOperators, INCOHERENT_INTERACTIONS, spin_names=None):
    """
    Symbolic relaxation superoperator. 
    NOTE: This is the main equation in the referenced publication.
    
    Input:
        - SpinOperators: SpinOperators object.
        - INCOHERENT_INTERACTIONS: Dictionary of incoherent interactions.
        - spin_names: List of spin names to handle chemically equivalent spins (default = None).
    """
    # Initialize the relaxation superoperator
    R_final = smp.zeros(SpinOperators.N_states**2, SpinOperators.N_states**2, complex=True)

    # Prepare coupling vector for linear interactions
    # NOTE: always along z-axis
    T_vector = vector_to_spherical_tensor([0, 0, 1])

    # Loop over all mechanism pairs
    for mechanism1, properties1 in INCOHERENT_INTERACTIONS.items():
        for mechanism2, properties2 in INCOHERENT_INTERACTIONS.items():

            # Single-spin single-spin mechanism pair
            if properties1[0][0] == 'S' and properties2[0][0] == 'S':
                print('\nComputing R for single-spin single-spin mechanism pairs:')

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
                                intr_name1 = mechanism1 + str(spin_1_index)
                                intr_name2 = mechanism2 + str(spin_2_index)
                                print(f'[{intr_name1} * {intr_name2}]')

                                # Handle chemically equivalent spins (homonuclear systems)
                                if spin_names is not None:
                                    for i, name in enumerate(spin_names):
                                        if i == spin_1_index:
                                            spin_1_name = name
                                        if i == spin_2_index:
                                            spin_2_name = name
                                else:
                                    spin_1_name = spin_1_index
                                    spin_2_name = spin_2_index

                                # Loop over all common ranks and components
                                for l in ls:
                                    for q in range(-l, l+1):

                                        # Check if linear or quadratic interaction
                                        if properties1[0][1] == 'L':
                                            T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index], T_vector, l, q)
                                        elif properties1[0][1] == 'Q':
                                            T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index], SpinOperators.T[spin_1_index], l, q)

                                        if properties2[0][1] == 'L':
                                            T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], T_vector, l, q)
                                        elif properties2[0][1] == 'Q':
                                            T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], SpinOperators.T[spin_2_index], l, q)

                                        # Compute the relaxation superoperator term
                                        if settings.FRAME == 'lab':
                                            R_term = sop_R_term_sigma_sigma_LAB(l, q, intr_name1, intr_name2, spin_2_name, T_left, T_right)
                                        elif settings.FRAME == 'rot':
                                            R_term = sop_R_term_sigma_sigma_ROT(l, q, intr_name1, intr_name2, spin_1_name, spin_2_name, T_left, T_right)

                                        # Add the relaxation superoperator term to the relaxation superoperator
                                        R_final += R_term

            # Single-spin double-spin mechanism pair
            elif properties1[0][0] == 'S' and properties2[0][0] == 'D':
                print('\nComputing R for single-spin double-spin mechanism pairs:')
                
                # List of coupling strengths and coupling-strength matrix
                coupling_strengths1 = properties1[1]
                coupling_strengths_matrix2 = properties2[1]

                ranks1 = properties1[2]
                ranks2 = properties2[2]

                ls = list(set(ranks1) & set(ranks2))

                # Loop over all interactions pairs (now including the coupling-strength matrix)
                for spin_1_index, coupling_strength_1 in enumerate(coupling_strengths1):
                    if coupling_strength_1 != 0:

                        for (spin_2_index_i, spin_2_index_j), coupling_strength_2 in np.ndenumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index)
                                intr_name2 = mechanism2 + str(spin_2_index_i) + str(spin_2_index_j)
                                print(f'[{intr_name1} * {intr_name2}]')

                                # Handle chemically equivalent spins (homonuclear systems)
                                if spin_names is not None:
                                    for i, name in enumerate(spin_names):
                                        if i == spin_1_index:
                                            spin_1_name = name
                                        if i == spin_2_index_i:
                                            spin_2_name_i = name
                                        if i == spin_2_index_j:
                                            spin_2_name_j = name
                                else:
                                    spin_1_name = spin_1_index
                                    spin_2_name_i = spin_2_index_i
                                    spin_2_name_j = spin_2_index_j

                                for l in ls:
                                    # Loop over q1 and q2 values in the symbolic case
                                    for q1 in range(-1, 2):
                                        for q2 in range(-1, 2):

                                            # Clebsch-Gordan restriction
                                            if np.abs(q1 + q2) <= l:

                                                if properties1[0][1] == 'L':
                                                    T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index], T_vector, l, (q1 + q2))
                                                elif properties1[0][1] == 'Q':
                                                    T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index], SpinOperators.T[spin_1_index], l, (q1 + q2))

                                                # Interaction 2 is always bilinear if type is 'D'
                                                T_right_i = SpinOperators.T[spin_2_index_i]
                                                T_right_j = SpinOperators.T[spin_2_index_j]
                                                T_right = T_right_i[1, q1] @ T_right_j[1, q2]

                                                if settings.FRAME == 'lab':
                                                    R_term = sop_R_term_sigma_delta_LAB(l, q1, q2, intr_name1, intr_name2, spin_2_name_i, spin_2_name_j, T_left, T_right)
                                                elif settings.FRAME == 'rot':
                                                    R_term = sop_R_term_sigma_delta_ROT(l, q1, q2, intr_name1, intr_name2, spin_1_name, spin_2_name_i, spin_2_name_j, T_left, T_right)

                                                R_final += R_term

            # Double-spin single-spin mechanism pair
            elif properties1[0][0] == 'D' and properties2[0][0] == 'S':
                print('\nComputing R for double-spin single-spin mechanism pairs:')
                    
                coupling_strengths_matrix1 = properties1[1]
                coupling_strengths2 = properties2[1]

                ranks1 = properties1[2]
                ranks2 = properties2[2]

                ls = list(set(ranks1) & set(ranks2))

                for (spin_1_index_i, spin_1_index_j), coupling_strength_1 in np.ndenumerate(coupling_strengths_matrix1):
                    if coupling_strength_1 != 0:

                        for spin_2_index, coupling_strength_2 in enumerate(coupling_strengths2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index_i) + str(spin_1_index_j)
                                intr_name2 = mechanism2 + str(spin_2_index)
                                print(f'[{intr_name1} * {intr_name2}]')

                                # Handle chemically equivalent spins (homonuclear systems)
                                if spin_names is not None:
                                    for i, name in enumerate(spin_names):
                                        if i == spin_1_index_i:
                                            spin_1_name_i = name
                                        if i == spin_1_index_j:
                                            spin_1_name_j = name
                                        if i == spin_2_index:
                                            spin_2_name = name
                                else:
                                    spin_1_name_i = spin_1_index_i
                                    spin_1_name_j = spin_1_index_j
                                    spin_2_name = spin_2_index

                                for l in ls:

                                    # Different cases for laboaratory and rotating frames
                                    if settings.FRAME == 'lab':                                        
                                        for q in range(-1, 2):
                                            
                                            T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index_i], SpinOperators.T[spin_1_index_j], l, q)

                                            if properties2[0][1] == 'L':
                                                T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], T_vector, l, q)
                                            elif properties2[0][1] == 'Q':
                                                T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], SpinOperators.T[spin_2_index], l, q)

                                            R_term = sop_R_term_delta_sigma_LAB(l, q, intr_name1, intr_name2, spin_2_name, T_left, T_right)
                                            R_final += R_term

                                    elif settings.FRAME == 'rot':
                                        for q1 in range(-1, 2):
                                            for q2 in range(-1, 2):

                                                if np.abs(q1 + q2) <= l:
                                                
                                                    T_left_i = SpinOperators.T[spin_1_index_i]
                                                    T_left_j = SpinOperators.T[spin_1_index_j]
                                                    T_left = T_left_i[1, q1] @ T_left_j[1, q2]

                                                    if properties2[0][1] == 'L':
                                                        T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], T_vector, l, (q1 + q2))
                                                    elif properties2[0][1] == 'Q':
                                                        T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], SpinOperators.T[spin_2_index], l, (q1 + q2))

                                                    R_term = sop_R_term_delta_sigma_ROT(l, q1, q2, intr_name1, intr_name2, spin_1_name_i, spin_1_name_j, spin_2_name, T_left, T_right)
                                                    R_final += R_term

            # Double-spin double-spin mechanism pair
            elif properties1[0][0] == 'D' and properties2[0][0] == 'D':
                print('\nComputing R for double-spin double-spin mechanism pairs:')
                
                coupling_strengths_matrix1 = properties1[1]
                coupling_strengths_matrix2 = properties2[1]

                ranks1 = properties1[2]
                ranks2 = properties2[2]

                ls = list(set(ranks1) & set(ranks2))

                for (spin_1_index_i, spin_1_index_j), coupling_strength_1 in np.ndenumerate(coupling_strengths_matrix1):
                    if coupling_strength_1 != 0:

                        for (spin_2_index_i, spin_2_index_j), coupling_strength_2 in np.ndenumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index_i) + str(spin_1_index_j)
                                intr_name2 = mechanism2 + str(spin_2_index_i) + str(spin_2_index_j)
                                print(f'[{intr_name1} * {intr_name2}]')

                                # Handle chemically equivalent spins (homonuclear systems)
                                if spin_names is not None:
                                    for i, name in enumerate(spin_names):
                                        if i == spin_1_index_i:
                                            spin_1_name_i = name
                                        if i == spin_1_index_j:
                                            spin_1_name_j = name
                                        if i == spin_2_index_i:
                                            spin_2_name_i = name
                                        if i == spin_2_index_j:
                                            spin_2_name_j = name
                                else:
                                    spin_1_name_i = spin_1_index_i
                                    spin_1_name_j = spin_1_index_j
                                    spin_2_name_i = spin_2_index_i
                                    spin_2_name_j = spin_2_index_j

                                for l in ls:
                                    
                                    if settings.FRAME == 'lab':
                                        for q1 in range(-1, 2):
                                            for q2 in range(-1, 2):

                                                if np.abs(q1 + q2) <= l:

                                                    T_left_i = SpinOperators.T[spin_1_index_i]
                                                    T_left_j = SpinOperators.T[spin_1_index_j]
                                                    T_left = T_left_i[1, q1] @ T_left_j[1, q2]

                                                    T_right_i = SpinOperators.T[spin_2_index_i]
                                                    T_right_j = SpinOperators.T[spin_2_index_j]
                                                    T_right = T_right_i[1, q1] @ T_right_j[1, q2]

                                                    R_term = sop_R_term_delta_delta_LAB(l, q1, q2, intr_name1, intr_name2, spin_2_name_i, spin_2_name_j, T_left, T_right)
                                                    R_final += R_term

                                    elif settings.FRAME == 'rot':
                                        for q1_d1 in range(-1, 2):
                                            for q2_d1 in range(-1, 2):
                                                for q1_d2 in range(-1, 2):
                                                    for q2_d2 in range(-1, 2):

                                                        # Clebsch-Gordan restriction
                                                        if np.abs(q1_d1 + q2_d1) <= l and (q1_d1 + q2_d1) == (q1_d2 + q2_d2):

                                                            T_left_i = SpinOperators.T[spin_1_index_i]
                                                            T_left_j = SpinOperators.T[spin_1_index_j]
                                                            T_left = T_left_i[1, q1_d1] @ T_left_j[1, q2_d1]

                                                            T_right_i = SpinOperators.T[spin_2_index_i]
                                                            T_right_j = SpinOperators.T[spin_2_index_j]
                                                            T_right = T_right_i[1, q1_d2] @ T_right_j[1, q2_d2]

                                                            R_term = sop_R_term_delta_delta_ROT(l, q1_d1, q2_d1, q1_d2, q2_d2, intr_name1, intr_name2,
                                                                spin_1_name_i, spin_1_name_j, spin_2_name_i, spin_2_name_j, T_left, T_right)
                                                            R_final += R_term        
    print('\n All done!')                                 
    return R_final

####################################################################################################
# Relaxation superoperator class.
####################################################################################################
class RelaxationSuperoperator(Superoperator):
    """
    General class for the relaxation superoperator of a spin system.
    Inherits from Superoperator.

    See Superoperator (and Operator) class for more information.
    """
    def __init__(self, sop_R):
        super().__init__(sop_R)

    def to_isotropic_rotational_diffusion(self, fast_motion_limit=False, slow_motion_limit=False):
        """
        Set all J(w) functions in the relaxation superoperator to the isotropic rotational
        diffusion spectral density function J_iso_rot_diff.

        Input:
            - fast_motion_limit: Boolean to set the fast motion limit (default = False).
            - slow_motion_limit: Boolean to set the slow motion limit (default = False).
        """
        J_w_functions = [function for function in self.functions_in if 'J' in str(function)]
        subst_dict = {}
        for J_w in J_w_functions:
            # See docstring for extract_J_w_arguments and J_w_iso_rot_diff
            intrs, lq, arg = extract_J_w_symbols_and_args(J_w)
            subst_dict[J_w] = J_w_isotropic_rotational_diffusion(*intrs, lq[0], arg, tau_c,
                                                                 fast_motion_limit=fast_motion_limit, slow_motion_limit=slow_motion_limit)
        self.substitute(subst_dict)

    def neglect_cross_correlated_terms(self, mechanism1, mechanism2):
        """
        Neglect all cross-correlated terms between mechanism1 and mechanism2 in the relaxation superoperator.

        Input:
            - mechanism1: Name of the first mechanism.
            - mechanism2: Name of the second mechanism.
        """
        J_w_functions = [function for function in self.functions_in if 'J' in str(function)]
        for J_w in J_w_functions:
            # See docstring for extract_J_w_arguments
            intrs, _, _ = extract_J_w_symbols_and_args(J_w)
            if ',' in intrs[0]:
                if mechanism1 == mechanism2:
                    if str(J_w).count(mechanism1) > 1:
                        self.substitute({J_w: 0})
                else:
                    if mechanism1 in str(J_w) and mechanism2 in str(J_w):
                        self.substitute({J_w: 0})

    def neglect_ALL_cross_correlated_terms(self):
        """
        Neglect all cross-correlated terms in the relaxation superoperator.
        """
        J_w_functions = [function for function in self.functions_in if 'J' in str(function)]
        for J_w in J_w_functions:
            # See docstring for extract_J_w_arguments
            intrs, _, _ = extract_J_w_symbols_and_args(J_w)
            if ',' in intrs[0]:
                self.substitute({J_w: 0})

####################################################################################################
# Master equations.
####################################################################################################
def ime_equations_of_motion(R, basis_op_symbols, expectation_values=True, operator_indexes=None):
    """
    System of differential equations resulting from the inhomogeneous master equation.
    
    Input:
        - R: Relaxation superoperator matrix representation.
        - basis_op_symbols: List of basis operator symbols.
        - expectation_values: Boolean to display as expectation values (default = True).
        - operator_indexes: List of indexes to select a subset of basis operators (default = None).
    """
    if operator_indexes is not None:
        R = pick_from_matrix(R, operator_indexes)
        basis_op_symbols = pick_from_list(basis_op_symbols, operator_indexes)
    if expectation_values:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(f_expectation_value_t(x), t))
    else:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(x, t))
    rhs = smp.Matrix([smp.Symbol(f'\\Delta {symbol}'.replace('*', '')) for symbol in basis_op_symbols])
    if expectation_values:
        rhs = rhs.applyfunc(lambda x: f_expectation_value_t(x))
    rhs = R * rhs
    return smp.Eq(lhs, rhs, evaluate=False)

def lindblad_equations_of_motion(R, basis_op_symbols, expectation_values=True, operator_indexes=None):
    """
    System of differential equations resulting from the Lindblad master equation.
    
    Input:
        - R: Relaxation superoperator matrix representation.
        - basis_op_symbols: List of basis operator symbols.
        - expectation_values: Boolean to display as expectation values (default = True).
        - operator_indexes: List of indexes to select a subset of basis operators (default = None).
    """
    if operator_indexes is not None:
        R = pick_from_matrix(R, operator_indexes)
        basis_op_symbols = pick_from_list(basis_op_symbols, operator_indexes)
    if expectation_values:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(f_expectation_value_t(x), t))
    else:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(x, t))
    rhs = smp.Matrix([symbol for symbol in basis_op_symbols])
    if expectation_values:
        rhs = rhs.applyfunc(lambda x: f_expectation_value_t(x))
    rhs = R * rhs
    return smp.Eq(lhs, rhs, evaluate=False)

def equations_of_motion_to_latex(eqs, savename):
    """
    Convert a master equation to LaTeX.
    
    Input:
        - eqs: System of differential equations.
        - savename: Name to save the LaTeX file.
    """
    diff_eqs = ''
    diff_eqs += '\\begin{cases}\n'

    for lhs_i, rhs_i in zip(eqs.lhs, eqs.rhs):
        eq_latex = smp.latex(lhs_i) + '=' + smp.latex(rhs_i)
        eq_latex = eq_latex.replace('\\partial', 'd').replace('\\left|', '')\
                  .replace('\\right|', '').replace('*', '')
        diff_eqs += eq_latex + '\\\\\n'

    diff_eqs += '\\end{cases}'

    with open(f'EOMs_{savename}.txt', 'w') as file:
        file.write(diff_eqs)

####################################################################################################
# Convenience functions.
####################################################################################################
def sop_R_in_T_basis(S, INCOHERENT_INTERACTIONS, spin_names=None, sorting=None):
    """
    Compute the relaxation superoperator object in the direct product basis of spherical tensor operators.
    
    Input:
        - S: Spin system object.
        - INCOHERENT_INTERACTIONS: Dictionary of incoherent interactions.
        - spin_names: List of spin names to handle chemically equivalent spins (default = None).
        - sorting: Sorting of the basis operators (default = None).
    """
    # Compute the relaxation superoperator object
    Sops = SpinOperators(S)
    R = sop_R(Sops, INCOHERENT_INTERACTIONS, spin_names=spin_names)
    R = RelaxationSuperoperator(R)

    # Convert to spherical tensor basis
    T_basis, T_symbols = T_product_basis_and_symbols(Sops, sorting=sorting)
    R.to_basis(T_basis)

    return R, T_basis, T_symbols
