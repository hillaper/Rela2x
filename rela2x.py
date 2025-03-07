"""
The main module for the Rela²x package.

Author:
    Perttu Hilla, 2024-2025
    perttu.hilla@oulu.fi
    NMR Research Unit, University of Oulu.
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
import nmr_isotopes
from constants_and_variables import *

####################################################################################################
# Settings and modes of the program.
####################################################################################################
def set_relaxation_theory(theory):
    """
    Set the level of theory for the relaxation superoperator.
    
    Input:
        - theory: 'sc' for semiclassical, 'qm' for quantum mechanical. 
        NOTE: Default is semiclassical.
    """
    if theory not in ['sc', 'qm']:
        raise ValueError("Invalid relaxation theory. Choose 'sc' for semiclassical or 'qm' for quantum mechanical.")
    settings.RELAXATION_THEORY = theory

####################################################################################################
# Mathematical tools.
# NOTE: General tools defined here, more specific functionalities in classes defined later.
####################################################################################################
def KroneckerProduct(*m):
    """
    Symbolic Kronecker product of multiple matrices.

    Input:
        - m: Matrices to be Kronecker producted.

    Returns:
        - result: Kronecker product of the matrices.
    """
    result = m[0]
    for i in range(1, len(m)):
        result = smp.Matrix(result.shape[0]*m[i].shape[0], result.shape[1]*m[i].shape[1],\
                 lambda p, q: result[p//m[i].shape[0], q//m[i].shape[1]] * m[i][p%m[i].shape[0], q%m[i].shape[1]])
    return result

# NOTE: op in the following functions refers to SymPy matrices.
def commutator(op1, op2):
    """Symbolic commutator of two operators."""
    return op1 * op2 - op2 * op1

# Liouville bracket and norm
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
        - op: Operator to be changed (matrix representation)
        - basis: Basis set (list of matrix representations of the basis states/operators).

    Returns:
        - op_new: Operator in the new basis.
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
# Information extraction from input of NMR isotopes
def spin_quantum_numbers(isotopes):
    """
    Spin quantum numbers of nuclear isotopes.
    
    Input:
        - isotopes: List of nuclear isotopes.

    Returns:
        - S: List of spin quantum numbers in the same order as the input isotopes.
    """
    try:
        return [nmr_isotopes.ISOTOPES[isotope][0] for isotope in isotopes]
    except KeyError:
        raise ValueError("Given NMR isotope not found in the nmr_isotopes.py file.")

# Information extraction from spherical tensor operator symbols 
# NOTE: This is the basis that Rela2x uses as a default.
# TODO: Smarter indexing system.
def T_symbol_spin_order(T_symbol):
    """
    Spin order of a spherical tensor operator symbol, i.e. the number of operators in the product.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).

    Returns:
        - Spin order of the symbol.
    """
    return str(T_symbol).count('T')

def T_symbol_coherence_order(T_symbol):
    """
    Coherence order of a spherical tensor operator symbol.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).

    Returns:
        - Coherence order of the symbol.
    """
    s = str(T_symbol)
    numbers = re.findall(r'(\-?\d)}\^\{\(\d+\)\}', s)
    return sum([int(num) for num in numbers])

def T_symbol_type(T_symbol):
    """
    Type (population or coherence) of a spherical tensor operator symbol.
    NOTE: Kind of hard-coded, but works for the purposes of Rela2x.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).

    Returns:
        - Type of the symbol, 0 for population, 1 for coherence.
    """
    s = str(T_symbol)
    # Up to rank 6 should be enough for the purposes of Rela2x
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
    Nth spin projection of a spherical tensor operator symbol (value of the component q).
    NOTE: N starts from 1 as opposed to 0 in Python.
    
    Input:
        - T_symbol: Spherical tensor operator symbol (SymPy symbol).

    Returns:
        - Nth spin projection of the symbol if found, 0 otherwise.
    """
    s = str(T_symbol)
    pattern = r'\\hat\{T\}\_{([^}]*)}\^\{\('+str(N)+'\)\}'
    match = re.search(pattern, s)
    if match:
        return match.group(1)[1:]
    else:
        return 0

def T_symbol_list_index(T_symbols, spin_index_lqs):
    """
    Find the list index of a symbol with given spin indexes, ls and qs from a list of spherical tensor operator symbols.
    
    Input:
        - T_symbols: List of spherical tensor operator symbols
        - spin_index_lqs: String of the form '210' for spin 2, l = 1, q = 0, or '110*210' for a product of two operators, etc.
        NOTE: The order of the spin indexes and lqs is important for the function to work.

    Returns:
        - Index of the symbol that matches with the given spin indexes and lqs. None if no match is found.
    """
    spin_index_lqs = spin_index_lqs.split('*')
    spin_index_lqs = [(int(spin_index_lq[0]), int(spin_index_lq[1]), int(spin_index_lq[2:])) for spin_index_lq in spin_index_lqs]

    for i, symbol in enumerate(T_symbols):
        pattern = ''
        for spin_index, l, q in spin_index_lqs:
            pattern += f'\\hat{{T}}_{{{l}{q}}}^{{({spin_index})}}'
            pattern += '*'
        pattern = pattern[:-1]

        if pattern == str(symbol):
            return i
        
    print(f'No match found from the basis symbols for {spin_index_lqs}.')
    return None

# String hashing (for sorting purposes)
def string_to_number(string):
    return int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16)

# Convenience function
def sort_interactions(intr1, intr2):
    """Sort an interaction pair. Used for cosmetic purposes."""
    if str(intr1) == str(intr2):
        return str(intr1)
    else:
        # Hash the strings and return the sorted pair
        return sorted([str(intr1), str(intr2)], key=string_to_number)

# List and matrix operations
def pick_from_list(lst, kept_indices):
    """Select elements from a list."""
    return [lst[i] for i in kept_indices]

def pick_from_matrix(matrix, kept_indices):
    """Select rows and columns from a matrix."""
    return matrix[kept_indices, :][:, kept_indices]

def cut_list(lst, removed_indices):
    """Cut a list to a smaller size."""
    return [item for i, item in enumerate(lst) if i not in removed_indices]

def cut_matrix(matrix, removed_indices):
    """Cut a matrix to a smaller size."""
    # Remove rows
    rows_to_keep = [i for i in range(matrix.rows) if i not in removed_indices]
    matrix_loc = matrix[rows_to_keep, :]
    
    # Remove columns
    cols_to_keep = [i for i in range(matrix.cols) if i not in removed_indices]
    matrix_loc = matrix_loc[:, cols_to_keep]
    
    return matrix_loc

# Filter functions based on allowed coherences, spin orders and types
# NOTE: General input and return structure defined in coherence_order_filter.
def coherence_order_filter(operator, basis_state_symbols, allowed_coherences):
    """
    Filter an operator and basis state symbols based on allowed coherences.

    Input:
        - operator: Operator (matrix representation) to be filtered.
        - basis_state_symbols: List of basis state symbols.
        - allowed_coherences: List of allowed coherences.

    Returns:
        - operator: Filtered operator.
        - basis_state_symbols: Filtered basis state symbols.
    """
    basis_coherences = [T_symbol_coherence_order(T_symbol) for T_symbol in basis_state_symbols]
    indexes_to_delete = [i for i, coherence in enumerate(basis_coherences) if coherence not in allowed_coherences]
    unique_indexes_to_delete = list(set(indexes_to_delete))
    return cut_matrix(operator, unique_indexes_to_delete), cut_list(basis_state_symbols, unique_indexes_to_delete)

def spin_order_filter(operator, basis_state_symbols, allowed_spin_orders):
    """
    Filter an operator and basis state symbols based on allowed spin orders.

    Input:
        - allowed_spin_orders: List of allowed spin orders.
    """
    basis_spin_orders = [T_symbol_spin_order(T_symbol) for T_symbol in basis_state_symbols]
    indexes_to_delete = [i for i, spin_order in enumerate(basis_spin_orders) if spin_order not in allowed_spin_orders]
    unique_indexes_to_delete = list(set(indexes_to_delete))
    return cut_matrix(operator, unique_indexes_to_delete), cut_list(basis_state_symbols, unique_indexes_to_delete)

def type_filter(operator, basis_state_symbols, allowed_type):
    """
    Filter an operator and basis state symbols based on allowed type.

    Input:
        - allowed_type: Allowed type, 0 for population, 1 for coherence.
        NOTE: This is not a list, but a single integer.
    """
    basis_types = [T_symbol_type(T_symbol) for T_symbol in basis_state_symbols]
    indexes_to_delete = [i for i, type in enumerate(basis_types) if type != allowed_type]
    unique_indexes_to_delete = list(set(indexes_to_delete))
    return cut_matrix(operator, unique_indexes_to_delete), cut_list(basis_state_symbols, unique_indexes_to_delete)

# List operations
def list_indexes(lst):
    """Return the indexes of a list."""
    return list(range(len(lst)))

# Combinatorics
def all_combinations(N, *args, reverse=False):
    """Generate all combinations of N lists. Used for spherical tensor operator product basis generation."""
    list_combinations = list(itertools.combinations(args, N))

    # For each combination of lists, generate all combinations with one element from each list
    all_combinations = []
    for lists in list_combinations:
        all_combinations.extend(itertools.product(*lists))

    # Reverse the order of the combinations (convenient for basis sorting)
    if reverse:
        all_combinations = list(reversed(all_combinations))

    return all_combinations

####################################################################################################
# Visualization tools.
# TODO: Search for cases where the functions do not work as intended.
####################################################################################################
def matrix_nonzeros(matrix):
    """Nonzero elements of a matrix."""
    return matrix.applyfunc(lambda x: 1 if x != 0 else 0)

def visualize_operator(operator, rows_start=0, rows_end=None, basis_symbols=None, fontsize=8):
    """
    Visualize a given operator (its matrix representation).
    Plot is shown automatically.

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

    # Increase font size for small matrices
    if operator_nonzeros.shape[0] < 16:
        fontsize += 2

    # Create the plot with a suitable size
    if operator_nonzeros.shape[0] <= 16:
        _, ax = plt.subplots(figsize=(4, 4), dpi=150)
    else:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Create color map for the nonzeros
    norm = plt.Normalize(0, np.amax(operator_nonzeros))
    cmap = plt.cm.Blues

    # Plot the nonzeros
    ax.imshow(operator_nonzeros, cmap=cmap, alpha=0.9, norm=norm)

    # Shift the grid
    ax.set_xticks(np.arange(-.5, operator_nonzeros.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, operator_nonzeros.shape[0], 1), minor=True)
    if operator_nonzeros.shape[0] <= 64:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    else:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Move x-axis ticks to the top
    ax.xaxis.tick_top()

    # Add tick labels if basis symbols are not given
    if basis_symbols is None:
        # Set major ticks to start from 1
        # Include only every second or fourth tick if the matrix is large
        if operator_nonzeros.shape[0] <= 16:
            ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
            ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1))
        elif operator_nonzeros.shape[0] <= 64:
            ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 2))
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 2))
            ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1, 2))
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 2))
        else:
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 4))
            ax.set_xticklabels([])
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 4))

        # Apply font size to ticks
        if operator_nonzeros.shape[0] <= 16:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
        elif operator_nonzeros.shape[0] <= 64:
            ax.tick_params(axis='both', which='major', labelsize=fontsize + 1)
        else:
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    # Add basis symbols next to y tick labels if given
    elif basis_symbols is not None:
        basis_symbols = basis_symbols[rows_start:rows_end]
        basis_symbols = [f'${symbol}$'.replace('*', '').replace(' ', '') for symbol in basis_symbols]
        basis_symbols = [f'{label}{" " * (5 - len(str(i + 1)))}{i + 1}' for i, label in enumerate(basis_symbols)]

        ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
        ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
        ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
        ax.set_yticklabels(basis_symbols)

        # Apply font size to ticks
        if operator_nonzeros.shape[0] <= 16:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
        else:
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    plt.tight_layout()
    plt.show()

def visualize_many_operators(operators, rows_start=0, rows_end=None, basis_symbols=None, fontsize=8):
    """
    Visualize a list of operators (their matrix representations).
    Plot is shown automatically.

    NOTE: This is useful for, e.g., visualizing differences between different relaxation superoperators.

    Input:
        - operators: List of operators to be visualized.
        - rows_start: Starting row index for the visualization.
        - rows_end: Ending row index for the visualization.
        - basis_symbols: Basis symbols for the visualization (effectively a legend for the basis states)
        - fontsize: Font size for the basis symbols.

    NOTE: Apart from basic plotting, mostly just for pretty visualization purposes.
    """
    operators = [operator[rows_start:rows_end, rows_start:rows_end] for operator in operators]
    operators_nonzeros = [np.array(matrix_nonzeros(operator), dtype=np.float32) for operator in operators]
    operator_nonzeros = np.sum(operators_nonzeros, axis=0)

    # Increase font size for small matrices
    if operator_nonzeros.shape[0] < 16:
        fontsize += 2

    # Create the plot with a suitable size
    if operator_nonzeros.shape[0] <= 16:
        _, ax = plt.subplots(figsize=(4, 4), dpi=150)
    else:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Create color map for the nonzeros
    norm = plt.Normalize(0, np.amax(operator_nonzeros))
    cmap = plt.cm.Blues

    # Plot the nonzeros
    ax.imshow(operator_nonzeros, cmap=cmap, alpha=0.9, norm=norm)

    # Shift the grid
    ax.set_xticks(np.arange(-.5, operator_nonzeros.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, operator_nonzeros.shape[0], 1), minor=True)
    if operator_nonzeros.shape[0] <= 64:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
    else:
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Move x-axis ticks to the top
    ax.xaxis.tick_top()

    # Add tick labels if basis symbols are not given
    if basis_symbols is None:
        # Set major ticks to start from 1
        # Include only every second or fourth tick if the matrix is large
        if operator_nonzeros.shape[0] <= 16:
            ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
            ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1))
        elif operator_nonzeros.shape[0] <= 64:
            ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 2))
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 2))
            ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1, 2))
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 2))
        else:
            ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 4))
            ax.set_xticklabels([])
            ax.set_yticklabels(np.arange(1, operator_nonzeros.shape[0] + 1, 4))

        # Apply font size to ticks
        if operator_nonzeros.shape[0] <= 16:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
        elif operator_nonzeros.shape[0] <= 64:
            ax.tick_params(axis='both', which='major', labelsize=fontsize + 1)
        else:
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    # Add basis symbols next to y tick labels if given
    elif basis_symbols is not None:
        basis_symbols = basis_symbols[rows_start:rows_end]
        basis_symbols = [f'${symbol}$'.replace('*', '').replace(' ', '') for symbol in basis_symbols]
        basis_symbols = [f'{label}{" " * (5 - len(str(i + 1)))}{i + 1}' for i, label in enumerate(basis_symbols)]

        ax.set_xticks(np.arange(0, operator_nonzeros.shape[1], 1))
        ax.set_yticks(np.arange(0, operator_nonzeros.shape[0], 1))
        ax.set_xticklabels(np.arange(1, operator_nonzeros.shape[1] + 1))
        ax.set_yticklabels(basis_symbols)

        # Apply font size to ticks
        if operator_nonzeros.shape[0] <= 16:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
        else:
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 1)

    plt.tight_layout()
    plt.show()
    
####################################################################################################
# Symbolic operators and expectation values.
####################################################################################################
# Spherical tensor operators
def op_T_symbol(l, q, index):
    """
    Symbolic spherical tensor operator
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
        - index: Spin index.

    Returns:
        - Spherical tensor operator symbol.
    """
    return smpq.Operator(f'\\hat{{T}}_{{{l}{q}}}^{{({index})}}')

def product_op_T_symbol(ls, qs, indices):
    """
    Symbolic product of spherical tensor operators
    
    Input:
        - ls: List of ranks of the spherical tensor operators.
        - qs: List of projections of the spherical tensor operators.
        - indices: List of spin indices.

    Returns:
        - Product-operator symbol.
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
    """Spin angular momentum operator for quantum number S in the x-direction."""
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
    """Spin angular momentum operator for quantum number S in the y-direction."""
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
    """Spin angular momentum operator for quantum number S in the z-direction."""
    m = np.arange(-S, S+1)
    m = np.flip(m)

    Sz = smp.zeros(len(m), len(m), complex=True)
    for i in range(len(m)):
        Sz[i, i] = m[i]

    return smp.Matrix(Sz.T, complex=True).applyfunc(smp.nsimplify)

def op_Sp(S):
    """Spin angular momentum raising operator for quantum number S."""
    return op_Sx(S) + smp.I * op_Sy(S)

def op_Sm(S):
    """Spin angular momentum lowering operator for quantum number S."""
    return op_Sx(S) - smp.I * op_Sy(S)

def op_Svec(S):
    """Cartesian spin angular momentum vector operator for quantum number S."""
    return [op_Sx(S), op_Sy(S), op_Sz(S)]

####################################################################################################
# Spherical tensors and spherical tensor operators.
# NOTE: These functions use dictionaries of the form {(l, q): T_lq} for spherical tensors.
####################################################################################################
# Classical spherical tensors:
def vector_to_spherical_tensor(vector):
    """
    Convert a vector to a spherical tensor of rank 1.

    Input:
        - vector: Vector in the form [x, y, z].

    Returns:
        - dictionary of the form {(l, q): T_lq}.
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

# Coupling of spherical tensor operators.
# NOTE: Used for bilinear contributions in the relaxation superoperator.
def op_T_coupled_lq(T1_dict, T2_dict, l, q):
    """
    Coupled spherical tensor operator of rank l and projection q from two spherical tensors of rank 1.

    Input:
        - T1_dict: First dictionary of spherical tensor components.
        - T2_dict: Second dictionary of spherical tensor components.
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
        - spin_index: Index of the desired spin.

    Returns:
        - The many-spin system version of the single-spin operator.
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
    General class for the spin operators (spherical tensor operators) for a spin system.
    NOTE: One of the main classes of Rela2x.
    
    Input:
        - spinsystem: List of nuclear isotopes (strings) that define the spin system.
        
    Attributes:
        - S: Spin quantum numbers of the spins.
        - N_spins: Number of spins in the system.
        - N_states: Number of states in the system.
        - T: Spherical tensor operators for each spin.
        - T_symbol: Spherical tensor operator symbols for each spin.
    """
    def __init__(self, spinsystem):
        # Check that the input is a list of strings
        if not all(isinstance(isotope, str) for isotope in spinsystem):
            raise ValueError("The spinsystem input has to be a list of strings corresponding to NMR isotopes (e.g. ['1H', '13C']).")
                
        self.spinsystem = spinsystem
        self.S = spin_quantum_numbers(spinsystem)

        self.N_spins = len(self.S)
        self.gen_N_states()
        
        self.gen_many_spin_T_operators()
        self.gen_T_operator_symbols()

    def gen_N_states(self):
        """Generate the number of states in the spin system."""
        self.N_states = 1
        for i in range(self.N_spins):
            self.N_states *= int(2*self.S[i] + 1)

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
# TODO: Generation of the product basis operators should be simplified.
# TODO: Generalize the basis operator sorting.
####################################################################################################
# Product basis of spherical tensor operators
def T_product_basis(SpinOperators):
    """
    Generate the direct product basis of spherical tensor operators.
    Each T_lq operator of each spin is multiplied with the T_lq operators of the other spins.

    Input:
        - SpinOperators: SpinOperators object.

    Returns:
        - T_product_basis: list of product basis operators.
    """
    S = SpinOperators.S
    N_spins = SpinOperators.N_spins
    T = SpinOperators.T

    # Combinatorics
    ops = [[T[i][(l, q)] for l in range(int(2*S[i])+1) for q in range(-l, l+1)] for i in range(N_spins)]
    op_indexes = [list_indexes(ops[i]) for i in range(N_spins)]
    op_indexes = all_combinations(N_spins, *op_indexes, reverse=True)
    spin_indexes = [tuple(range(N_spins)) for _ in range(len(op_indexes))]

    # Generate the product basis
    T_product_basis = []
    for spin_index_tuple, op_index_tuple in zip(spin_indexes, op_indexes):
        T_product = 1
        for i, j in zip(spin_index_tuple, op_index_tuple):
            if isinstance(T_product, int):
                T_product = ops[i][j]
            else:
                T_product = T_product @ ops[i][j]
        T_product_basis.append(T_product)

    # Norms of the basis operators. Used for normalization and later for basis to observables conversion.
    norms = [Lv_norm(T_product_basis[i]) for i in range(len(T_product_basis))]

    # Normalize the basis
    T_product_basis = [T_product_basis[i] / norms[i] for i in range(len(T_product_basis))]
    return T_product_basis, norms

def T_product_basis_symbols(SpinOperators):
    """
    Generate the direct product basis symbols of spherical tensor operators.
    Each T_lq symbol of each spin is multiplied with the T_lq symbols of the other spins.

    Input:
        - SpinOperators: SpinOperators object.

    Returns:
        - T_product_basis_symbols: list of product basis symbols.
    """
    S = SpinOperators.S
    N_spins = SpinOperators.N_spins
    T_symbol = SpinOperators.T_symbol

    # Combinatorics
    symbols = [[T_symbol[i][(l, q)] for l in range(int(2*S[i])+1) for q in range(-l, l+1)] for i in range(N_spins)]
    symbol_indexes = [list_indexes(symbols[i]) for i in range(N_spins)]
    symbol_indexes = all_combinations(N_spins, *symbol_indexes, reverse=True)
    spin_indexes = [tuple(range(N_spins)) for i in range(len(symbol_indexes))]

    # Generate the product basis symbols
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
# NOTE: There are multiple ways to do the sorting, and it is fairly arbitrary. These are just two possibilities.
def T_basis_split_to_coherence_orders(T_product_basis, T_product_basis_symbols, T_product_basis_norms):
    """
    Split a basis of spherical tensor operators to groups of same coherence order.
    
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
        - T_product_basis_norms: Norms of the basis operators

    Returns:
        - T_basis_groups: Dictionary of groups of same coherence order (key = coherence order, value = list of matrices)
        - T_symbol_groups: Dictionary of groups of same coherence order (key = coherence order, value = list of symbols)
        - T_norm_groups: Dictionary of groups of same coherence order (key = coherence order, value = list of norms)
    """
    coherence_orders = [T_symbol_coherence_order(T_symbol) for T_symbol in T_product_basis_symbols]
    T_basis_groups = {}
    T_symbol_groups = {}
    T_norm_groups = {}

    for i, coherence_order in enumerate(coherence_orders):
        if coherence_order in T_basis_groups:
            T_basis_groups[coherence_order].append(T_product_basis[i])
            T_symbol_groups[coherence_order].append(T_product_basis_symbols[i])
            T_norm_groups[coherence_order].append(T_product_basis_norms[i])
        else:
            T_basis_groups[coherence_order] = [T_product_basis[i]]
            T_symbol_groups[coherence_order] = [T_product_basis_symbols[i]]
            T_norm_groups[coherence_order] = [T_product_basis_norms[i]]

    return T_basis_groups, T_symbol_groups, T_norm_groups

# NOTE: Functions below have the same input and return structure as spin_order_sort_T_product_basis.
def spin_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms):
    """
    Sort the product basis of spherical tensor operators by spin order.
    
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
        - T_product_basis_norms: Norms of the basis operators

    Returns:
        - tuple of the form (sorted_T_product_basis, sorted_T_product_basis_symbols, sorted_T_product_basis_norms)
    """
    spin_orders = [T_symbol_spin_order(op) for op in T_product_basis_symbols]
    sorting = np.argsort(spin_orders)
    return [T_product_basis[i] for i in sorting], [T_product_basis_symbols[i] for i in sorting], [T_product_basis_norms[i] for i in sorting]

def coherence_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms):
    """
    Sort the product basis of spherical tensor operators by coherence order.
    """
    coherence_orders = [T_symbol_coherence_order(op) for op in T_product_basis_symbols]
    sorting = np.lexsort((coherence_orders, np.abs(coherence_orders)))
    return [T_product_basis[i] for i in sorting], [T_product_basis_symbols[i] for i in sorting], [T_product_basis_norms[i] for i in sorting]

def type_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms):
    """
    Sort the product basis of spherical tensor operators by type.
    """
    types = [T_symbol_type(op) for op in T_product_basis_symbols]
    sorting = np.argsort(types)
    return [T_product_basis[i] for i in sorting], [T_product_basis_symbols[i] for i in sorting], [T_product_basis_norms[i] for i in sorting]

def projection_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms, spin_index):
    """
    Sort the product basis of spherical tensor operators by projection (q) of the operator
    acting on the specified spin.
        
    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
        - T_product_basis_norms: Norms of the basis operators
        - spin_index: Spin index for the projection sorting.

    Returns:
        - tuple of the form (sorted_T_product_basis, sorted_T_product_basis_symbols, sorted_T_product_basis_norms)
    """
    T_product_basis_NEW = []
    T_product_basis_symbols_NEW = []
    T_product_basis_norms_NEW = []

    T_product_basis_coherence_orders, T_product_basis_coherence_order_symbols, T_product_basis_coherence_order_norms\
        = T_basis_split_to_coherence_orders(T_product_basis, T_product_basis_symbols, T_product_basis_norms)

    for coherence_order, T_symbols in T_product_basis_coherence_order_symbols.items():
        projections = [T_symbol_Nth_spin_projection(T_symbol, spin_index) for T_symbol in T_symbols]
        sorting = np.argsort(projections)
        T_product_basis_NEW += [T_product_basis_coherence_orders[coherence_order][i] for i in sorting]
        T_product_basis_symbols_NEW += [T_product_basis_coherence_order_symbols[coherence_order][i] for i in sorting]
        T_product_basis_norms_NEW += [T_product_basis_coherence_order_norms[coherence_order][i] for i in sorting]

    return T_product_basis_NEW, T_product_basis_symbols_NEW, T_product_basis_norms_NEW

def identity_first_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms):
    """
    Move the identity operator to the first position in the product basis.
    """
    identity_index = [i for i, T_symbol in enumerate(T_product_basis_symbols) if '{E}' in str(T_symbol)][0]
    T_product_basis_NEW = [T_product_basis[identity_index]] + T_product_basis[:identity_index] + T_product_basis[identity_index+1:]
    T_product_basis_symbols_NEW = [T_product_basis_symbols[identity_index]] + T_product_basis_symbols[:identity_index] + T_product_basis_symbols[identity_index+1:]
    T_product_basis_norms_NEW = [T_product_basis_norms[identity_index]] + T_product_basis_norms[:identity_index] + T_product_basis_norms[identity_index+1:]
    return T_product_basis_NEW, T_product_basis_symbols_NEW, T_product_basis_norms_NEW

# Quick sorting that combines the functions above.
def full_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms, sorting='v1'):
    """
    Fully sort the product basis of spherical tensor operators.

    Version 1 sorts by type --> spin order --> coherence order.
    Version 2 sorts by projection of each spin --> coherence order --> identity operator first.

    Input:
        - T_product_basis: Basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Basis of spherical tensor operator symbols (list of SymPy symbols)
        - T_product_basis_norms: Norms of the basis operators
        - sorting: Sorting version ('v1' or 'v2').

    Returns:
        - tuple of the form (sorted_T_product_basis, sorted_T_product_basis_symbols, sorted_T_product_basis_norms)
    """
    if sorting == 'v1':
        T_product_basis, T_product_basis_symbols, T_product_basis_norms = type_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms)
        T_product_basis, T_product_basis_symbols, T_product_basis_norms = spin_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms)
        T_product_basis, T_product_basis_symbols, T_product_basis_norms = coherence_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms)

    elif sorting == 'v2':
        # Loop over all spins and sort by projection
        for i in range(1, 10): # NOTE: This will always suffice for up to 10 spins
            T_product_basis, T_product_basis_symbols, T_product_basis_norms = projection_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms, i)
        T_product_basis, T_product_basis_symbols, T_product_basis_norms = coherence_order_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms)
        T_product_basis, T_product_basis_symbols, T_product_basis_norms = identity_first_sort_T_product_basis(T_product_basis, T_product_basis_symbols, T_product_basis_norms)

    return T_product_basis, T_product_basis_symbols, T_product_basis_norms

def T_product_basis_and_symbols(SpinOperators, sorting='v1'):
    """
    Generate and sort the product basis of spherical tensor operators.
    
    Input:
        - SpinOperators: SpinOperators object.
        - sorting: Sorting version (None, 'v1' or 'v2').

    Returns:
        - T_product_basis: Sorted basis of spherical tensor operators (list of matrices)
        - T_product_basis_symbols: Sorted basis of spherical tensor operator symbols (list of SymPy symbols)
    """
    basis, norms = T_product_basis(SpinOperators)
    symbols = T_product_basis_symbols(SpinOperators)
    if sorting == None:
        return basis, symbols, norms
    else:
        return full_sort_T_product_basis(basis, symbols, norms, sorting=sorting)
    
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
        # Check that the input is a SymPy matrix
        if not isinstance(op, smp.Matrix):
            raise ValueError("The operator input for the Operator class has to be a SymPy matrix.")
        self.op = op
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

    # Matrix algebra
    def to_basis(self, basis):
        """
        Converts self.op to a different basis.
        
        Input:
            - basis: List of new basis operators.
        """
        self.op = op_change_of_basis(self.op, basis)

    # Symbols, functions and substitutions
    def get_symbols(self):
        """
        Get the symbols in self.op.
        
        Returns:
            - List of symbols in the operator.
        """
        return sorted(list(self.op.free_symbols), key=lambda x: str(x))
    
    def get_functions(self):
        """
        Get the functions in self.op.
        
        Returns:
            - List of functions in the operator.
        """
        return sorted(list(self.op.atoms(smp.Function)), key=lambda x: str(x))
    
    def substitute(self, substitutions_dict):
        """
        Substitute symbols and functions in self.op with numerical values.
        
        Input:
            - substitutions_dict: Dictionary of substitutions of the form {symbol: value}.
        
        NOTE: This can be useful for SymPy --> NumPy conversion.
        """
        self.op = self.op.subs(substitutions_dict)

        # Update symbols_in and functions_in
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()
    
    # Visualization
    def visualize(self, rows_start=0, rows_end=None, basis_symbols=None, fontsize=8):
        """
        Visualize self.op.
        See visualize_operator function for more information.
        
        Input:
            - rows_start: Starting row index for the visualization.
            - rows_end: Ending row index for the visualization.
            - basis_symbols: Basis symbols for the visualization (effectively a legend for the basis states)
            - fontsize: Font size for the basis symbols.
        """
        visualize_operator(self.op, rows_start=rows_start, rows_end=rows_end, basis_symbols=basis_symbols, fontsize=fontsize)

class Superoperator(Operator):
    """
    General class for superoperators. Inherits from Operator.

    See Operator class for more information.
    """
    def __init__(self, sop):
        super().__init__(sop)

    # Change of basis for superoperators
    def to_basis(self, basis):
        """
        Convert self.op to a different basis.
        
        Input:
            - basis: List of new basis operators.
        """
        basis_vectorized = vectorize_all(basis)
        print('\nChanging basis...')
        self.op = op_change_of_basis(self.op, basis_vectorized)
        print('Basis changed.')

####################################################################################################
# Spectral density functions and relaxation constants.
####################################################################################################
def Lorentzian(w, tau_c, fast_motion_limit=False, slow_motion_limit=False):
    """
    Lorentzian function (normalized to tau_c at w = 0). Used for spectral density functions.
    
    Input:
        - w: Frequency.
        - tau_c: Correlation time.
        - fast_motion_limit: Whether to use the fast motion limit where (w * tau_c) << 1. Default is False.
        - slow_motion_limit: Whether to use the slow motion limit where (w * tau_c) >> 1. Default is False.

    Returns:
        - J(w) = tau_c / (1 + (w * tau_c)^2).
    """
    # This is to handle division by zero
    if w == 0:
        return tau_c
    else:
        if fast_motion_limit:
            return tau_c
        elif slow_motion_limit:
            return 1 / (w**2 * tau_c)
        else:
            return tau_c / (1 + (w * tau_c)**2)

def Schofield_theta(w):
    """
    Schofield thermal correction exp(-1/2 * beta * w) in the quantum mechanical spectral density function.
    NOTE: beta = hbar / (k_B * T), defined in constants_and_variables.py.
    """
    return smp.exp(-beta * w / 2)
        
def J_w(intr1, intr2, l, argument):
    """
    Spectral density function J(w).
    
    Input:
        - intr1: String for the first interaction.
        - intr2: String for the second interaction.
        - l: Rank of the spherical tensor operator (q = 0 because of Hubbard's result)
        - argument: Argument of the spectral density function (combination of angular frequencies).

    Returns:
        - J(w) or J(w)*e^(-1/2 * beta * w), depending on the relaxation theory.
    """
    intr_sorted = sort_interactions(intr1, intr2)

    # NOTE: If same interaction twice, use it only once in the superscript
    if isinstance(intr_sorted, str):
        expr = smp.Function(f'J^{{{intr_sorted}}}_{{{l, 0}}}')(smp.Abs(argument))
    else:
        expr = smp.Function(f'J^{{{intr_sorted[0]}, {intr_sorted[1]}}}_{{{l, 0}}}')(smp.Abs(argument))

    if settings.RELAXATION_THEORY == 'sc':
        return expr
    elif settings.RELAXATION_THEORY == 'qm':
        return expr * Schofield_theta(argument)

def J_w_isotropic_rotational_diffusion(intr1, intr2, l, argument,
                                       fast_motion_limit=False, slow_motion_limit=False):
    """
    Isotropic rotational diffusion spectral density function, which is a Lorentzian function.
    
    Input:
        - intr1: String for the first interaction.
        - intr2: String for the second interaction.
        - l: Rank of the spherical tensor operator (NOTE: applies only for l > 0).
        - argument: Argument of the spectral density function (combination of angular frequencies).
        - fast_motion_limit: Whether to use the fast motion limit where (w * tau_c) << 1. Default is False.
        - slow_motion_limit: Whether to use the slow motion limit where (w * tau_c) >> 1. Default is False.

    Returns:
        - J(w) in the isotropic rotational diffusion model.
    """
    intr_sorted = sort_interactions(intr1, intr2)
    tau_c_l = 6*tau_c / (l*(l+1))

    if isinstance(intr_sorted, str):
        G = smp.Function(f'G^{{{intr_sorted}}}_{{{l, 0}}}')(0)
    else:
        G = smp.Function(f'G^{{{intr_sorted[0]}, {intr_sorted[1]}}}_{{{l, 0}}}')(0)

    J_w = 2*G * Lorentzian(argument, tau_c_l, fast_motion_limit=fast_motion_limit, slow_motion_limit=slow_motion_limit)
    return J_w
    
# Helper functions for RelaxationSuperoperator object
def extract_J_w_symbols_and_args(J):
    """
    Extract the symbols and arguments in the spectral density function J(w)
    (or G(0) in the case of isotropic rotational diffusion).
    Used for substitution of symbols and functions in the relaxation superoperator.

    Input:
        - J: Spectral density function symbol.

    Returns:
        - intrs: Interaction names.
        - lq: Rank and projection of the spherical tensor operator.
        - arg: Argument of the spectral density function.
    """
    J_str = str(J.func)

    intrs = re.findall(r'\^{(.*?)}', J_str)[0]
    lq = re.findall(r'\_\{\((.*?)\)\}', J_str)[0]

    intrs = tuple(intrs.split(', ')) if isinstance(intrs, tuple) else tuple([intrs, intrs])
    lq = tuple(map(int, lq.split(', ')))
    arg = J.args[0]

    return intrs, lq, arg
    
####################################################################################################
# Relaxation superoperators.
# NOTE: alpha is single-spin interaction and beta is two-spin interaction.
# See https://doi.org/10.1016/j.jmr.2024.107828)
# TODO: Allow non-secular terms in the relaxation superoperator.
####################################################################################################
def sop_R_term(op_T_left, J_w, op_T_right):
    """
    Terms appearing in the sum for the relaxation superoperator: T * J(w) * T^dagger.
    NOTE: The left and right "order" of the operators is just for bookkeeping purposes.

    Input:
        - op_T_left: Left spherical tensor operator.
        - J_w: Spectral density function.
        - op_T_right: Right spherical tensor operator.

    Returns:
        - Term in the relaxation superoperator.
    """
    if settings.RELAXATION_THEORY == 'sc':
        return smp.Rational(1, 2) * J_w * sop_double_commutator(op_T_left.H, op_T_right)
    elif settings.RELAXATION_THEORY == 'qm':
        return -J_w * sop_D(op_T_left.H, op_T_right)
    
# NOTE: The functions below have the same in input and return structure as sop_R_term_alpha_alpha.
def sop_R_term_alpha_alpha(l, q, alpha1, alpha2, alpha1_spin_name, alpha2_spin_name,
                           op_T_left, op_T_right,
                           keep_non_secular=False):
    """
    Term in the relaxation superoperator between two single-spin interactions.
    
    Input:
        - l: Rank of the spherical tensor operator.
        - q: Projection of the spherical tensor operator.
        - alpha1: Name of the first single-spin interaction.
        - alpha2: Name of the second single-spin interaction.
        - alpha1_spin_name: Name of the first spin associated with the first single-spin interaction.
        - alpha2_spin_name: Name of the second spin associated with the second single-spin interaction.
        - op_T_left: Left spherical tensor operator.
        - op_T_right: Right spherical tensor operator.

        - keep_non_secular: Whether to keep non-secular terms in the relaxation superoperator.
            NOTE: Only applicable for the semiclassical relaxation theory.

    Returns:
        - Term in the relaxation superoperator.
    """
    w_s1 = smp.Symbol(f'\\omega_{{{alpha1_spin_name}}}', real=True)
    w_s2 = smp.Symbol(f'\\omega_{{{alpha2_spin_name}}}', real=True)

    # Dirac delta function argument for the secular approximation
    delta_sec = q*(w_s1 - w_s2)

    # Check secular approximation
    if delta_sec != 0:

        # Keep non-secular terms if specified (and semiclassical relaxation theory is used)
        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

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

def sop_R_term_alpha_beta(l, q1, q2, alpha, beta, alpha_spin_name, beta_spin_name1, beta_spin_name2,
                          op_T_left, op_T_right,
                          keep_non_secular=False):
    """
    Term in the relaxation superoperator between a single-spin interaction and a two-spin interaction.
    """
    w_s = smp.Symbol(f'\\omega_{{{alpha_spin_name}}}', real=True)
    w_t1 = smp.Symbol(f'\\omega_{{{beta_spin_name1}}}', real=True)
    w_t2 = smp.Symbol(f'\\omega_{{{beta_spin_name2}}}', real=True)
    
    delta_sec = (q1+q2)*w_s - q1*w_t1 - q2*w_t2

    if delta_sec != 0:

        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            argument = q1*w_t1 + q2*w_t2
            J = J_w(alpha, beta, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        argument = q1*w_t1 + q2*w_t2
        J = J_w(alpha, beta, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()

def sop_R_term_beta_alpha(l, q1, q2, beta, alpha, beta_spin_name1, beta_spin_name2, alpha_spin_name,
                          op_T_left, op_T_right,
                          keep_non_secular=False):
    """
    Term in the relaxation superoperator between a two-spin interaction and a single-spin interaction.
    """
    w_t1 = smp.Symbol(f'\\omega_{{{beta_spin_name1}}}', real=True)
    w_t2 = smp.Symbol(f'\\omega_{{{beta_spin_name2}}}', real=True)
    w_s = smp.Symbol(f'\\omega_{{{alpha_spin_name}}}', real=True)

    delta_sec = q1*w_t1 + q2*w_t2 - (q1+q2)*w_s

    if delta_sec != 0:

        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            argument = (q1+q2)*w_s
            J = J_w(beta, alpha, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        argument = (q1+q2)*w_s
        J = J_w(beta, alpha, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1, 1, q2, l, (q1+q2)).doit()

def sop_R_term_beta_beta(l, q1_t1, q2_t1, q1_t2, q2_t2, beta1, beta2,
                         beta1_spin_name1, beta1_spin_name2, beta2_spin_name1, beta2_spin_name2,
                         op_T_left, op_T_right,
                         keep_non_secular=False):
    """
    Term in the relaxation superoperator between two two-spin interactions.
    """
    w1_t1 = smp.Symbol(f'\\omega_{{{beta1_spin_name1}}}', real=True)
    w1_t2 = smp.Symbol(f'\\omega_{{{beta1_spin_name2}}}', real=True)
    w2_t1 = smp.Symbol(f'\\omega_{{{beta2_spin_name1}}}', real=True)
    w2_t2 = smp.Symbol(f'\\omega_{{{beta2_spin_name2}}}', real=True)

    delta_sec = q1_t1*w1_t1 + q2_t1*w1_t2 - q1_t2*w2_t1 - q2_t2*w2_t2
    
    if delta_sec != 0:

        if keep_non_secular and settings.RELAXATION_THEORY == 'sc':

            argument = q1_t1*w1_t1 + q2_t1*w1_t2
            J = J_w(beta1, beta2, l, argument)

            return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_t1, 1, q2_t1, l, (q1_t1+q2_t1)).doit()\
                                                        * CG(1, q1_t2, 1, q2_t2, l, (q1_t2+q2_t2)).doit()
        
        else:
            return smp.zeros(op_T_left.shape[0]**2, op_T_right.shape[0]**2)
    
    else:
        argument = q1_t1*w1_t1 + q2_t1*w1_t2
        J = J_w(beta1, beta2, l, argument)

        return sop_R_term(op_T_left, J, op_T_right) * CG(1, q1_t1, 1, q2_t1, l, (q1_t1+q2_t1)).doit()\
                                                    * CG(1, q1_t2, 1, q2_t2, l, (q1_t2+q2_t2)).doit()

def sop_R(SpinOperators, INCOHERENT_INTERACTIONS, 
          keep_non_secular=False):
    """
    Matrix representation of the relaxation superoperator in Liouville space.
    NOTE: This is the implementation of the main equations in https://doi.org/10.1016/j.jmr.2024.107828
    
    Input:
        - SpinOperators: SpinOperators object.
        - INCOHERENT_INTERACTIONS: Dictionary of incoherent interactions (see README.md for details)

        - keep_non_secular: Whether to keep non-secular terms in the relaxation superoperator.
            NOTE: Only applicable for the semiclassical relaxation theory.

    Returns:
        - R_final: Matrix representation of the relaxation superoperator in Liouville space.
    """
    # Initialize the relaxation superoperator
    R_final = smp.zeros(SpinOperators.N_states**2, SpinOperators.N_states**2, complex=True)

    # Prepare coupling vector for linear interactions
    # NOTE: assumed always along z-axis
    T_vector = vector_to_spherical_tensor([0, 0, 1])

    print('\nComputing R for interaction pairs...')

    # Loop over all mechanism pairs
    for mechanism1, properties1 in INCOHERENT_INTERACTIONS.items():
        for mechanism2, properties2 in INCOHERENT_INTERACTIONS.items():

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
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name = SpinOperators.spinsystem[spin_1_index]
                                spin_2_name = SpinOperators.spinsystem[spin_2_index]

                                # Loop over all common ranks and components
                                for l in ls:
                                    for q in range(-l, l+1):

                                        # Check if linear or quadratic interaction
                                        if properties1[0][1] == 'L':
                                            T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index], T_vector, l, q)
                                        elif properties1[0][1] == 'Q':
                                            T_left = SpinOperators.T[spin_1_index][l, q]

                                        if properties2[0][1] == 'L':
                                            T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], T_vector, l, q)
                                        elif properties2[0][1] == 'Q':
                                            T_right = SpinOperators.T[spin_2_index][l, q]

                                        R_term = sop_R_term_alpha_alpha(l, q, intr_name1, intr_name2, spin_1_name, spin_2_name, T_left, T_right,
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

                        for (spin_2_index_i, spin_2_index_j), coupling_strength_2 in np.ndenumerate(coupling_strengths_matrix2):
                            if coupling_strength_2 != 0:

                                # Interaction names
                                intr_name1 = mechanism1 + str(spin_1_index + 1)
                                intr_name2 = mechanism2 + str(spin_2_index_i + 1) + str(spin_2_index_j + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name = SpinOperators.spinsystem[spin_1_index]
                                spin_2_name_i = SpinOperators.spinsystem[spin_2_index_i]
                                spin_2_name_j = SpinOperators.spinsystem[spin_2_index_j]

                                for l in ls:
                                    # Loop over q1 and q2 values in the symbolic case
                                    for q1 in range(-1, 2):
                                        for q2 in range(-1, 2):

                                            # Clebsch-Gordan restriction
                                            if np.abs(q1 + q2) <= l:

                                                if properties1[0][1] == 'L':
                                                    T_left = op_T_coupled_lq(SpinOperators.T[spin_1_index], T_vector, l, (q1 + q2))
                                                elif properties1[0][1] == 'Q':
                                                    T_left = SpinOperators.T[spin_1_index][l, (q1 + q2)]

                                                # Interaction 2 is always bilinear if type is '2'
                                                T_right_i = SpinOperators.T[spin_2_index_i]
                                                T_right_j = SpinOperators.T[spin_2_index_j]
                                                T_right = T_right_i[1, q1] @ T_right_j[1, q2]

                                                R_term = sop_R_term_alpha_beta(l, q1, q2, intr_name1, intr_name2, spin_1_name, spin_2_name_i, spin_2_name_j, T_left, T_right,
                                                                               keep_non_secular=keep_non_secular)
                                                R_final += R_term

            # Double-spin single-spin mechanism pair
            elif properties1[0][0] == '2' and properties2[0][0] == '1':
                    
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
                                intr_name1 = mechanism1 + str(spin_1_index_i + 1) + str(spin_1_index_j + 1)
                                intr_name2 = mechanism2 + str(spin_2_index + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name_i = SpinOperators.spinsystem[spin_1_index_i]
                                spin_1_name_j = SpinOperators.spinsystem[spin_1_index_j]
                                spin_2_name = SpinOperators.spinsystem[spin_2_index]

                                for l in ls:
                                    for q1 in range(-1, 2):
                                        for q2 in range(-1, 2):

                                            if np.abs(q1 + q2) <= l:
                                            
                                                T_left_i = SpinOperators.T[spin_1_index_i]
                                                T_left_j = SpinOperators.T[spin_1_index_j]
                                                T_left = T_left_i[1, q1] @ T_left_j[1, q2]

                                                if properties2[0][1] == 'L':
                                                    T_right = op_T_coupled_lq(SpinOperators.T[spin_2_index], T_vector, l, (q1 + q2))
                                                elif properties2[0][1] == 'Q':
                                                    T_right = SpinOperators.T[spin_2_index][l, (q1 + q2)]

                                                R_term = sop_R_term_beta_alpha(l, q1, q2, intr_name1, intr_name2, spin_1_name_i, spin_1_name_j, spin_2_name, T_left, T_right,
                                                                               keep_non_secular=keep_non_secular)
                                                R_final += R_term

            # Double-spin two-spin mechanism pair
            elif properties1[0][0] == '2' and properties2[0][0] == '2':
                
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
                                intr_name1 = mechanism1 + str(spin_1_index_i + 1) + str(spin_1_index_j + 1)
                                intr_name2 = mechanism2 + str(spin_2_index_i + 1) + str(spin_2_index_j + 1)
                                print(f'{intr_name1} * {intr_name2}')

                                # Handle chemically equivalent (homonuclear) spins
                                spin_1_name_i = SpinOperators.spinsystem[spin_1_index_i]
                                spin_1_name_j = SpinOperators.spinsystem[spin_1_index_j]
                                spin_2_name_i = SpinOperators.spinsystem[spin_2_index_i]
                                spin_2_name_j = SpinOperators.spinsystem[spin_2_index_j]

                                for l in ls:
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

                                                        R_term = sop_R_term_beta_beta(l, q1_d1, q2_d1, q1_d2, q2_d2, intr_name1, intr_name2,
                                                                        spin_1_name_i, spin_1_name_j, spin_2_name_i, spin_2_name_j, T_left, T_right,
                                                                        keep_non_secular=keep_non_secular)
                                                        R_final += R_term
      
            else:
                raise ValueError('Invalid interaction dictionary. See README.md for details.')
            
    print('R computed.')
    return R_final

####################################################################################################
# Relaxation superoperator class.
####################################################################################################
class RelaxationSuperoperator(Superoperator):
    """
    General class for the relaxation superoperator of a spin system. Inherits from Superoperator.
    See Superoperator and Operator classes for more information.

    NOTE: The main class of Rela2x.

    Input:
        - sop_R: Relaxation superoperator matrix representation.
        - basis_symbols: List of basis operator symbols.
    """
    def __init__(self, sop_R, basis_symbols, basis_norms):
        Superoperator.__init__(self, sop_R)
        self.basis_symbols = basis_symbols
        self.basis_norms = basis_norms

    def to_observables(self):
        """
        Fix basis operator normalization in order to get the correct relaxation rates
        and equations of motion of observables.
        """
        for i in range(self.op.shape[0]):
            for j in range(self.op.shape[1]):
                self.op[i, j] *= (self.basis_norms[i] / self.basis_norms[j])

        # Simplify the relaxation superoperator
        self.op = smp.simplify(self.op)

    def rate(self, spin_index_lqs_1, spin_index_lqs_2=None):
        """
        Relaxation rate between two basis operators.

        Input:
            - spin_index_lqs_1: String of the first spin index and lq values (see T_symbol_list_index).
            - spin_index_lqs_2: String of the second spin index and lq values. If None, it is the same as spin_index_lqs_1.

        Returns:
            - The relaxation rate between the two basis operators.
        """
        if spin_index_lqs_2 is None:
            spin_index_lqs_2 = spin_index_lqs_1
        index_1 = T_symbol_list_index(self.basis_symbols, spin_index_lqs_1)
        index_2 = T_symbol_list_index(self.basis_symbols, spin_index_lqs_2)
        try:
            return self.op[index_1, index_2]
        except IndexError:
            # print('Invalid basis operator indexes. Try changing the order of the operators in the product.')
            pass

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
            # Include only anisotropic rotationally modulated interactions (l > 0)
            if lq[0] > 0:
                subst_dict[J_w] = J_w_isotropic_rotational_diffusion(*intrs, lq[0], arg,
                                    fast_motion_limit=fast_motion_limit, slow_motion_limit=slow_motion_limit)
        self.substitute(subst_dict)

    def neglect_cross_correlated_terms(self, mechanism1=None, mechanism2=None):
        """
        Neglect cross-correlated terms between mechanism1 and mechanism2 in the relaxation superoperator.
        
        Input:
            - mechanism1: Name of the first mechanism.
            - mechanism2: Name of the second mechanism. If None, mechanism1 is used.

        NOTE: If mechanism1 and mechanism2 are both None, all cross-correlated terms are neglected.
        """
        J_w_functions = [function for function in self.functions_in if 'J' in str(function) or 'G' in str(function)]

        # Neglect all cross-correlated terms if mechanism1 and mechanism2 are both None
        if mechanism1 is None and mechanism2 is None:
            for J_w in J_w_functions:
                # See docstring for extract_J_w_arguments
                intrs, _, _ = extract_J_w_symbols_and_args(J_w)
                if ',' in intrs[0]:
                    self.substitute({J_w: 0})

        # Neglect cross-correlated terms between mechanism1 and mechanism2
        else:
            if mechanism2 is None:
                mechanism2 = mechanism1
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

    def filter(self, filter_name, filter_value):
        """
        Filter out regions of the relaxation superoperator based on given criteria.
        See coherence_order_filter, spin_order_filter and type_filter for more information.

        Input:
            - filter_name: String of filter name. 
                'c' for coherence order, 
                's' for spin order, 
                't' for type.
            - filter_value: List or integer of filter values. (See coherence_order_filter, spin_order_filter and type_filter).
            NOTE: Determines the filter values that are kept.
        """
        if filter_name == 'c':
            self.op, self.basis_symbols = coherence_order_filter(self.op, self.basis_symbols, filter_value)
        elif filter_name == 's':
            self.op, self.basis_symbols = spin_order_filter(self.op, self.basis_symbols, filter_value)
        elif filter_name == 't':
            self.op, self.basis_symbols = type_filter(self.op, self.basis_symbols, filter_value)

        # Update symbols_in and functions_in
        self.symbols_in = self.get_symbols()
        self.functions_in = self.get_functions()

####################################################################################################
# Master equations.
####################################################################################################
def equations_of_motion(R, basis_op_symbols, expectation_values=True, included_operators=None):
    """
    System of differential equations resulting from the master equation of the
    given relaxation theory (defined in RELAXATION_THEORY in settings.py).

    Input:
        - R: Relaxation superoperator matrix representation.
        - basis_op_symbols: List of basis operator symbols.
        - expectation_values: Boolean to display as expectation values (default = True).
        - included_operators: List of indexes to select a subset of basis operators (default = None).
    """
    # Include only a subset of operators if desired
    if included_operators is not None:
        R = pick_from_matrix(R, included_operators)
        basis_op_symbols = pick_from_list(basis_op_symbols, included_operators)

    # Compute the left-hand side of the differential equations
    # Use expectation values if desired
    if expectation_values:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(f_expectation_value_t(x), t))
    else:
        lhs = smp.Matrix(basis_op_symbols).applyfunc(lambda x: smp.Derivative(x, t))

    # Check the relaxation theory
    if settings.RELAXATION_THEORY == 'sc':
        rhs = smp.Matrix([smp.Symbol(f'\\Delta {symbol}'.replace('*', '')) for symbol in basis_op_symbols])
    elif settings.RELAXATION_THEORY == 'qm':
        rhs = smp.Matrix([symbol for symbol in basis_op_symbols])

    # Compute the right-hand side of the differential equations
    if expectation_values:
        rhs = rhs.applyfunc(lambda x: f_expectation_value_t(x))

    rhs = -R * rhs
    return smp.Eq(lhs, rhs, evaluate=False)

def equations_of_motion_to_latex(eqs, savename):
    """
    Convert a master equation to LaTeX.
    NOTE: Saves the LaTeX file to the current working directory.

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
# Combined functions.
####################################################################################################
def R_object_in_T_basis(spinsystem, INCOHERENT_INTERACTIONS,
                        sorting='v1', keep_non_secular=False):
    """
    Compute the relaxation superoperator object, basis, and symbols in the
    product basis of spherical tensor operators.
    
    Input:
        - spinsystem: List of nuclear isotopes (strings) that define the spin system.
        - INCOHERENT_INTERACTIONS: Dictionary of incoherent interactions.
        - sorting: Sorting of the basis operators (default = 'v1').

        - keep_non_secular: Whether to keep non-secular terms in the relaxation superoperator.
            NOTE: Only applicable for the semiclassical relaxation theory.

    Returns:
        - R: Relaxation superoperator object.
    """
    # Create SpinOperators object
    Sops = SpinOperators(spinsystem)

    # Compute the matrix representation of the relaxation superoperator
    R = sop_R(Sops, INCOHERENT_INTERACTIONS, keep_non_secular=keep_non_secular)

    # Compute the direct product basis of the spherical tensor operators
    T_basis, T_symbols, norms = T_product_basis_and_symbols(Sops, sorting=sorting)

    # Create the relaxation superoperator and convert to the product basis
    R = RelaxationSuperoperator(R, T_symbols, norms)
    R.to_basis(T_basis)

    print('\nFinal clean-ups...')
    # Convert the relaxation rates to correspond to observables
    R.to_observables()

    # Expand matrix elements to simplify the expressions
    R.op = R.op.expand()
    print('Done.')

    return R
