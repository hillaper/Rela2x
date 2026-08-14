"""
Master equations of motion.

The equations of motion for the expectation values of the basis operators are
assembled here, in the form dictated by the relaxation theory in use and by the
sign convention of the superoperator generating the motion.
"""

# NOTE: Postponed evaluation of annotations, so that the modern union syntax can be
# used in type hints while the package keeps supporting the Python versions declared
# in pyproject.toml.
from __future__ import annotations

import sympy as smp

from rela2x._core import _settings
from rela2x._core._constants import t
from rela2x._core._superoperators import Superoperator
from rela2x._core._symbols import f_expectation_value_t
from rela2x._core._utils import pick_from_list, pick_from_matrix


def equations_of_motion(
    superoperator: Superoperator,
    expectation_values: bool=True,
    included_operators: list[int] | None=None,
) -> smp.Eq:
    """
    Build the system of differential equations resulting from the master
    equation of the relaxation theory set in `_settings.RELAXATION_THEORY`.

    NOTE: The superoperator object is taken as such, rather than its matrix
    representation, because the sign convention of the equations of motion
    differs between the superoperators: the relaxation superoperator is
    positive-definite and enters with a minus sign, whereas the Liouvillian is
    already the generator of the motion. Each object reports its own convention
    through its `generator` property, so that relaxation superoperators,
    Hamiltonian superoperators and Liouvillians are all handled here without
    further branching.

    NOTE: For the semiclassical relaxation theory the equations are written for
    the deviations from thermal equilibrium. Under the high-field secular
    approximation the coherent part commutes with the equilibrium state, so
    that separation is exact; it holds only approximately if the non-secular
    terms are retained (see `LiouvillianSuperoperator`).

    Parameters
    ----------
    superoperator : Superoperator
        Superoperator generating the motion, e.g. a `RelaxationSuperoperator`
        or a `LiouvillianSuperoperator`.
    expectation_values : bool, optional
        Whether to display the equations in terms of expectation values. Default is True.
    included_operators : list of int, optional
        Indices selecting a subset of basis operators to include. If None,
        all basis operators are included. Default is None.

    Returns
    -------
    sympy.Eq
        System of differential equations for the observables.
    """
    # Take the generator and the basis operator symbols from the superoperator.
    generator = superoperator.generator
    basis_symbols = superoperator.basis_symbols

    # Include only a subset of operators if desired.
    if included_operators is not None:
        generator = pick_from_matrix(generator, included_operators)
        basis_symbols = pick_from_list(basis_symbols, included_operators)

    # Compute the left-hand side of the differential equations, optionally as expectation values.
    if expectation_values:
        lhs = smp.Matrix(basis_symbols).applyfunc(lambda x: smp.Derivative(f_expectation_value_t(x), t))
    else:
        lhs = smp.Matrix(basis_symbols).applyfunc(lambda x: smp.Derivative(x, t))

    # Build the right-hand side according to the relaxation theory: deviations
    # from thermal equilibrium for the semiclassical theory, or the
    # operators themselves for the quantum mechanical (Lindbladian) theory.
    if _settings.RELAXATION_THEORY == 'sc':
        rhs = smp.Matrix([smp.Symbol(f'\\Delta {symbol}'.replace('*', '')) for symbol in basis_symbols])
    elif _settings.RELAXATION_THEORY == 'qm':
        rhs = smp.Matrix([symbol for symbol in basis_symbols])

    # Compute the right-hand side of the differential equations.
    if expectation_values:
        rhs = rhs.applyfunc(lambda x: f_expectation_value_t(x))

    rhs = generator * rhs
    return smp.Eq(lhs, rhs, evaluate=False)


def equations_of_motion_to_latex(
    eqs: smp.Eq,
    savename: str,
) -> None:
    """
    Convert a system of equations of motion to LaTeX and save it to file.

    NOTE: Saves the LaTeX source to a file in the current working directory.

    Parameters
    ----------
    eqs : sympy.Eq
        System of differential equations, as returned by `equations_of_motion`.
    savename : str
        Name used to construct the saved file name, ``EOMs_{savename}.txt``.
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
