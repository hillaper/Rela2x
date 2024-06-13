# ––––– Rela²x –––––

## Description

Rela²x is a freely available Python package that contains a collection of functions and classes for Analytic and Automatic high-field liquid-state NMR relaxation theory. 

The package provides tools to compute, approximate and analyze the Liouville-space matrix representation of the relaxation superoperator, *R*, for various small spin systems with arbitrary spin quantum numbers and relaxation mechanisms. Every possible cross-term between each interaction is included. Approximations and simplifications for the analysis of *R*, as well as visualization tools, are available. Rela²x is designed to be easy to use and understand, requiring only basic knowledge of Python.

## Notes

Before using Rela²x, it is recommended that you have read the related publication []. (There, the Greek letter Gamma is used for the relaxation superoperator, but in Python this is inconvenient, so here, and also in the code, R is used.)

Basic knowledge of Python is assumed. Additional experience with the *SymPy* library can provide helpful because the matrix representation of *R* is a *SymPy* matrix object.

For detailed information on the functions and classes of Rela²x, see the documentation directly in `rela2x.py`.

## Installation

To install and run Rela²x:

1. Clone the repository or download it as a ZIP file and extract it.
2. Navigate to the project directory.
3. Run the provided Jupyter notebooks or create your own.

## Dependencies

The following Python packages are required:

- numpy
- matplotlib
- sympy

The necessary packages are listed in the `requirements.txt` file. Rela²x is designed to be an interactive program, so an installation of 

- Jupyter Notebook

 is also required. The *Anaconda* distribution provides all the necessary packages.

## Usage

The usage of Rela²x is summarized below. Specifics, such as variable names, are up to you.

1. Import `rela2x.py`:

    - `from rela2x import *`

    Usually `*` imports are not recommended, but Rela²x is a relatively small library so this is not an issue. It is actually quite convenient to have all the functions in Jupyter Notebook's memory space for automatic recommendations and, *e.g.*, function docstrings while coding.

2. Define the spin system:

    Spin systems are defined via a list of isotope names. For instance:

    - `spin_system = ['14N', '1H', '1H']`

    A collection of NMR isotopes and their spin quantum numbers are listed in `nmr_isotopes.py`. The values are taken from https://www.kherb.io/docs/nmr_table.html . If your favourite nucleus is not listed, feel free to add it :-).

3. Choose general settings (optional):

    Rela²x supports three general settings included in the `settings.py` file.

    - `RELAXATION_THEORY` handles the level of theory used: semiclassical `'sc'`, or quantum mechanical (Lindbladian) `'qm'` [].
    
    - `FRAME` determines whether *R* is computed in the rotating `'rot'` or laboratory `'lab'` frame.
    
    - `SECULAR` is used to turn the secular approximation on `'True'` or off `'False'`. This affects calculations only in the rotating frame.

    So, the possible values (default) are

    - `RELAXATION_THEORY = ('sc')` or `'qm'`
    - `FRAME = ('rot')` or `'lab'`
    - `SECULAR = (True)` or `False`

    The easiest way to access these is through `set_relaxation_theory`, `set_frame`, and `set_secular` functions. For instance

    - `set_relaxation_theory('qm')`

    could be called for Lindbladian description of *R* [].

    Note that Rela²x handles homonuclear spins automatically, even in the rotating frame when the secular approximation is used. The arguments of the rapidly oscillating complex exponentials are explicitly checked.

4. Define the incoherent interactions that drive relaxation:

    Incoherent interactions are defined via a Python dictionary with key-value pairs of the following type:

    `'mechanism_name': ('type', interaction_array, rank_list)`

    - `mechanism_name` appears in the spectral-density function symbols and is mostly a cosmetic label that does not affect the actual calculation. However, these names are utilised if cross-correlated couplings are neglected (see below).

    - For single-spin linear or single-spin quadratic interactions, `type` is either `'SL'` or `'SQ'`, respectively. For double-spin bilinear interactions, type is always `'D'`. Bilinearity of double-spin interactions does not need to be specified.

    - The `interaction_array` for single-spin mechanisms is a Python list of values `1` or `0`, defining which spins in `spin_system` are included in that interaction. For double-spin mechanisms, a coupling matrix (list of lists) is provided where the `1`s define which spins are coupled. Only the upper triangle needs to be provided.

    - `rank_list` is a list of ranks *l* included in the given mechanism.

    For instance, for our example `spin_system = ['14N', '1H', '1H']` with chemical-shift anisotropy (including all ranks) and quadrupolar interactions on ¹⁴N, and dipole-dipole couplings between all of the spins, we would have:

    - `INCOHERENT_INTERACTIONS = {`  
        `'CSA': ('SL', [1, 0, 0], [0, 1, 2]),`  
        `'Q': ('SQ', [1, 0, 0], [2]),`  
        `'DD': ('D', [[0, 1, 1], [0, 0, 1], [0, 0, 0]], [2])}`

5. Compute the matrix representation of *R*, convert it to a suitable product operator basis and create a `RelaxationSuperoperator` object:

    It is useful to represent *R* in a basis where it obtains a block-diagonal form. The best basis for this purpose is the direct product basis of spherical tensor operators []. All of this is automatically done by calling:

    - `R, T_basis, T_symbols = R_object_and_T_basis(spin_system, INCOHERENT_INTERACTIONS, sorting='v1')`

    The `R_object_and_T_basis` function takes as input the `spin_system` and `INCOHERENT_INTERACTIONS` variables as defined above, and optionally information about how to sort the operator basis via `sorting`. Three options are available: `'v1'`, `'v2'` or `None` (for details, see documentation in `rela2x.py`).
    
    The function returns a `RelaxationSuperoperator` object, a list of basis operators and a list of corresponding operator symbols. The `RelaxationSuperoperator` object has the following attributes:

    - `.op` returns the matrix representation of *R*

    - `.symbols_in` returns all symbols appearing in *R*

    - `.functions_in` returns all functions appearing in *R*

    and functions:

    - `rate(spin_index_lqs_1, spin_index_lqs_2=None)` returns the relaxation rate between two basis operators. The `spin_index_lqs` arguments have to be strings of the form `'110'`, where the first number refers to the index of the spin, the second numbers to the rank *l* and the third number to the component *q* of that operator. Product operators are simply of the form `'110*210'`. Providing `spin_index_lqs_1` only will return the auto-relaxation rate of that operator, whereas if `spin_index_lqs_2` is also provided the cross-relaxation rate between those two operators is returned (see the examples provided in the repository)

    - `.to_basis(basis)` performs a change of basis using a list of basis operators `basis`

    - `.to_isotropic_rotational_diffusion(fast_motion_limit=False, slow_motion_limit=False)` applies the isotropic rotational diffusion model with the fast-motion or slow-motion limit approximation if desired

    - `.neglect_cross_correlated_terms(mechanism1, mechanism2)` neglects cross-correlated contributions in *R* between two mechanisms. The arguments `mechanism1` and `mechanism2` have to correspond to the names chosen for `mechanism_name`s in `INCOHERENT_INTERACTIONS`. The same mechanism can be provided twice to, *e.g.*, neglect cross-correlated dipole-dipole couplings

    - `.neglect_ALL_cross_correlated_terms()` neglects all cross-correlated contributions

    - `.substitute(substitutions_dict)` substitutes symbols and functions in *R* with given numerical values (allows easy conversion to NumPy arrays for numerical use)

    - `.visualize(rows_start=0, rows_end=None, basis_symbols=None, fontsize=None)` visualizes *R* as a matrix plot. If desired, only certain sections of *R* can be visualised via `rows_start` and `rows_end`. A legend where the basis-operator symbols appear is drawn if `basis_symbols` is provided. Fontsize can be adjusted for large matrices.
    
    Best way to get acquainted is to try these for yourself :-).

6. After *R* is computed, the resulting relaxation equations of motion for the observables can be constructed. Different functions need to be used depending on `RELAXATION_THEORY`, because the semiclassical  and Lindbladian forms of the master equation are different:

    - `eoms = ime_equations_of_motion(R.op, T_symbols, expectation_values=True, included_operators=None)` uses the semiclassical (or inhomogeneous) master equation (IME)

    - `eoms = lindblad_equations_of_motion(R.op, T_symbols, expectation_values=True, included_operators=None)` uses the Lindbladian (quantum mechanical) master equation.

    Here, `R.op` is the matrix representation of *R*, `T_symbols` is the list of basis operator symbols and the rest are for cosmetic purposes (try it yourself). The returned `eoms` is a *SymPy* equation object.

7. The equations of motion can be automatically saved in LaTeX format to the current working directory as a .txt file for further use in, *e.g.*, publications:

    - `equations_of_motion_to_latex(eoms, savename)`.

    `savename` is an arbitrary string.

## Examples

Three example notebooks that showcase the usage of Rela²x are included in the repository.

## Warnings

Rela²x is not designed, and should hence be used with caution for spin systems where the number of rows in *R* is in the hundreds. In particular, displaying the entire operator might lead to Jupyter Notebook crashing.

## Advanced users

Additional features not covered can be found in `rela2x.py`. The code is well documented and advanced Python/SymPy users should find it relatively easy to go through.

## Contributing

-

## License

MIT?

## Contact Information

If you have questions, comments, or suggestions, please feel free to reach out:

Email: perttu.hilla@oulu.fi

LinkedIn: www.linkedin.com/in/perttu-hilla-19777a22a

## Citations

- Hilla, Vaara

- Bengs, Levitt