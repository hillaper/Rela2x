# ––––– Rela²x –––––

## Description

Rela²x is a Python package containing a collection of functions and classes for Analytic and Automatic high-field liquid-state NMR relaxation theory. The package provides tools to compute and analyze the matrix representation of the relaxation superoperator, *R*, for various small spin systems with arbitrary spin quantum numbers and interactions. Every possible cross-term between the interactions is included. Different approximations and simplifications for the analysis of *R*, as well as visualization tools, are available. Rela²x is designed to be easy to use and understand, requiring only basic knowledge of Python.

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

The necessary packages are also listed in the `requirements.txt` file.

Rela²x is designed to be an interactive program, so an installation of Jupyter Notebook is also required. For instance, the *Anaconda* distribution provides all necessary packages.

## Usage

The usage of Rela²x is summarized below (specifics, such as variable names, are up to you). Basic experience with Python, particularly with the *SymPy* library, can provide helpful, since the matrix representation of *R* is a *SymPy* matrix object.

1. Import `rela2x.py`:

    - `from rela2x import *`

2. Define the spin system. Spin systems are defined by providing a list of spin quantum numbers:

    - `S = [1, 1/2, 1/2]`

    Rela²x can handle homonuclear spins in the spin system, even when the secular approximation is used. No Kronecker deltas are used, but instead, the argument of the complex exponential is explicitly checked (see `rela2x.py`). A list of spin names can be provided, where the homonuclear spins share the same name:

    - `spin_names = ['N', 'H', 'H']`

    This list will be used as input when calling the `sop_R` function (see below) to generate the relaxation superoperator *R*. Importantly, spins with the same name share the same Larmor frequency.

3. Create a `SpinOperators` object. `SpinOperators` holds the matrix representations of all necessary operators. A list of spin quantum numbers (here `S`) is required as input:

    - `Sops = SpinOperators(S)`

4. Choose general settings (optional). Rela²x supports three general settings included in the `settings.py` file: `RELAXATION_THEORY`, `FRAME`, and `SECULAR`:

    The default values are

    - `RELAXATION_THEORY = 'sc'`
    - `FRAME = 'rot'`
    - `SECULAR = True`

    and possible values (in the same order) are

    - `'sc'` or `'qm'`
    - `'rot'` or `'lab'`
    - `True` or `False`

    `RELAXATION_THEORY` handles the level of theory used, semiclassical, or quantum mechanical (see Ref. 2). The `FRAME` variable determines whether *R* is computed in the rotating or laboratory frame. `SECULAR` is used to turn the secular approximation on or off.

    The three variables can be accessed through `set_relaxation_theory`, `set_frame`, and `set_secular` functions defined in `rela2x.py`. For instance

    - `set_relaxation_theory('qm')`

    could be called.

5. Define the incoherent interactions that drive relaxation. The interactions are defined by providing a Python dictionary with key-value pairs of the following type:

    - `'mechanism_name': ('type', interaction_array, rank_list)`

    `mechanism_name` appears in the spectral-density function symbols and is mostly a cosmetic label that does not affect the actual calculation. However, these names are utilized if cross-correlated couplings are neglected (see below).

    For single-spin interactions, `type` is either `'SL'` or `'SQ'` for Single-spin Linear or Single-spin Quadratic interaction, respectively. For Double-spin interactions, type is always `'D'`. Bilinearity of double-spin interactions does not need to be specified.

    The `interaction_array` for single-spin mechanisms is a list of values `1` or `0`, defining which spins in the spin system (as in `S`) are included in that interaction. For double-spin mechanisms, a coupling matrix is provided where the `1`s define which spins are coupled. Only the upper triangle needs to be provided.

    `rank_list` is a list of ranks *l* included in the given mechanism.

    For instance, for the example spin system `S = [1, 1/2, 1/2]` with SHielding, Quadrupolar, and Dipole-Dipole mechanisms, we could have:

    - `INCOHERENT_INTERACTIONS = {`  
        `'SH': ('SL', [1, 1, 1], [0, 1, 2]),`  
        `'Q': ('SQ', [1, 0, 0], [2]),`  
        `'DD': ('D', [[0, 1, 1], [0, 0, 1], [0, 0, 0]], [2])`  
        `}`

6. Compute the relaxation superoperator using `sop_R`. Here we can provide information about heteronuclear spins through the optional `spin_names` argument:

    - `R = sop_R(Sops, INCOHERENT_INTERACTIONS, spin_names=spin_names)`

7. Create a `RelaxationSuperoperator` object:

    - `R = RelaxationSuperoperator(R)`

    This object contains useful attributes,

    - `op`, which returns the matrix representation of *R*,

    and functions:

    - `to_basis(basis)`, which performs a change of basis using a list of basis operators `basis`

    - `to_isotropic_rotational_diffusion(fast_motion_limit=False, slow_motion_limit=False)`, which applies the isotropic rotational diffusion model and the fast-motion or slow-motion limit approximations if desired

    - `neglect_cross_correlated_terms(mechanism1, mechanism2)`, which neglects cross-correlated contributions in *R* between the two mechanisms, corresponding to two strings `mechanism_name` in `INCOHERENT_INTERACTIONS`

    - `neglect_ALL_cross_correlated_terms()`, neglects all cross-correlated contributions

    - `visualize(rows_start=0, rows_end=None, basis_symbols=None, fontsize=None)`, which visualizes *R* as a matrix plot. Only certain sections can also be visualized, and the set of basis symbols provided for a convenient legend. Fontsize can be adjusted for large matrices.

    Best way to get acquainted is to try these for yourself.

8. It is useful to represent *R* in a basis where it obtains a block-diagonal form. The best basis for this purpose is the direct product basis of spherical tensor operators (STO). Rela²x provides functions that compute the STO basis and the corresponding operator symbols automatically, given the `SpinOperators` object:

    - `T_basis = T_product_basis(Sops, normalize=True)`

    - `T_symbols = T_product_basis_symbols(Sops)`

    `T_product_basis` returns a list of basis operators (matrices), and `T_symbols` a list of operator symbols. The basis should always be normalized, but can be controlled via the optional `normalize` argument. 
    
    It is convenient to sort the basis so that the block-diagonal form of *R* is more apparent:

    - `quick_sort_T_product_basis(T_product_basis, T_product_basis_symbols, sorting='v1')`

    Two options are supported for the `sorting` argument: `'v1'`, and `'v2'`. See the documentation in `rela2x.py` for more details. For large spin systems, the sorting becomes very complicated, so trying both options can be useful.
    
    Computing the STO basis and symbols, and sorting them, are conveniently combined by calling:

    - `T_basis, T_symbols = T_product_basis_and_symbols(Sops, normalize=True, sorting=None)`

    Here `sorting` also accepts `None` which corresponds to no sorting.

9. On top of the semiclassical (`'sc'`) formulation of NMR relaxation, Rela²x also supports the Lindbladian (`'qm'`) version (see Ref. 2). This is controlled through the `RELAXATION_THEORY` variable.

10. After *R* is computed, the resulting relaxation equations of motion (EOMs) for the observables can be constructed. Different functions should be used depending on `RELAXATION_THEORY`:

    - `eoms = ime_equations_of_motion(R, basis_op_symbols, expectation_values=True, operator_indexes=None)`, which is the inhomogeneous master equation

    or

    - `eoms = lindblad_equations_of_motion(R, basis_op_symbols, expectation_values=True, operator_indexes=None)`, which is the Lindbladian master equation.

    `R` is the matrix representation of *R* (most easily acquired with the `op` attribute of `RelaxationSuperoperator`), `basis_op_symbols` is a list of symbols for the basis operators and the rest are for cosmetic purposes (try it yourself).

11. The EOMs can be automatically saved as a LaTeX expression to the current working directory for further use:

    - `equations_of_motion_to_latex(eoms, savename)`.

## Examples

Two example notebooks that showcase the usage of Rela²x are included in the repository.

## Warnings

Rela²x should be used with caution for spin systems where the number of rows in *R* is in the hundreds. In particular, displaying the entire operator might lead to Jupyter crashing.

## Advanced

Some additional features not covered here are implemented in `rela2x.py`. Advanced users can find these features from therein.

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

- Bengs, Levitt?