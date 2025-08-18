# ––––– Rela²x –––––
# *A*nalytic and *A*utomatic NMR relaxation theory

## Description

Rela²x is a freely available Python package that offers a collection of functions and classes for analytic and automatic high-field liquid-state NMR relaxation theory.

The package provides tools to compute and analyze the Liouville-space matrix representation of the relaxation superoperator, *R*, for arbitrary small spin systems with any spin quantum numbers and relaxation mechanisms. It includes every possible cross-term between the interactions. Approximations, simplifications for the analysis of *R*, and visualization tools are also available. Rela²x is designed to be user-friendly, requiring only a basic knowledge of Python.

## Notes

Before using Rela²x, it is recommended that you read the related publication https://doi.org/10.1016/j.jmr.2024.107828. (There, the Greek letter Gamma is used for the relaxation superoperator; however, in Python, this is inconvenient, so *R* is used here and in the code.)

Only basic knowledge of Python is required. Additional experience with the *SymPy* library can be helpful because it is the main library used by Rela²x.

For detailed information on the functions and classes of Rela²x, refer to the documentation directly in `rela2x.py`.

## Installation from PyPI
Rela²x is available from the Python Package Index (PyPI) repository:

   ```python
   pip install rela2x
   ```

## Installation from source
To install the version from GitHub manually:

1. Ensure the `build` module is installed:

      ```python
      pip install build
      ```

2. Download the source code archive (.zip or .tar.gz).

3. Extract the archive.

4. Navigate to the extracted folder:

      ```python
      cd /your/path/rela2x
      ```
      
5. Build the wheel from the source:

      ```python
      python -m build --wheel
      ```

6. Navigate to the dist folder:

      ```python
      cd /your/path/rela2x/dist
      ```

7. Install the wheel using pip:
      pip install rela2x-0.0.1-py3-none-any.whl

## Dependencies

 The following Python packages are required:

- numpy
- matplotlib
- sympy

These necessary packages are listed in the `requirements.txt` file. Rela²x is designed to be an interactive program, so an installation of 

- Jupyter Notebook

 is also required. The *Anaconda* distribution includes all the necessary packages and is recommended for ease of setup.

## Usage

The usage of Rela²x is summarized below. Specifics, such as variable names, can be customized as needed.

**Import `rela2x.py`:**

   ```python
   from rela2x import *
   ```
   
   Although wildcard imports (`*`) are generally not recommended, Rela²x is a relatively small library, so this is not an issue. It is quite convenient to have all the functions in Jupyter Notebook's memory space for automatic recommendations and, for example, function docstrings while coding.
   
**Define the spin system:**

   Spin systems are defined via a list of isotope names. For instance:
   
   ```python
   spin_system = ['14N', '1H', '1H']
   ```
   
   A collection of NMR isotopes and their spin quantum numbers is listed in `nmr_isotopes.py`. The values are sourced from [this NMR table](https://www.kherb.io/docs/nmr_table.html). If your preferred nucleus is not listed, feel free to add it!
   
**Choose general settings (optional):**

   Rela²x currently supports one general setting included in the `settings.py` file.
   
   - `RELAXATION_THEORY` handles the level of theory used: semiclassical `'sc'`, or quantum mechanical (Lindbladian) `'qm'`.
   
   So, the possible values are:
   
   - `RELAXATION_THEORY = 'sc'` or `'qm'`
   
   where the default value is the first one. The easiest way to access this is through the `set_relaxation_theory` function:
   
   ```python
   set_relaxation_theory('qm')
   ```
   
   could be called for the Lindbladian description of *R*.
   
**Define the incoherent interactions that drive relaxation:**

   Incoherent interactions are defined via a Python dictionary with key-value pairs of the following type:
   
   `'mechanism_name': ('type', intr_array, rank_list)`
   
   - `mechanism_name` appears in the spectral-density function symbols and is mostly a cosmetic label that does not affect the actual calculation. However, these names are utilized if cross-correlated couplings are neglected (see below).
   
   - For single-spin linear or single-spin quadratic interactions, `type` is either `'1L'` or `'1Q'`, respectively. For two-spin bilinear interactions, `type` is always `'2'`. Bilinearity of two-spin interactions does not need to be specified.
   
   - The `intr_array` for single-spin mechanisms is a Python list of values `1` or `0`, defining which spins in `spin_system` are included in that interaction. For two-spin mechanisms, a coupling matrix (list of lists) is provided where the `1`s define which spins are coupled. Only the upper triangle needs to be provided.
   
   - `rank_list` is a list of ranks *l* of the given mechanism.
   
   For instance, for our example `spin_system = ['14N', '1H', '1H']` with chemical-shift anisotropy (including all ranks) and quadrupolar interactions on ¹⁴N, and dipole-dipole couplings between all of the spins, we would have:
   
   ```python
   intrs = {
       'CSA': ('1L', [1, 0, 0], [0, 1, 2]),
       'Q':   ('1Q', [1, 0, 0], [2]),
       'DD':  ('2', [[0, 1, 1],
                     [0, 0, 1], 
                     [0, 0, 0]], 
                     [2])
   }
   ```
   
**Compute the matrix representation of *R*, convert it to the product operator basis of spherical tensor operators, and create a `RelaxationSuperoperator` object:**

   It is useful to represent *R* in a basis where it achieves a block-diagonal form. A good basis for this purpose is the direct product basis of spherical tensor operators. This is automatically done by calling:
   
   ```python
   R = R_object_in_T_basis(spin_system, intrs, sorting='v1', keep_non_secular=False)
   ```

   The `R_object_in_T_basis` function takes as input the `spin_system` and `intrs` variables as defined above, and optionally information about how to sort the operator basis via `sorting`. Three options are available: `'v1'`, `'v2'`, or `None` (for details, see the documentation in `rela2x.py`). `keep_non_secular` allows to keep non-secular terms in the relaxation superoperator.

   The function returns a `RelaxationSuperoperator` object that has the following attributes:

   - `op` returns the matrix representation of *R*.
   - `symbols_in` returns all symbols appearing in *R*.
   - `functions_in` returns all functions appearing in *R*.
   - `basis_symbols` returns all basis operator symbols corresponding to the direct product basis of spherical tensor operators.

   And functions:

   - `to_basis(basis)` performs a change of basis using a list of basis operators `basis`.

   - `substitute(substitutions_dict)` substitutes symbols and functions in *R* with given numerical values. This allows easy conversion to NumPy arrays for numerical use.

   - `visualize(rows_start=0, rows_end=None, basis_symbols=None, fontsize=None)` visualizes *R* as a matrix plot. If desired, only certain sections of *R* can be visualized via `rows_start` and `rows_end`. A legend with the basis operator symbols will be drawn if `basis_symbols` is provided. Font size can be adjusted for large matrices.

   - `rate(spin_index_lqs_1, spin_index_lqs_2=None)` returns the relaxation rate between two basis operators. The `spin_index_lqs` arguments must be strings of the form `'110'`, where the first number refers to the index of the spin, the second number refers to the rank *l*, and the third number refers to the component *q* of that operator. Product operators are simply of the form `'110*210'`. Providing `spin_index_lqs_1` only will return the auto-relaxation rate of that operator. If `spin_index_lqs_2` is also provided, the cross-relaxation rate between those two operators is returned (see the examples provided in the repository).

   - `to_isotropic_rotational_diffusion(fast_motion_limit=False, slow_motion_limit=False)` applies the isotropic rotational diffusion model with the fast-motion or slow-motion limit approximation if desired.

   - `neglect_cross_correlated_terms(mechanism1=None, mechanism2=None)` neglects cross-correlated contributions in *R* between two mechanisms. The arguments `mechanism1` and `mechanism2` must correspond to the names chosen for `mechanism_name`s in `intrs`. If `mechanism2` is not provided, `mechanism1` is used, and if neither is provided, all cross-correlated contributions are neglected.

   - `filter(filter_name, filter_value)` filters out potentially uninteresting regions of *R* based on given criteria. `filter_name` must be one of the following: 'c' for coherence order, 's' for spin order, or 't' for type. This determines the criteria for filtration. `filter_value` is an integer or a list of integers depending on the filtration type (see the documentation in `rela2x.py`) and determines which values are kept (not filtered out) in *R*. For instance, calling `R.filter('c', [0])` would filter out those sections that correspond to basis operators with coherence order other than 0.

   The best way to get acquainted is to try these functions yourself!
   
**After *R* is computed, construct the resulting relaxation equations of motion for the observables:**

   ```python
   eoms = equations_of_motion(R.op, R.basis_symbols, expectation_values=True, included_operators=None)
   ```

   Here, `R.op` is the matrix representation of *R*, `R.basis_symbols` is the list of basis operator symbols, and the rest are for cosmetic purposes (try it yourself). The returned `eoms` is a *SymPy* equation object. The outcome depends on `RELAXATION_THEORY`, because the semiclassical and Lindbladian master equations are different.
   
**Save the equations of motion in LaTeX format to the current working directory as a .txt file for further use in, for example, publications:**

   ```python
   equations_of_motion_to_latex(eoms, savename)
   ```

   `savename` is an arbitrary string.

## Examples

Four example notebooks that showcase the usage of Rela²x are included in the repository.

## Warnings

Rela²x is not designed for spin systems where the dimension of *R* exceeds ~150, and should be used with caution in such cases. Specifically, displaying the entire matrix `R.op` may cause Jupyter Notebook to crash. Large systems can nevertheless be computed, and the `rate` function can be useful in these scenarios.

## Advanced Users

Additional features not covered in this guide can be found in `rela2x.py`. The code is fairly well-documented, and advanced Python/SymPy users should find it relatively straightforward to navigate.

## License

Rela²x is licensed under the MIT License. See the LICENSE file in the repository for more details.

## Contact Information

If you have questions, comments, or suggestions, please feel free to reach out:

Email: perttu.hilla@oulu.fi

## Citations

If you use Rela²x in your work, please include the following citation:


P. Hilla, J. Vaara, Rela²x: Analytic and automatic NMR relaxation theory, *J. Magn. Reson.*, 2025; https://doi.org/10.1016/j.jmr.2024.107828
