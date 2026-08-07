# ––––– Rela²x –––––
# *A*nalytic and *A*utomatic NMR relaxation theory

## Description

Rela²x is a freely available Python package that offers a collection of functions and classes for analytic and automatic high-field liquid-state NMR relaxation theory (and spin physics in general). 

The package provides tools to compute and analyze the Liouville-space matrix representation of the relaxation superoperator, *R*, for arbitrary small spin systems with any spin quantum numbers and relaxation mechanisms. It includes every possible cross-term between the interactions that drive relaxation. Approximations and simplifications for the analysis of *R*, and visualization tools, are also available. Rela²x is designed to be user-friendly, requiring only a basic knowledge of Python.

## Releases

**Rela²x 0.0.1:**  
Initial release of the Rela2x program described in:

P. Hilla, J. Vaara, Rela²x: Analytic and automatic NMR relaxation theory, *J. Magn. Reson.*, 2025;  
[https://doi.org/10.1016/j.jmr.2024.107828](https://doi.org/10.1016/j.jmr.2024.107828)

**Rela²x 0.0.2:**  
New features, bug fixes, type hints, naming consistency, documentation and cosmetic improvements. 
<!-- The main additions are the Cartesian product operator basis (`basis='C'`) and analytical operator decomposition (`op_decomposition`).  -->

The most significant fixes concern the sorting of the product operator basis. The order in which the basis operators appear therefore differs from version 0.0.1, and from the figures in the publication; the matrix elements themselves are unaffected.

Some names changed in this version:

| 0.0.1 | 0.0.2 |
| --- | --- |
| `R_object_in_T_basis` | `R_object_in_product_operator_basis` |
| `KroneckerProduct` | `Kronecker_product` |
| `T_symbol_spin_order`, `T_symbol_coherence_order`, `T_symbol_type`, `T_symbol_Nth_spin_projection` | `T_index_spin_order`, `T_index_coherence_order`, `T_index_type`, `T_index_spin_projection` |
| `T_symbol_list_index` | `basis_index_list_index` |
| `full_sort_T_product_basis`, and the individual sorting passes | `sort_T_product_basis`, together with `T_basis_sort_keys` |
| `T_basis_split_to_coherence_orders` | removed; use `R.filter('c', ...)` |

<!-- The `T_symbol_*` functions determined the properties of a basis operator by reading its printed symbol. The `T_index_*` functions that replace them read the basis operator indices instead (see Usage below). -->

The full changelog is in the [0.0.2 release notes](https://github.com/hillaper/Rela2x/releases/tag/0.0.2). 

## Notes

Before using Rela²x, it is recommended that you read the related publication https://doi.org/10.1016/j.jmr.2024.107828. There, the Greek letter Gamma is used for the relaxation superoperator; however, in Python, this is inconvenient, so *R* is used here and in the code. 

**Important:**
This documentation contains the most up-to-date information regarding the code itself, and it differs in some respects from the publication. The underlying theory is the same, but the code has been updated and improved since the publication.

Only basic knowledge of Python is required. Additional experience with the *SymPy* library can be helpful because it is the main library used by Rela²x.

For detailed information on the functions and classes of Rela²x, refer to the documentation directly in `rela2x.py`. (A thorough dedicated documentation page is planned for the future.)

## Installation from PyPI
The most recent release published on the Python Package Index (PyPI) can be installed with:

   ```bash
   pip install rela2x
   ```

   Note that the PyPI release may lag behind the repository. To get the current state of the code, install from source as described below.

## Installation from source
To install the current version from GitHub manually:

1. Ensure the `build` module is installed:

      ```bash
      pip install build
      ```

2. Download the source code archive (.zip or .tar.gz).

3. Extract the archive.

4. Navigate to the extracted folder:

      ```bash
      cd /your/path/rela2x
      ```
      
5. Build the wheel from the source:

      ```bash
      python -m build --wheel
      ```

6. Navigate to the dist folder:

      ```bash
      cd /your/path/rela2x/dist
      ```

7. Install the built wheel using pip (adjust the filename to match the version you built):
      ```bash
      pip install rela2x-<version>-py3-none-any.whl
      ```

## Dependencies

The following Python packages are required:

- numpy
- matplotlib
- sympy

These dependencies are listed in `pyproject.toml`. Rela²x is designed to be an interactive program, so a Jupyter Notebook installation is also recommended. 
<!-- The *Anaconda* distribution includes all the necessary packages and is recommended for ease of setup. -->

## Usage

The usage of Rela²x is summarized below. Specifics, such as variable names, can be customized as needed. (See also the example notebooks included in the repository.)

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
   
   The default value is `'sc'`. Note that `RELAXATION_THEORY` is not brought into your namespace by `from rela2x import *`, so assigning to a bare `RELAXATION_THEORY` in your own script has no effect. Set it through the `set_relaxation_theory` function instead. For instance:
   
   ```python
   set_relaxation_theory('qm')
   ```
   
   selects the Lindbladian description of *R*.
   
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
   
**Compute the matrix representation of *R*, convert it to the product operator basis, and create a `RelaxationSuperoperator` object:**

   ```python
   R = R_object_in_product_operator_basis(spin_system, intrs, basis='T', sorting='v1', keep_non_secular=False)
   ```

   The `R_object_in_product_operator_basis` function takes as input the `spin_system` and `intrs` variables as defined above, information about which product operator basis to use, and optionally about how to sort the basis via `sorting`. It is useful to represent *R* in a basis where it achieves a block-diagonal form. A good basis for this purpose is the direct product basis of spherical tensor operators, provided via `basis='T'`. For a system of spin-1/2 nuclei, the Cartesian product operator basis can also be used by choosing `basis='C'`.
   
   Three options are available for `sorting` (currently only supported for the spherical tensor basis): `'v1'`, `'v2'`, or `None` (for details, see the documentation in `rela2x.py`). `keep_non_secular` allows to keep non-secular terms in the relaxation superoperator.

   Note that the non-unit norms of observables are taken into account in the relaxation rates, i.e., the matrix elements. The rates directly correspond to observables.

   The function returns a `RelaxationSuperoperator` object that has the following attributes:

   - `op` returns the matrix representation of *R*.
   - `symbols_in` returns all symbols appearing in *R*.
   - `functions_in` returns all functions appearing in *R*.
   - `basis_symbols` returns all basis operator symbols corresponding to the chosen direct product operator basis.
   - `basis_indices` returns the basis operator indices (see below). Each entry of `basis_indices` is a tuple describing one basis operator, holding one item per spin that carries something other than the identity operator. For the spherical tensor basis the items are `(spin index, l, q)` triples, and for the Cartesian basis they are `(spin index, direction)` pairs. Spin indices start from 1, matching the operator symbols. The identity operator of the whole system is described by an empty tuple. For instance, in a two-spin system, $\hat T_{10}^{(1)} \hat T_{1-1}^{(2)}$ has the index `((1, 1, 0), (2, 1, -1))`.

   <!-- The `T_index_*` functions use these to determine the spin order, coherence order, type and individual spin projections of a basis operator, and the sorting and filtering tools are built on them. -->

   <!-- `RelaxationSuperoperator` has also the following methods: -->
   and the following methods:

   - `to_basis(basis)` performs a change of basis using a list of basis operators `basis`.

   - `substitute(substitutions_dict)` substitutes symbols and functions in *R* with given numerical values. This allows easy conversion to NumPy arrays for numerical use.

   - `visualize(rows_start=0, rows_end=None, basis_symbols=None, fontsize=8)` visualizes *R* as a matrix plot. If desired, only certain sections of *R* can be visualized via `rows_start` and `rows_end`. A legend with the basis operator symbols will be drawn if `basis_symbols` is provided. Font size can be adjusted for large matrices.

   - `rate(spin_index_op_index_1, spin_index_op_index_2=None)` returns the relaxation rate between two observables. For the spherical tensor basis, the `spin_index_op_index_X` arguments must be strings of the form `'110'`, where the first number refers to the index of the spin, the second number refers to the rank *l*, and the remaining characters refer to the component *q* of that operator. Negative projections are written with the minus sign, so *q* = -1 of rank *l* = 1 on spin 1 is `'11-1'`. Product operators are simply of the form `'110*210'`, or `'110*21-1'`. Providing `spin_index_op_index_1` only will return the auto-relaxation rate of that operator. If `spin_index_op_index_2` is also provided, the cross-relaxation rate between those two operators is returned (see the examples provided in the repository). For the Cartesian basis, `spin_index_op_index_X` are of the form `'1x'`, `'1z*2z'`, etc.

   - `to_isotropic_rotational_diffusion(fast_motion_limit=False, slow_motion_limit=False)` applies the isotropic rotational diffusion model with the fast-motion or slow-motion limit approximation if desired.

   - `neglect_cross_correlated_terms(mechanism1=None, mechanism2=None)` neglects cross-correlated contributions in *R* between two mechanisms. The arguments `mechanism1` and `mechanism2` must correspond to the names chosen for `mechanism_name`s in `intrs`. If `mechanism2` is not provided, `mechanism1` is used, and if neither is provided, all cross-correlated contributions are neglected.

   - `neglect_cross_relaxation()` neglects all cross-relaxation in *R*, setting every off-diagonal element to zero and leaving only the auto-relaxation rates on the diagonal. Note the distinction from the previous method: cross-*correlation* is between two interaction mechanisms, whereas cross-*relaxation* is between two basis operators.

   - (Only available for the spherical tensor basis): `filter(filter_name, filter_value)` filters out potentially uninteresting regions of *R* based on given criteria. `filter_name` must be one of the following: 'c' for coherence order, 's' for spin order, or 't' for type. This determines the criteria for filtration. `filter_value` is an integer or a list of integers depending on the filtration type (see the documentation in `rela2x.py`) and determines which values are kept (not filtered out) in *R*. For instance, calling `R.filter('c', [0])` would filter out those sections that correspond to basis operators with coherence order other than 0.

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

Five example notebooks that showcase the usage of Rela²x are included in the repository.

## Warnings

Rela²x is not designed for spin systems where the dimension of *R* exceeds ~150, and should be used with caution in such cases. Specifically, displaying the entire matrix `R.op` may cause Jupyter Notebook to crash. Large systems can nevertheless be computed, and the `rate` function can be useful in these scenarios.

## Advanced Users

Additional features not covered in this guide can be found in `rela2x.py`. The code is well-documented, and advanced Python/SymPy users should find it relatively straightforward to navigate.

## License

Rela²x is licensed under the MIT License. See the LICENSE file in the repository for more details.

## Contact Information

If you have questions, comments, or suggestions, please feel free to reach out:

Email: perttu.hilla@oulu.fi

I'm also happy to help with any issues you may encounter while using Rela²x.

## Citations

If you use Rela²x in your work, please include the following citation:

P. Hilla, J. Vaara, Rela²x: Analytic and automatic NMR relaxation theory, *J. Magn. Reson.*, 2025; https://doi.org/10.1016/j.jmr.2024.107828
