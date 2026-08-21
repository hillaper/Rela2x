# ––––– Rela²x –––––
# *A*nalytic and *A*utomatic NMR (relaxation) theory

[![DOI](https://zenodo.org/badge/799492066.svg)](https://doi.org/10.5281/zenodo.21934667)

## Description

Rela²x is a freely available Python package for **analytic** and **automatic** (hence the *a* squared) high-field liquid-state NMR theory. It builds the Liouville-space matrix representations of the superoperators that govern the dynamics of a spin system symbolically, and hands them back as *SymPy* expressions that you can read, manipulate, approximate and publish.

Rela²x was originally built for one thing, and it remains the package's distinctive contribution: Symbolic Redfield relaxation theory derived automatically for arbitrary small spin systems with any spin quantum numbers and any relaxation mechanisms, including every possible cross-correlated term between the interactions that drive relaxation. This incoherent part of the dynamics, the relaxation superoperator ***R***, is what the accompanying publication describes.

Nowadays, Rela²x also includes the coherent part — the Hamiltonian superoperator ***H***, covering the Zeeman interaction (including the chemical shift), the J-coupling, and the residual dipolar and quadrupolar couplings — and combines the two into the Liouvillian ***L*** = −i[*H*, ·] − *R*, the generator of the full equation of motion. Each is available from a single function call:

| | superoperator | describes | entry point |
| --- | --- | --- | --- |
| ***H*** | Hamiltonian | the coherent dynamics | `hamiltonian_superoperator` |
| ***R*** | relaxation | the incoherent dynamics | `relaxation_superoperator` |
| ***L*** | Liouvillian, *L* = −i[*H*, ·] − *R* | the full dynamics | `liouvillian_superoperator` |

Around these come the tools to work with them: spherical tensor and Cartesian product operator bases with sorting and filtering, approximations and simplifications, matrix-element lookup, visualization, and automatic construction of the equations of motion, exportable as LaTeX for publications, etc. 

Everything rests on one physical assumption, the **high-field limit**, and it is applied consistently to the coherent and incoherent parts alike (see [The high-field approximation](#the-high-field-approximation)). The zero- to low-field regime is not yet supported, but is planned for the future.

Rela²x is designed to be user-friendly, requiring only a basic knowledge of Python. Additional experience with the *SymPy* library can be helpful because it is the main library used by Rela²x.

## Notes

Before using Rela²x, it is recommended that you read the related publication https://doi.org/10.1016/j.jmr.2024.107828. There, the Greek letter $\Gamma$ is used for the relaxation superoperator; however, in Python, this is inconvenient, so *R* is used here and in the code. 

**Important:**
This documentation contains the most up-to-date information regarding the code itself, and it differs in many respects from the publication. The underlying relaxation theory is the same, but the code has been updated and improved since the publication.

**Erratum:**
The irreducible spherical tensor components printed at the end of the publication have errors. However, the implementation in Rela²x is and has always been correct.

**API reference:**
The [API reference](#api-reference) below lists everything that `from rela2x import *` exposes. For the full detail of any individual function or class, refer to its docstring in the source modules under `src/rela2x/_core/`. (A thorough dedicated documentation page is planned for the future.)

## Releases

**Rela²x 0.0.1:**  
Initial release of the Rela2x program described in:

P. Hilla, J. Vaara, Rela²x: Analytic and automatic NMR relaxation theory, *J. Magn. Reson.*, 2025; [https://doi.org/10.1016/j.jmr.2024.107828](https://doi.org/10.1016/j.jmr.2024.107828)

**Rela²x 0.0.2:**  
New features, bug fixes, type hints, naming consistency, documentation, cosmetic improvements, and Cartesian product operator basis support.

The most significant fixes concern the sorting of the product operator basis. The order in which the basis operators appear therefore differs from version 0.0.1, and from the figures in the publication; the matrix elements themselves are unaffected.

**Rela²x 0.0.3:**  
Coherent interactions. The Zeeman interaction (including the chemical shift) and the J-coupling interaction can now be computed as the Hamiltonian superoperator *H* (`hamiltonian_superoperator`), and combined with the relaxation superoperator into the Liouvillian *L* = −i[*H*, ·] − *R* (`liouvillian_superoperator`). The package was also reorganised from the single `rela2x.py` file into logical modules under `src/rela2x/_core/`. Several minor updates were also added along the way.

The high-field secular approximation is now applied to the coherent interactions as well as to *R*, and against the same Larmor frequency symbols, so the two parts of the Liouvillian rest on the same assumption. In practice this means a J-coupling is treated as weak between spins with different isotope labels and strong between spins sharing one. `keep_non_secular` switches the approximation off in both parts at once. See [The high-field approximation](#the-high-field-approximation).

**Rela²x 0.0.4:**  
Residual dipolar and quadrupolar couplings, added as the coherent mechanisms `'RDC'` and `'RQC'`. These are the parts of the dipole-dipole and quadrupolar interactions that survive incomplete motional averaging, and they are the coherent counterparts of the `'DD'` and `'Q'` relaxation mechanisms: the average of an anisotropic interaction belongs in *H*, the fluctuation about it in *R*. 

The residual dipolar coupling shares the secular treatment of the J-coupling, on the same Larmor frequency symbols, and its constants $D_{ij}$ are defined as the splittings they produce for a heteronuclear pair — so such a pair carrying both couplings splits by $J_{ij} + D_{ij}$. The residual quadrupolar coupling constants $\omega_Q^{(i)}$ are angular frequencies, defined as the spacing between adjacent single-quantum transitions of the spin.

See [The high-field approximation](#the-high-field-approximation) for the assumption these rest on: **the secular approximation is still made against the Zeeman Hamiltonian alone**, so the residual couplings are assumed small compared with the Larmor frequency differences it tests.

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

These dependencies are listed in `pyproject.toml`. Rela²x is designed to be an interactive program, so a Jupyter Notebook installation is strongly recommended.

## Usage

The usage of Rela²x is summarized below. Specifics, such as variable names, can be customized as needed. (See also the example notebooks included in the repository.)

**Import Rela²x:**

   ```python
   from rela2x import *
   ```
   
   Although wildcard imports (`*`) are generally not recommended, Rela²x is a relatively small library, so this is not an issue. It is quite convenient to have all the functions in Jupyter Notebook's memory space for automatic recommendations and, for example, function docstrings while coding.
   
**Define the spin system:**

   Spin systems are defined via a list of isotope names. For instance:
   
   ```python
   spin_system = ['14N', '1H', '1H']
   ```
   
   A collection of NMR isotopes and their spin quantum numbers is held in the `ISOTOPES` dictionary, which is brought into your namespace by the import above. Each entry maps an isotope label to a `[spin quantum number, gyromagnetic ratio in MHz/T]` pair. If your preferred nucleus is not listed, feel free to add it. 
   
   Adding a suffixed copy of an existing isotope is also how chemically inequivalent nuclei of the same isotope are given distinct Zeeman frequencies (i.e., chemical shifts).

   ```python
   ISOTOPES['1H_X'] = ISOTOPES['1H']
   ```

   The distinct label is what produces a distinct Zeeman frequency symbol $\omega_{\mathrm{1H}}$, and that symbol is shared by *H* and *R* alike (see `w_symbol`). This is how the chemical shift enters: there is no separate shielding or chemical-shift symbol.
   
**Choose general settings (optional):**

   Rela²x currently supports two general settings, held in `_core/_settings.py`.
   
   - `RELAXATION_THEORY` handles the level of theory used: semiclassical `'sc'`, or quantum mechanical (Lindbladian) `'qm'`.
   - `VERBOSE` controls whether the entry points print their progress (section headers, timing, and per-mechanism messages) while building *R*, *H* or *L*.
   
   The default values are `'sc'` and `True`. Note that neither `RELAXATION_THEORY` nor `VERBOSE` is brought into your namespace by `from rela2x import *`, so assigning to a bare name in your own script has no effect. Set them through `set_relaxation_theory` and `set_verbose` instead. For instance:
   
   ```python
   set_relaxation_theory('qm')
   set_verbose(False)
   ```
   
   selects the Lindbladian description of *R* and silences the progress output.
   
**Define the coherent interactions:**

   Coherent interactions are defined via a Python dictionary, in a 0/1-flag style. For the example ¹⁴N–¹H–¹H spin system, we could have

   ```python
   coh_intrs = {
       'Z':   [1, 1, 1],
       'J':   [[0, 1, 1],
               [0, 0, 1],
               [0, 0, 0]],
       'RDC': [[0, 1, 1],
               [0, 0, 1],
               [0, 0, 0]],
       'RQC': [1, 0, 0]
   }
   ```

   - `'Z'` is the Zeeman interaction, including the chemical shift. Its value is a list of `1`s and `0`s defining which spins carry the interaction. The Hamiltonian is $\hat H_Z = \sum_i \omega_i \hat S_z^{(i)}$, in the laboratory frame and in angular frequency units.

   - `'J'` is the J-coupling interaction. Its value is a coupling matrix whose `1`s define which spins are coupled; **only the upper triangle is read.** The coupling constants $J_{ij}$ are ordinary frequencies (in Hz), converted internally by the factor of $2\pi$.

     The high-field secular approximation is applied here (see [The high-field approximation](#the-high-field-approximation) for the same treatment applied to *R*, below). The longitudinal term $\hat S_z^{(i)}\hat S_z^{(j)}$ commutes with the Zeeman Hamiltonian and is always retained, whereas the flip-flop term $\hat S_x^{(i)}\hat S_x^{(j)} + \hat S_y^{(i)}\hat S_y^{(j)}$ oscillates at the difference of the two Larmor frequencies and is retained only when that difference (approximately) vanishes. So

     $$\hat H_J = \sum_{i<j} 2\pi J_{ij}\, \hat S_z^{(i)}\hat S_z^{(j)} \quad\text{(heteronuclear, weak coupling)},$$
     $$\hat H_J = \sum_{i<j} 2\pi J_{ij}\, \hat{\mathbf S}^{(i)} \cdot \hat{\mathbf S}^{(j)} \quad\text{(homonuclear, strong coupling)}.$$

     The test is made on the same Larmor frequency symbols that the secular approximation of *R* uses below, so spins sharing an isotope label are degenerate and spins carrying distinct labels are not. `keep_non_secular=True` restores the full dot product for every pair.

   - `'RDC'` is the residual dipolar coupling — the part of the dipole-dipole interaction that survives incomplete motional averaging, as in partially oriented or otherwise anisotropic environments. Its value is a coupling matrix, exactly as for `'J'`, and its constants $D_{ij}$ are likewise ordinary frequencies (in Hz).

     The interaction is the rank-2, projection-0 spherical tensor of the two spin vectors, $\hat T_{20} \propto 3\hat S_z^{(i)}\hat S_z^{(j)} - \hat{\mathbf S}^{(i)} \cdot \hat{\mathbf S}^{(j)}$, so its secular approximation works exactly as the J-coupling one does, on the same Larmor frequency symbols:

     $$\hat H_{\mathrm{RDC}} = \sum_{i<j} 2\pi D_{ij}\, \hat S_z^{(i)}\hat S_z^{(j)} \quad\text{(heteronuclear)},$$
     $$\hat H_{\mathrm{RDC}} = \sum_{i<j} \pi D_{ij} \left( 3\hat S_z^{(i)}\hat S_z^{(j)} - \hat{\mathbf S}^{(i)} \cdot \hat{\mathbf S}^{(j)} \right) \quad\text{(homonuclear)}.$$

     Written this way, the residual dipolar coupling is the J-coupling with the weight of the flip-flop term changed from $1$ to $-1/2$, and $D_{ij}$ is by construction the splitting it produces in the spectrum of a heteronuclear pair. A heteronuclear pair carrying both couplings therefore splits by $J_{ij} + D_{ij}$, the familiar result, whereas a homonuclear pair keeps the two distinct.

     Explicitly, $D_{ij}$ is the ensemble-averaged dipolar coupling

     $$D_{ij} = -\frac{\mu_0 \gamma_i \gamma_j \hbar}{4\pi^2 r_{ij}^3} \left\langle \frac{3\cos^2\theta_{ij} - 1}{2} \right\rangle = -\frac{\mu_0 \gamma_i \gamma_j h}{8\pi^3 r_{ij}^3} \left\langle \frac{3\cos^2\theta_{ij} - 1}{2} \right\rangle,$$

     where $r_{ij}$ is the internuclear distance, $\theta_{ij}$ the angle between the internuclear vector and the magnetic field, and $\langle\cdot\rangle$ the ensemble average. This is the standard convention of the residual dipolar coupling: $D_{ij}$ is exactly the quantity obtained by subtracting the isotropic splitting $J_{ij}$ from the splitting measured in an aligned sample. It vanishes under isotropic motion, which is why the dipole-dipole interaction contributes only to *R* in ordinary liquid-state NMR.

   - `'RQC'` is the residual quadrupolar coupling, the single-spin counterpart of the above for spin $I>1/2$ nuclei. Its value is a list of `1`s and `0`s, exactly as for `'Z'`. The interaction is the rank-2, projection-0 spherical tensor of a single spin, $\hat T_{20} \propto 3(\hat S_z)^2 - S(S+1)$:

     $$\hat H_{\mathrm{RQC}} = \sum_i \frac{\omega_Q^{(i)}}{6} \left( 3\left(\hat S_z^{(i)}\right)^2 - S_i(S_i+1) \right).$$

     Only the projection-0 component survives the secular approximation, so this Hamiltonian is secular as it stands and `keep_non_secular` does not affect it. Unlike $J_{ij}$ and $D_{ij}$, the constants $\omega_Q^{(i)}$ are **angular** frequencies, following the usual convention in which the constant is the spacing it produces between adjacent single-quantum transitions of the spin; the central transition of a half-integer spin is left unshifted, as it must be to first order. A spin-1/2 nucleus carries no quadrupole moment, so including one raises a `ValueError` rather than contributing nothing.

     Explicitly, $\omega_Q^{(i)}$ is the motionally averaged quadrupolar coupling

     $$\omega_Q^{(i)} = \frac{3\, e^2 q_i Q_i}{2 S_i (2 S_i - 1)\, \hbar} \left\langle \frac{3\cos^2\theta_i - 1}{2} + \frac{\eta_i}{2}\sin^2\theta_i \cos 2\phi_i \right\rangle,$$

     where $C_Q^{(i)} = e^2 q_i Q_i / h$ is the quadrupolar coupling constant, $\eta_i$ the asymmetry of the electric field gradient tensor, and $\theta_i$, $\phi_i$ the polar angles of the magnetic field in the principal axis frame of that tensor. For an axially symmetric electric field gradient on a spin-1 nucleus this reduces to the familiar deuterium result, a splitting of $\tfrac{3}{2} C_Q \langle P_2(\cos\theta) \rangle$. Like the residual dipolar coupling, it vanishes under isotropic motion.

     Note that the residual quadrupolar coupling constants are keyed by the spin *index* rather than by the isotope label, unlike the Larmor frequencies, because they are properties of the local environment of an individual nucleus.

   The two definitions above are given so that it is unambiguous what number to substitute for each symbol; Rela²x itself never evaluates them, and treats $D_{ij}$ and $\omega_Q^{(i)}$ as the free symbolic parameters you manipulate, exactly as it does $J_{ij}$.
   
   A missing key means the mechanism is absent, so `{}` gives a vanishing Hamiltonian and `{'J': ...}` alone gives J-coupling without Zeeman terms.

**Compute the matrix representation of *H*, convert it to the product operator basis, and create a `HamiltonianSuperoperator` object:**

   ```python
   H = hamiltonian_superoperator(spin_system, coh_intrs, basis='T', sorting='v1', keep_non_secular=False)
   ```

   The `hamiltonian_superoperator` function takes as input the `spin_system` and `coh_intrs` variables as defined above, information about which product operator basis to use, and optionally about how to sort the basis via `sorting`. It is useful to represent *H* in a basis where it achieves a block-diagonal form. A good basis for this purpose is the direct product basis of spherical tensor operators, provided via `basis='T'`. For a system of spin-1/2 nuclei, the Cartesian product operator basis can also be used by choosing `basis='C'`.

   Three options are available for `sorting` (currently only supported for the spherical tensor basis): `'v1'`, `'v2'`, or `None` (for details, see the documentation in `_core/_basis.py`). `keep_non_secular` allows keeping non-secular terms, exactly as above.

   The function returns a `HamiltonianSuperoperator` object.

   *Attributes.* These come from the shared `Superoperator` base class, so they are present on *R* and *L* in exactly the same way:

   - `op` — the matrix representation of *H*.
   - `symbols_in` — all symbols appearing in *H*.
   - `functions_in` — all functions appearing in *H*.
   - `basis_symbols` — the basis operator symbols of the chosen product operator basis.
   - `basis_norms` — the Liouville norms of the (unnormalized) basis operators.
   - `basis_indices` — the basis operator indices. Each entry is a tuple describing one basis operator, holding one item per spin that carries something other than the identity operator. For the spherical tensor basis the items are `(spin index, l, q)` triples, and for the Cartesian basis they are `(spin index, direction)` pairs. Spin indices start from 1, matching the operator symbols. The identity operator of the whole system is the empty tuple. For instance, in a two-spin system, $\hat T_{10}^{(1)} \hat T_{1-1}^{(2)}$ has the index `((1, 1, 0), (2, 1, -1))`.
   - `generator` — the matrix that actually drives the equations of motion. Each superoperator carries its own sign convention here: for *H* it is $-i[\hat H, \cdot]$, for *R* it is $-R$, and for *L* it is *L* itself. This is what `equations_of_motion` reads.

   *Shared methods.* Also provided by `Superoperator`, and likewise available on *R* and *L*:

   - `to_basis(basis)` performs a change of basis using a list of basis operators `basis`.

   - `to_observables()` fixes the basis operator normalization, so that the matrix elements correspond to observables. The entry points call this for you.

   - `substitute(substitutions_dict)` substitutes symbols and functions with given numerical values. This allows easy conversion to NumPy arrays for numerical use.

   - `visualize(rows_start=0, rows_end=None, basis_symbols=None, fontsize=8)` visualizes the matrix as a nonzero-pattern plot. If desired, only certain sections can be visualized via `rows_start` and `rows_end`. A legend with the basis operator symbols will be drawn if `basis_symbols` is provided. Font size can be adjusted for large matrices.

   - `element(spin_index_op_index_1, spin_index_op_index_2=None)` returns the matrix element between two basis operators. For the spherical tensor basis, the `spin_index_op_index_X` arguments must be strings of the form `'110'`, where the first number refers to the index of the spin, the second number refers to the rank *l*, and the remaining characters refer to the component *q* of that operator. Negative projections are written with the minus sign, so *q* = -1 of rank *l* = 1 on spin 1 is `'11-1'`. Product operators are simply of the form `'110*210'`, or `'110*21-1'`. Providing `spin_index_op_index_1` only returns the diagonal element of that operator. For the Cartesian basis, `spin_index_op_index_X` are of the form `'1x'`, `'1z*2z'`, etc.

   - (Only available for the spherical tensor basis): `filter(filter_name, filter_value)` filters out potentially uninteresting regions based on given criteria. `filter_name` must be one of the following: 'c' for coherence order, 's' for spin order, or 't' for type. This determines the criteria for filtration. `filter_value` is an integer or a list of integers depending on the filtration type (see the documentation in `_core/_basis.py`) and determines which values are kept (not filtered out). For instance, calling `H.filter('c', [0])` would filter out those sections that correspond to basis operators with coherence order other than 0.

   *Coherent-specific methods.* These belong to `HamiltonianSuperoperator`, and on a Liouvillian they are reached through `L.H`:

   - `frequency(spin_index_op_index_1, spin_index_op_index_2=None)` returns the coherent frequency between two basis operators — the coherent counterpart of `rate` below, taking exactly the same arguments.

   The best way to get acquainted is to try these functions yourself!

**Define the incoherent interactions that drive relaxation:**

   Incoherent interactions are defined via a Python dictionary with key-value pairs of the following type:
   
   `'mechanism_name': ('type', intr_array, rank_list)`
   
   - `mechanism_name` appears in the spectral-density function symbols and is mostly a cosmetic label that does not affect the actual calculation. However, these names are utilized if cross-correlated couplings are neglected (see below).
   
   - For single-spin linear or single-spin quadratic interactions, `type` is either `'1L'` or `'1Q'`, respectively. For two-spin bilinear interactions, `type` is always `'2'`. Bilinearity of two-spin interactions does not need to be specified.
   
   - The `intr_array` for single-spin mechanisms is a Python list of values `1` or `0`, defining which spins in `spin_system` are included in that interaction. For two-spin mechanisms, a coupling matrix (list of lists) is provided where the `1`s define which spins are coupled. Only the upper triangle needs to be provided.
   
   - `rank_list` is a list of ranks *l* of the given mechanism.
   
   For instance, for our example `spin_system = ['14N', '1H', '1H']` with chemical-shift anisotropy (including all ranks) and quadrupolar interactions on ¹⁴N, and dipole-dipole couplings between all of the spins, we would have:
   
   ```python
   incoh_intrs = {
       'CSA': ('1L', [1, 0, 0], [0, 1, 2]),
       'Q':   ('1Q', [1, 0, 0], [2]),
       'DD':  ('2', [[0, 1, 1],
                     [0, 0, 1], 
                     [0, 0, 0]], 
                     [2])
   }
   ```

   Note the difference from `coh_intrs` above: there, the key *is* the mechanism, and only `'Z'` and `'J'` are accepted; here, the key is a free cosmetic label and the type lives in the value.

**Compute the matrix representation of *R*, convert it to the product operator basis, and create a `RelaxationSuperoperator` object:**

   ```python
   R = relaxation_superoperator(spin_system, incoh_intrs, basis='T', sorting='v1', keep_non_secular=False)
   ```

   Takes the same `basis`, `sorting` and `keep_non_secular` arguments as `hamiltonian_superoperator` above, with the same meaning.

   Note that the non-unit norms of observables are taken into account in the relaxation rates, i.e., the matrix elements. The rates directly correspond to observables.

   The function returns a `RelaxationSuperoperator` object. It shares the attributes and shared methods described for *H* above (`op`, `symbols_in`, `functions_in`, `basis_symbols`, `basis_norms`, `basis_indices`, `generator`, `to_basis`, `to_observables`, `substitute`, `visualize`, `element`, `filter`), plus its own:

   *Relaxation-specific methods.* These belong to `RelaxationSuperoperator`, and on a Liouvillian they are reached either directly (see below) or through `L.R`:

   - `rate(spin_index_op_index_1, spin_index_op_index_2=None)` returns the relaxation rate between two observables — the relaxation-flavoured name for `element`, taking exactly the same arguments. Providing one operator returns its auto-relaxation rate; providing two returns the cross-relaxation rate between them (see the examples provided in the repository).

   - `to_isotropic_rotational_diffusion(fast_motion_limit=False, slow_motion_limit=False)` applies the isotropic rotational diffusion model with the fast-motion or slow-motion limit approximation if desired.

   - `neglect_cross_correlated_terms(mechanism1=None, mechanism2=None)` neglects cross-correlated contributions in *R* between two mechanisms. The arguments `mechanism1` and `mechanism2` must correspond to the names chosen for `mechanism_name`s in `incoh_intrs`. If `mechanism2` is not provided, `mechanism1` is used, and if neither is provided, all cross-correlated contributions are neglected.

   - `neglect_cross_relaxation()` neglects all cross-relaxation in *R*, setting every off-diagonal element to zero and leaving only the auto-relaxation rates on the diagonal. Note the distinction from the previous method: cross-*correlation* is between two interaction mechanisms, whereas cross-*relaxation* is between two basis operators.

**Combine *H* and *R* into the Liouvillian *L*, the generator of the full equation of motion** (note that *H* and *R* need not be computed first; `liouvillian_superoperator` can build both parts from the dictionaries directly):

   ```python
   L = liouvillian_superoperator(spin_system, coh_intrs, incoh_intrs, basis='T', sorting='v1', keep_non_secular=False)
   ```

   `liouvillian_superoperator` takes the coherent dictionary first, then the incoherent one, builds both parts against a **single shared basis**, and returns a `LiouvillianSuperoperator` object with

   - `op` returning the combination $L = -i[\hat H, \cdot] - R$,
   - `H` and `R` returning the two parts, which remain fully usable — `L.H.frequency(...)`, `L.R.rate(...)`, and `L.element(...)` for the combined matrix element, which carries both a coherent frequency and a relaxation rate. Shown below in turn; use `display` rather than `print`, so that the *SymPy* expressions are typeset rather than stringified:

   ```python
   display(L.H.frequency('110'))
   display(L.R.rate('110'))
   display(L.element('110'))
   ```

   The relaxation-only approximations (`to_isotropic_rotational_diffusion`, `neglect_cross_correlated_terms`, `neglect_cross_relaxation`) are available directly on the `LiouvillianSuperoperator` and act on its relaxation part, leaving the coherent terms untouched; `to_basis`, `to_observables`, `substitute` and `filter` act on both parts. In every case the combination is rebuilt from the parts, so `L.op` and `L.H`/`L.R` can never disagree.

   Combining the parts after the normalization to observables is exact, because that rescaling is a diagonal similarity transform and therefore distributes over the sum.

   Passing an empty `incoh_intrs` gives a purely coherent, undamped Liouvillian, and passing an empty `coh_intrs` recovers the relaxation-only dynamics — so *H* and *R* alone are really just the two limits of *L*.

**After *R*, *H* or *L* is computed, construct the resulting equations of motion for the observables:**

   ```python
   eoms = equations_of_motion(L, expectation_values=True, included_operators=None)
   ```

   Here the superoperator object itself is passed, not its matrix representation — this works identically for *R*, *H* or *L*, since each object supplies its own sign convention through its `generator` property, so nothing else has to change. The remaining arguments are for cosmetic purposes (try it yourself). The returned `eoms` is a *SymPy* equation object. The outcome depends on `RELAXATION_THEORY`, because the semiclassical and Lindbladian master equations are different.

**Save the equations of motion in LaTeX format to the current working directory as a .txt file for further use in, for example, publications:**

   ```python
   equations_of_motion_to_latex(eoms, savename)
   ```

   `savename` is an arbitrary string.

## The high-field approximation

The derivation of *R* (see the publication) rests on the high-field approximation, so that the Zeeman Hamiltonian is assumed as the dominant coherent interaction.

**The secular approximation this yields is made against the Zeeman Hamiltonian alone, not the complete static Hamiltonian.** A term survives it if the frequency at which it oscillates in the interaction frame of $\hat H_Z$ vanishes, and this single test governs the coherent and incoherent parts alike. Every other coherent interaction Rela²x builds is consequently a spectator to it: for the J-coupling this means the flip-flop term is kept between spins sharing an isotope label and dropped otherwise, and *R*'s own secular test never sees the J-coupling at all; the residual dipolar and quadrupolar couplings, though themselves part of the static Hamiltonian, likewise never feed back into deciding what survives. Strictly, the test should be made in the eigenbasis of the complete static Hamiltonian rather than of the Zeeman term alone; the present treatment is the correct leading-order one as long as these other couplings stay small compared with the Larmor frequency differences the test is made on — the usual high-field situation, where e.g. $J \ll \omega$ — but it is an assumption rather than an identity, and Rela²x does not currently describe a regime where that ordering breaks down. (Perhaps an update for the future.)

## API reference

For reference: the Usage walkthrough above tells the whole story of building *H*, *R* and *L*, and covers the handful of names needed for ordinary work. Everything below is the rest of the public namespace — building blocks Rela²x uses internally and exposes because they are useful on their own. `from rela2x import *` brings 73 names into your namespace in total; only names related to spin physics or to the package's main functionality are exposed this way. The internal plumbing behind them (string/index parsing, basis-sorting machinery, the individual relaxation/Hamiltonian term-builders, and similar helpers) remains fully usable, just not part of the flat namespace — reach it via its submodule under `rela2x._core` if you need it. Grouped below by layer, roughly from the top down; full documentation for every one of them lives in the docstrings, under `src/rela2x/_core/`.

### Entry points

| | |
| --- | --- |
| `hamiltonian_superoperator(spin_system, coherent_interactions, basis='T', sorting='v1', keep_non_secular=False)` | Compute *H* in a product operator basis |
| `relaxation_superoperator(spin_system, incoherent_interactions, basis='T', sorting='v1', keep_non_secular=False)` | Compute *R* in a product operator basis |
| `liouvillian_superoperator(spin_system, coherent_interactions, incoherent_interactions, basis='T', sorting='v1', keep_non_secular=False)` | Compute both and combine them into *L* |
| `set_relaxation_theory(theory)` | Select `'sc'` (semiclassical) or `'qm'` (Lindbladian) |
| `set_verbose(verbose)` | Toggle the progress output of the entry points |

### Superoperator classes

| | |
| --- | --- |
| `Operator` | Base class: a matrix representation plus its symbols and functions |
| `Superoperator` | Adds the basis attributes, `to_observables`, `element`, `filter` and `generator` |
| `HamiltonianSuperoperator` | *H*: adds `frequency` |
| `RelaxationSuperoperator` | *R*: adds `rate`, the relaxation counterpart of `frequency`, and the relaxation approximations |
| `LiouvillianSuperoperator` | *L*: holds `H` and `R`, and keeps `op` in step with them |

### Spin systems and isotopes

| | |
| --- | --- |
| `ISOTOPES` | Isotope table, mapping a label to `[spin quantum number, gyromagnetic ratio in MHz/T]` |
| `spin_quantum_numbers(isotopes)` | Look up the spin quantum numbers of a list of isotopes |
| `SpinOperators(spin_system)` | The spin operators of a spin system; the object every builder below takes |

`SpinOperators` exposes `spin_system`, `S`, `N_spins`, `N_states`, the many-spin Cartesian operators `E`, `Sx`, `Sy`, `Sz`, `Sp`, `Sm`, the spherical tensor operators `T` (a `{(l, q): matrix}` dict per spin), and a `*_symbol` counterpart for each.

### Hilbert-space operators

| | |
| --- | --- |
| `op_Sx(S)`, `op_Sy(S)`, `op_Sz(S)`, `op_Sp(S)`, `op_Sm(S)` | Single-spin angular momentum operators |
| `op_Svec(S)` | The Cartesian vector operator, as a list |
| `op_T(S, l, q)` | Single-spin spherical tensor operator |

### Liouville space

| | |
| --- | --- |
| `vectorize(op)`, `vectorize_all(ops)` | Vectorize operators into Liouville-space supervectors |
| `sop_left_mul(op)`, `sop_right_mul(op)` | Left- and right-multiplication superoperators |
| `sop_commutator(op)` | The commutation superoperator [op, ·] |
| `sop_double_commutator(op1, op2)` | The double-commutation superoperator |
| `sop_D(op1, op2)` | The Lindbladian dissipation superoperator |

### Product operator bases

| | |
| --- | --- |
| `T_product_basis_and_symbols(spin_operators, sorting='v1')` | Spherical tensor basis, symbols, norms and indices — sorted |
| `Cartesian_product_basis_and_symbols(spin_operators)` | The same for the Cartesian basis |

Basis operator indices, the tuples returned above and described under [Usage](#usage), can be read (spin order, coherence order, ...), converted from strings such as `'110'` or `'1z*2z'`, and filtered — see `Superoperator.filter` and `.element` for the user-facing versions of this machinery.

### Symbols

| | |
| --- | --- |
| `w_symbol(spin_name)` | Larmor frequency $\omega$ of a spin — shared by *H* and *R* |
| `J_coupling_symbol(spin_index_1, spin_index_2)` | J-coupling constant, in Hz |
| `D_coupling_symbol(spin_index_1, spin_index_2)` | Residual dipolar coupling constant, in Hz |
| `w_Q_symbol(spin_index)` | Residual quadrupolar coupling constant $\omega_Q$, as an angular frequency |
| `op_S_symbol(direction, index)`, `op_T_symbol(l, q, index)` | Printed operator symbols |
| `product_op_S_symbol(...)`, `product_op_T_symbol(...)` | Products thereof |
| `expectation_value(op_symbol)`, `f_expectation_value_t(op_symbol)` | Observable and time-dependent observable symbols |

### Spectral densities

| | |
| --- | --- |
| `J_w(intr1, intr2, l, argument)` | The abstract spectral density function *J*(ω) of an interaction pair |
| `J_w_isotropic_rotational_diffusion(...)` | Its isotropic rotational diffusion form |
| `Lorentzian(w, tau_c, ...)` | A Lorentzian, with optional motional limits |
| `Schofield_theta(w)` | Thermal correction factor of the quantum mechanical theory |

### Superoperator construction

The engines behind the entry points, for when you want the raw matrix rather than the object.

| | |
| --- | --- |
| `op_H_Z(spin_operators, coupling_strengths)` | The Zeeman Hamiltonian, including the chemical shift |
| `op_H_J(spin_operators, coupling_strengths_matrix, keep_non_secular=False)` | The J-coupling Hamiltonian |
| `op_H_RDC(spin_operators, coupling_strengths_matrix, keep_non_secular=False)` | The residual dipolar coupling Hamiltonian |
| `op_H_RQC(spin_operators, coupling_strengths)` | The residual quadrupolar coupling Hamiltonian |
| `op_H(spin_operators, coherent_interactions, keep_non_secular=False)` | The total coherent Hamiltonian |
| `sop_H(spin_operators, coherent_interactions, keep_non_secular=False)` | Its commutation superoperator |
| `sop_R(spin_operators, incoherent_interactions, keep_non_secular=False)` | The relaxation superoperator matrix |

### Equations of motion

| | |
| --- | --- |
| `equations_of_motion(superoperator, expectation_values=True, included_operators=None)` | Build the differential equations for the observables |
| `equations_of_motion_to_latex(eqs, savename)` | Write them to a LaTeX file |

### Linear algebra

| | |
| --- | --- |
| `Kronecker_product(*m)` | Symbolic Kronecker product |
| `commutator(op1, op2)` | Symbolic commutator |
| `Liouville_bracket(op1, op2)`, `Liouville_norm(op)`, `Liouville_amplitude(op1, op2)` | Hilbert–Schmidt inner product, norm and amplitude |
| `op_change_of_basis(op, basis)` | Change of basis |
| `op_decomposition(op, basis, basis_symbols)` | Decompose an operator in a basis set |

### Visualization

| | |
| --- | --- |
| `visualize_operator(op, ...)` | Nonzero-pattern plot of one matrix |
| `visualize_many_operators(ops, ...)` | The combined pattern of several, for comparison |
| `matrix_nonzeros(matrix)` | The underlying 0/1 mask |

### Symbolic constants

`hbar`, `k_B`, `mu_0`, `y_0` (an arbitrary $\gamma$), `w_0` (an arbitrary $\omega$), `B` (magnetic field), `temperature`, `beta` ($\hbar/k_\mathrm{B}T$), `t` (time), `tau`, `tau_c` (correlation time). All are *SymPy* symbols, ready to substitute into expressions.

## Examples

Eight example notebooks that showcase the usage of Rela²x are included in the repository. `rela2x_example5_coherent_interactions.ipynb` and `rela2x_example6_liouvillian.ipynb` cover the coherent interactions and the Liouvillian introduced in 0.0.3: the Hamiltonian superoperator on its own, and combined with the relaxation superoperator into the full dynamics. `rela2x_example7_residual_couplings.ipynb` covers the residual dipolar and quadrupolar couplings introduced in 0.0.4, alongside the relaxation they are the coherent counterparts of.

## Warnings

Rela²x is not designed for spin systems where the dimension of the superoperators exceeds ~150, and should be used with caution in such cases. Specifically, displaying an entire matrix such as `H.op`, `R.op` or `L.op` may cause Jupyter Notebook to crash. Large systems can nevertheless be computed, and the `element` method (`frequency` for *H*, `rate` for *R*) can be useful in these scenarios.

## Advanced Users

The [API reference](#api-reference) lists the whole public namespace, including the lower-level building blocks that the Usage walkthrough does not touch. The code is well-documented, and advanced Python/SymPy users should find the source modules under `src/rela2x/_core/` relatively straightforward to navigate.

## License

Rela²x is licensed under the MIT License. See the LICENSE file in the repository for more details.

## Contact Information

If you have questions, comments, or suggestions, please feel free to reach out:

Email: perttu.hilla@oulu.fi

I'm also happy to help with any issues you may encounter while using Rela²x.

## Citations

If you use Rela²x in your work, please include the following citations:

P. Hilla, J. Vaara, Rela²x: Analytic and automatic NMR relaxation theory, *J. Magn. Reson.*, 2025; https://doi.org/10.1016/j.jmr.2024.107828

P. Hilla, J. Eronen, Rela²x [software], Zenodo, 2026; https://doi.org/10.5281/zenodo.21934667
