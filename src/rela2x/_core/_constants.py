"""
Symbolic constants and variables that can be readily used in Rela2x.

All quantities are defined as SymPy symbols so that they can be used
directly in the analytic expressions built up throughout the package.
"""

import sympy as smp

# Reduced Planck constant.
hbar = smp.Symbol('hbar', real=True, positive=True)

# Boltzmann constant.
k_B = smp.Symbol('k_B', real=True, positive=True)

# Vacuum permeability.
mu_0 = smp.Symbol('\\mu_0', real=True, positive=True)

# Arbitrary gyromagnetic ratio.
y_0 = smp.Symbol('\\gamma_0', real=True)

# Arbitrary Larmor frequency.
w_0 = smp.Symbol('\\omega_0', real=True)

# Magnetic field amplitude.
B = smp.Symbol('B', real=True, positive=True)

# Temperature.
# NOTE: Named in full so that the bare name T stays free for the spherical
# tensor operators, which the rest of the package uses it for.
temperature = smp.Symbol('T', real=True, positive=True)

# Inverse temperature multiplied by hbar.
beta = hbar / (k_B * temperature)

# Time.
t = smp.Symbol('t', real=True, positive=True)

# Time constant.
tau = smp.Symbol('\\tau', real=True, positive=True)

# Correlation time.
tau_c = smp.Symbol('\\tau_c', real=True, positive=True)
