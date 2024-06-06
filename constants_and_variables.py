"""
Variables and constants that can be readily used in the Rela2x package.
"""

import sympy as smp

# Symbolic constants.
hbar = smp.Symbol('hbar', real=True, positive=True) # Reduced Planck constant
k_B = smp.Symbol('k_B', real=True, positive=True) # Boltzmann constant
mu_0 = smp.Symbol('\\mu_0', real=True, positive=True) # Vacuum permeability

y_0 = smp.Symbol('\\gamma_0', real=True) # Arbitrary gyromagnetic ratio
w_0 = smp.Symbol('\\omega_0', real=True) # Arbitrary Larmor frequency

# Symbolic variables.
B = smp.Symbol('B', real=True, positive=True) # Magnetic field amplitude
T = smp.Symbol('T', real=True, positive=True) # Temperature
beta = hbar / (k_B * T) # Inverse temperature multiplied by hbar

t = smp.Symbol('t', real=True, positive=True) # Time
tau = smp.Symbol('\\tau', real=True, positive=True) # Time constant
tau_c = smp.Symbol('\\tau_c', real=True, positive=True) # Correlation time
