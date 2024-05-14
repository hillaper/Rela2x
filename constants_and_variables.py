"""
Variables and constants that can be readily used in the Rela2x package.

Authors: 
    Perttu Hilla.
    perttu.hilla@oulu.fi
"""

import sympy as smp

# Constants.
hbar = smp.Symbol('hbar', real=True, positive=True) # Reduced Planck constant
k_B = smp.Symbol('k_B', real=True, positive=True) # Boltzmann constant
mu_0 = smp.Symbol('\\mu_0', real=True, positive=True) # Vacuum permeability

w_0 = smp.Symbol('\\omega_0', real=True, positive=True) # Arbitrary Larmor frequency

y_el = smp.Symbol('\\gamma_el', real=True) # Gyromagnetic ratio of the electron
y_1H = smp.Symbol('\\gamma_1H', real=True) # Gyromagnetic ratio of 1H
y_13C = smp.Symbol('\\gamma_13C', real=True) # Gyromagnetic ratio of 13C
y_14N = smp.Symbol('\\gamma_14N', real=True) # Gyromagnetic ratio of 14N

# Symbolic variables.
B = smp.Symbol('B', real=True, positive=True) # Magnetic field
T = smp.Symbol('T', real=True, positive=True) # Temperature
beta = hbar / (k_B * T) # Inverse temperature multiplied by hbar

w_el = smp.Symbol('\\omega_el', real=True) # Electron Larmor frequency
w_1H = smp.Symbol('\\omega_1H', real=True) # 1H Larmor frequency
w_13C = smp.Symbol('\\omega_13C', real=True) # 13C Larmor frequency
w_14N = smp.Symbol('\\omega_14N', real=True) # 14N Larmor frequency

t = smp.Symbol('t', real=True, positive=True) # Time
tau = smp.Symbol('\\tau', real=True, positive=True) # Time constant
tau_c = smp.Symbol('\\tau_c', real=True, positive=True) # Correlation time
