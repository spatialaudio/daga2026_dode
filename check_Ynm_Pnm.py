# see https://github.com/spatialaudio/daga2026_dode

# check that scipy.special assoc_legendre_p() and sph_harm_y()
# are consistent with the conventions from
# [Raf19] Boaz Rafaely (2019): Fundamentals of Spherical Array Processing,
# Springer, 2nd Ed., https://doi.org/10.1007/978-3-319-99561-8
# we check exemparily for some orders from table 1.1 / 1.3
# we could/should have used Python's errors and exceptions handling
# but we lazily go for simple prints of True/False

import numpy as np
import scipy as sp
from scipy.special import (sph_harm_y,
                           assoc_legendre_p,
                           )
print(np.__version__)  # tested with 2.4.2
print(sp.__version__)  # tested with 1.17.0

print('check Spherical Harmonics')
# use random angles, checking 0th, 1st and 4th order
# seems reasonable w.r.t given table
n, theta, phi = (0,
                 np.random.uniform(0, np.pi, 1),
                 np.random.uniform(0, 2*np.pi, 1))
for m in range(-n, n+1):
    print(np.allclose(
        sph_harm_y(n, m, theta, phi),
        1/np.sqrt(4*np.pi)))

n, theta, phi = (1,
                 np.random.uniform(0, np.pi, 1),
                 np.random.uniform(0, 2*np.pi, 1))
for m in range(-n, n+1):
    if m == -1:
        tmp = np.sqrt(3/8/np.pi) * np.sin(theta) * np.exp(-1j*phi)
    elif m == 0:
        tmp = np.sqrt(3/4/np.pi) * np.cos(theta)
    elif m == 1:
        tmp = -np.sqrt(3/8/np.pi) * np.sin(theta) * np.exp(+1j*phi)
    print(np.allclose(sph_harm_y(n, m, theta, phi), tmp))

n, theta, phi = (4,
                 np.random.uniform(0, np.pi, 1),
                 np.random.uniform(0, 2*np.pi, 1))
for m in range(-n, n+1):
    if m == -4:
        tmp = (np.sqrt(315/512/np.pi) * np.sin(theta)**4 * np.exp(-4j*phi))
    elif m == -3:
        tmp = (np.sqrt(315/64/np.pi) * np.cos(theta) * np.sin(theta)**3 * np.exp(-3j*phi))
    elif m == -2:
        tmp = (np.sqrt(45/128/np.pi) * (7*np.cos(theta)**2-1) * np.sin(theta)**2 * np.exp(-2j*phi))
    elif m == -1:
        tmp = (np.sqrt(45/64/np.pi) * (7*np.cos(theta)**3 - 3*np.cos(theta)) * np.sin(theta) * np.exp(-1j*phi))
    elif m == 0:
        tmp = (np.sqrt(9/256/np.pi) * (35*np.cos(theta)**4 - 30*np.cos(theta)**2 + 3))
    elif m == +1:
        tmp = (-np.sqrt(45/64/np.pi) * (7*np.cos(theta)**3 - 3*np.cos(theta)) * np.sin(theta) * np.exp(+1j*phi))
    elif m == +2:
        tmp = (np.sqrt(45/128/np.pi) * (7*np.cos(theta)**2-1) * np.sin(theta)**2 * np.exp(+2j*phi))
    elif m == +3:
        tmp = (-np.sqrt(315/64/np.pi) * np.cos(theta) * np.sin(theta)**3 * np.exp(+3j*phi))
    if m == +4:
        tmp = (np.sqrt(315/512/np.pi) * np.sin(theta)**4 * np.exp(+4j*phi))
    print(np.allclose(sph_harm_y(n, m, theta, phi), tmp))

print('check Legendre Polynomial')

x = np.linspace(-1, 1, num=2**16, endpoint=True)
# checking 0th, 1st and 4th order seems reasonable w.r.t given table
n, m = 0, 0
print(np.allclose(1, assoc_legendre_p(n, m, x)))
n, m = 1, -1
print(np.allclose(1/2*(1-x**2)**(1/2), assoc_legendre_p(n, m, x)))
n, m = 1, 0
print(np.allclose(x, assoc_legendre_p(n, m, x)))
n, m = 1, +1
print(np.allclose(-(1-x**2)**(1/2), assoc_legendre_p(n, m, x)))
# explicitly check if this bug
# https://github.com/scipy/scipy/issues/23101
# was fixed:
n, m, x_tst = 1, 0, -1
print(np.allclose(x_tst, assoc_legendre_p(1, 0, x_tst)))

n = 4
for m in range(-n, n+1):
    P_sp = assoc_legendre_p(n, m, x)
    if m == -4:
        P_eq = 1/384*(1-x**2)**2
    elif m == -3:
        P_eq = 1/48*x*(1-x**2)**(3/2)
    elif m == -2:
        P_eq = 1/48*(7*x**2-1)*(1-x**2)
    elif m == -1:
        P_eq = 1/8*(7*x**3-3*x)*(1-x**2)**(1/2)
    elif m == 0:
        P_eq = 1/8*(35*x**4-30*x**2+3)
    elif m == +1:
        P_eq = -5/2*(7*x**3-3*x)*(1-x**2)**(1/2)
    elif m == +2:
        P_eq = 15/2*(7*x**2-1)*(1-x**2)
    elif m == +3:
        P_eq = -105*x*(1-x**2)**(3/2)
    elif m == +4:
        P_eq = 105*(1-x**2)**2
    else:
        P_eq = []
    print(np.allclose(P_sp, P_eq))

print('all tests should yield True')
