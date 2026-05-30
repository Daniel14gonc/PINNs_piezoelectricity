"""Piezoelectric material coefficients used by the simulation.

The values and the assembly of the coupled stiffness / coupling /
permittivity matrices follow the original ``geom_creation.ipynb``
notebook one-to-one.
"""

from __future__ import annotations

import numpy as np


# --- Basic material properties ----------------------------------------------
scale = 1  # 1000 ** 2

E = 2.0e9 / scale          # Elastic modulus (N/m^2)
NU = 0.29                  # Poisson's ratio
G = 0.775e9 / scale        # Shear modulus (N/m^2)

d31 = 2.2e-11              # Piezoelectric strain coefficients (C/N)
d33 = -3.0e-11

rel_permittivity = 12
permittivity_free_space = 8.854187817620e-12 / 1  # F/m

C11 = E / (1 - NU ** 2)
C12 = NU * C11
C22 = C11


# --- Coupled matrices --------------------------------------------------------
c2d = np.array([[C11, C12, 0],
                [C12, C11, 0],
                [0,   0,   G]])

pze_D = np.array([[0, d31],
                  [0, d33],
                  [0, 0]])

pze_E = c2d @ pze_D
D_const_stress = np.array([[rel_permittivity, 0],
                           [0, rel_permittivity]]) * permittivity_free_space
D_const_strain = D_const_stress - pze_D.T @ pze_E

cond_scaling = 1

c11 = np.array([c2d[0, 0], c2d[0, 2], c2d[2, 0], c2d[2, 2]])
c12 = np.array([[c2d[0, 2], c2d[0, 1]],
                [c2d[2, 2], c2d[1, 2]]])
c21 = c12.T

c22 = np.array([c2d[2, 2], c2d[1, 2], c2d[2, 1], c2d[1, 1]])
c13 = np.array([[pze_E[0, 0], pze_E[0, 1]],
                [pze_E[2, 0], pze_E[2, 1]]])
c31 = cond_scaling * c13.T
c23 = np.array([[pze_E[2, 0], pze_E[2, 1]],
                [pze_E[1, 0], pze_E[1, 1]]])
c32 = cond_scaling * c23.T

c33 = cond_scaling * np.array([D_const_strain[0, 0],
                               D_const_strain[1, 0],
                               D_const_strain[0, 1],
                               D_const_strain[1, 1]])

ctop = np.concatenate([
    c11,
    c21.flatten('F'),
    -c31.flatten('F'),
    c12.flatten('F'),
    c22,
    -c32.flatten('F'),
    -c13.flatten('F'),
    -c23.flatten('F'),
    -c33,
])

cbot = np.concatenate([
    c11,
    c21.flatten('F'),
    c31.flatten('F'),
    c12.flatten('F'),
    c22,
    c32.flatten('F'),
    c13.flatten('F'),
    c23.flatten('F'),
    -c33,
])


# --- Scalar piezoelectric coefficients used by the loss assembly ------------
epsilon_1 = -c33[0]
epsilon_2 = -c33[-1]

e11_top = -c31[0, 0]
e14_top = -c31[0, 1]
e13_top = -c31[0, 1]
e31_top = -c31[1, 0]
e34_top = -c31[1, 1]

e11_bottom = c31[0, 0]
e14_bottom = c31[0, 1]
e13_bottom = c31[0, 1]
e31_bottom = c31[1, 0]
e34_bottom = c31[1, 1]

e33_top = -c32[-1, -1]
e33_bottom = c32[-1, -1]
