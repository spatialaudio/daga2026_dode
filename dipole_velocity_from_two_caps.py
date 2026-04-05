# see https://github.com/spatialaudio/daga2026_dode

# this is not included in the poster / paper
# but it is informative to check the consistency
# of all involved normalisations/conventions

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from pyfar.classes.coordinates import cart2sph
from util_cap import (figure_path,
                      surface_velocity,
                      vnm_spherical_cap,
                      )

plt.rcParams.update({'font.size': 20})
figw, figh, dpi = 10, 10, 600

fig, axs = plt.subplots(nrows=1, ncols=1,
                        figsize=(figw, figh),
                        subplot_kw=dict(projection='3d'),
                        constrained_layout=True)

# sphere sampling such that plot_surface has
# nice triangulation patches
M = 2**5
u = np.linspace(0, 2*np.pi, 2*M)
v = np.linspace(0, np.pi, M)
# get unit sphere with polar/azimuth convention:
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
phi, theta, _ = cart2sph(x, y, z)
Theta = np.array([np.reshape(theta, (-1)), np.reshape(phi, (-1))]).T

N = 40  # SHT max order

# velocity on surface

beta_kaiser = 0

# positive north pole cap
alpha_cap, theta_cap, phi_cap = np.pi/2, 0, 0
vnm_north = +vnm_spherical_cap(N,
                               alpha_cap,
                               theta_cap,
                               phi_cap,
                               beta_kaiser)

# negative south pole cap
alpha_cap, theta_cap, phi_cap = np.pi/2, np.pi, 0
vnm_south = -vnm_spherical_cap(N,
                               alpha_cap,
                               theta_cap,
                               phi_cap,
                               beta_kaiser)

vnm = vnm_north + vnm_south
# print(vnm)

velocity = surface_velocity(vnm, Theta)
velocity = np.reshape(velocity, theta.shape)
print('imag negligible?', np.min(np.imag(velocity)), np.max(np.imag(velocity)))
velocity = np.real(velocity)
C_velocity = np.copy(velocity)
print('min/max velocity:', np.min(velocity), np.max(velocity))

# velocity sphere plot -> we should get +1 at north pole and -1 at south pole
min, max = -1, +1
bounds = np.arange(min, max+0.125, 0.125)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
color_map = cm.RdBu_r
scalarMap = cm.ScalarMappable(norm=norm, cmap=color_map)
C_colored = scalarMap.to_rgba(C_velocity)
surf = axs.plot_surface(
    x, y, z, rstride=1, cstride=1,
    facecolors=C_colored,
    cmap=color_map,
    antialiased=True)
m = cm.ScalarMappable(
    cmap=surf.cmap, norm=norm)
m.set_array(C_velocity)
clb = fig.colorbar(m, location='left', pad=0.1, shrink=0.8, ax=axs)
clb.set_label('velocity (linear)', loc='center', labelpad=0)
axs.set_xticks([0], labels=[''])
axs.set_yticks([0], labels=[''])
axs.set_zticks([0], labels=[''])
axs.set_xlim([-1, 1])
axs.set_ylim([-1, 1])
axs.set_zlim([-1, 1])
axs.set_aspect('equal')
axs.text(1.1, 0, -1, 'x')
axs.text(0, 1, 1, 'z')
axs.set_title('min: %1.4f, max:%1.4f' % (np.min(velocity), np.max(velocity)))

plt.savefig(figure_path()+'dipole_velocity_from_two_caps.png', dpi=dpi)
