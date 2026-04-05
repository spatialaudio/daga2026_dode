# see https://github.com/spatialaudio/daga2026_dode

# paper figure 4(a)

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from pyfar.classes.coordinates import cart2sph
from util_cap import (figure_path,
                      pressure_impulse_response_with_filterbank,
                      surface_velocity,
                      vnm_dodecahedron,
                      )


plt.rcParams.update({'font.size': 10})
figw, figh, dpi = (37/2)/2.54 / 2, (37/2)/2.54 / 2 * 20/16, 600
# 37 cm is the column width for Uni Rostock DIN A0 poster
# we go for 37/2 cm with 600 dpi and font size 10
# and use \includegraphics[width=37cm] in tex
# this handling looks nicer than png renderings
# with original size 37cm and font sizes >= 16

fig, axs = plt.subplots(nrows=2, ncols=1,
                        figsize=(figw, figh),
                        subplot_kw=dict(projection='3d'),
                        constrained_layout=True)

# sphere sampling such that plot_surface has
# nice triangulation patches
M = 2**8
u = np.linspace(0, 2*np.pi, 2*M)
v = np.linspace(0, np.pi, M)
# get unit sphere with polar/azimuth convention:
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
phi, theta, _ = cart2sph(x, y, z)
Theta = np.array([np.reshape(theta, (-1)), np.reshape(phi, (-1))]).T
print(Theta)

# dode parameters
L = 2**12  # time samples
N = 24  # SHT max order
fs = 32000  # Hz
rho0 = 1.2041  # kg/m^3
c = 343  # m/s
R = 0.4 / 2  # m
ac_dly_smp = 200
r = ac_dly_smp / fs * c + R  # m
dBmax = 0  # we normalise max value to 0 dB
dBspan = 36
dBmin = dBmax - dBspan
# Hz -> for N=24, the spatial aliasing is properly modelled
# for this third-octave:
fm = 4000
bw_oct = 1/3

# gl = np.ones(12)
dither_dB_span = 2
np.random.seed(3)  # for reproducible/consistent output
gl = 10**(np.random.uniform(-dither_dB_span,
                            +dither_dB_span,
                            12)/20)
beta_kaiser = 5
alpha_cap = 20 * np.pi/180
Lfir = 21
n0 = 10

# velocity on surface
vnm = vnm_dodecahedron(N, beta_kaiser, gl, alpha_cap)
velocity = surface_velocity(vnm, Theta)
velocity = np.reshape(velocity, theta.shape)
velocity_dB = 20*np.log10(np.abs(np.copy(velocity)))
velocity_dB -= np.max(velocity_dB)
velocity_dB += dBmax
velocity_dB[velocity_dB < dBmin] = dBmin
C_velocity_dB = np.copy(velocity_dB)

# pressure on surface
f = np.logspace(np.log10(2), np.log10(fs//2), num=2**10)  # Hz
pressure_ir = pressure_impulse_response_with_filterbank(
        vnm, f, Theta, r, R, L,
        c=c, rho0=rho0, fs=fs,
        farfield_flag=False, flag_pfe_only=False,
        Lfir=Lfir, n0=n0, num_fc=27)
P = np.fft.fft(pressure_ir, axis=1)
df = fs / P.shape[1]
f_fft = np.arange(P.shape[1]) * df
idxl = np.where(f_fft <= fm*2**(-bw_oct/2))[0][-1]
idxh = np.where(f_fft > fm*2**(+bw_oct/2))[0][0]
print(f_fft[idxl], f_fft[idxh])
P_dB = 10*np.log10(np.sum(np.abs(P[:, idxl:idxh])**2, axis=1))
P_dB = np.reshape(P_dB, theta.shape)
P_dB -= np.max(P_dB)
P_dB += dBmax
P_dB[P_dB < dBmin] = dBmin
C_P_dB = np.copy(P_dB)

# velocity sphere plot
min, max = C_velocity_dB.min(), C_velocity_dB.max()
bounds = np.arange(dBmin, dBmax+3, 3)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
color_map = cm.viridis
scalarMap = cm.ScalarMappable(norm=norm, cmap=color_map)
C_colored = scalarMap.to_rgba(C_velocity_dB)
surf = axs[0].plot_surface(
    x, y, z, rstride=1, cstride=1,
    facecolors=C_colored,
    cmap=color_map,
    antialiased=True)
m = cm.ScalarMappable(
    cmap=surf.cmap, norm=norm)
m.set_array(C_velocity_dB)
clb = fig.colorbar(m, location='left', pad=0.1, shrink=0.8, ax=axs[0])
clb.set_label(r'velocity $v(\mathbf{\theta})$, dB rel max',
              loc='center', labelpad=0)

# pressure sphere plot
min, max = C_P_dB.min(), C_P_dB.max()
bounds = np.arange(dBmin, dBmax+3, 3)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
color_map = cm.magma
scalarMap = cm.ScalarMappable(norm=norm, cmap=color_map)
C_colored = scalarMap.to_rgba(C_P_dB)
surf = axs[1].plot_surface(
    x, y, z, rstride=1, cstride=1,
    facecolors=C_colored,
    cmap=color_map,
    antialiased=True)
m = cm.ScalarMappable(
    cmap=surf.cmap, norm=norm)
m.set_array(C_P_dB)
clb = fig.colorbar(m, location='left', pad=0.1, shrink=0.8, ax=axs[1])
clb.set_label(r'pressure $p(\mathbf{\theta})$, dB rel max',
              loc='center', labelpad=0)

for i in range(2):
    axs[i].set_xticks([0], labels=[''])
    axs[i].set_yticks([0], labels=[''])
    axs[i].set_zticks([0], labels=[''])
    axs[i].set_xlim([-1, 1])
    axs[i].set_ylim([-1, 1])
    axs[i].set_zlim([-1, 1])
    axs[i].set_aspect('equal')
    axs[i].text(1.1, 0, -1, 'x')
    axs[i].text(0, 1, 1, 'z')

plt.savefig(figure_path()+'dodecahedron_sphere.png', dpi=dpi)
