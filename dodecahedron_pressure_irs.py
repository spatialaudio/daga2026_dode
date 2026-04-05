# see https://github.com/spatialaudio/daga2026_dode

# paper figure 4(b)

import matplotlib.pyplot as plt
import numpy as np

from spharpy.samplings import dodecahedron
from util_cap import (figure_path,
                      vnm_dodecahedron,
                      pressure_impulse_response_with_filterbank)

plt.rcParams.update({'font.size': 10})
figw, figh, dpi = (37/2)/2.54 / 2, (37/2)/2.54 / 2 * 20/16, 600
# 37 cm is the column width for Uni Rostock DIN A0 poster
# we go for 37/2 cm with 600 dpi and font size 10
# and use \includegraphics[width=37cm] in tex
# this handling looks nicer than png renderings
# with original size 37cm and font sizes >= 16
col_hro = '#004a99'  # original uni

fig, ax = plt.subplots(2, 1)
fig.set_figwidth(figw)
fig.set_figheight(figh)

# directions where to evaluate the pressure
# use (i) the first dode cap, i.e. the one showing into x-axis upwards
# and (ii) north pole and (iii) directly into x-axis
sph_sampling = dodecahedron()
theta, phi = sph_sampling.colatitude, sph_sampling.azimuth
Theta = np.array([[theta[0], 0, np.pi/2],
                  [phi[0], 0, 0]]).T
print(Theta.shape)

# dode parameters
L = 2**12  # time samples
N = 24  # SHT max order
fs = 32000  # Hz
f = np.logspace(np.log10(2), np.log10(fs//2), num=2**10)  # Hz
rho0 = 1.2041  # kg/m^3
c = 343  # m/s
Z0 = rho0*c  # kg/m^3 m/s
R = 0.4 / 2  # m
ac_dly_smp = 200
r = ac_dly_smp / fs * c + R  # m
lin_norm = Z0 * R / r
dB_norm = 20*np.log10(lin_norm)

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

vnm = vnm_dodecahedron(N, beta_kaiser, gl, alpha_cap)
h = pressure_impulse_response_with_filterbank(
        vnm, f, Theta, r, R, L,
        c=c, rho0=rho0, fs=fs,
        farfield_flag=False, flag_pfe_only=False,
        Lfir=Lfir, n0=n0, num_fc=27)
print(np.min(np.imag(h)), np.max(np.imag(h)))
h, t = np.real(h), np.arange(L)

case_str = ['(i)', '(ii)', '(iii)']

for n in range(Theta.shape[0]):
    ax[0].plot(t, h[n, :] / lin_norm,
               label=r'%s $\phi=$%1.f$^\circ$, $\theta=$%2.1f$^\circ$' %
               (case_str[n], Theta[n, 1]*180/np.pi, Theta[n, 0]*180/np.pi))
    ax[1].plot(t, 20*np.log10(np.abs(h[n, :]))-dB_norm,
               label=r'%s $\phi=$%1.f$^\circ$, $\theta=$%2.1f$^\circ$' %
               (case_str[n], Theta[n, 1]*180/np.pi, Theta[n, 0]*180/np.pi))
ax[0].set_xticks(np.arange(190, 230+10, 10))
ax[0].set_yticks(np.arange(-1, 1+0.5, 0.5))
ax[1].set_xticks(np.arange(150, 350+50, 50))
ax[1].set_yticks(np.arange(-8*12, 0+12, 12))
ax[0].set_xlim(ac_dly_smp-11, ac_dly_smp+30)
ax[0].set_ylim([-1, 1])
ax[1].set_xlim(ac_dly_smp-50, ac_dly_smp+150)
ax[1].set_ylim([-12*8, 0])
ax[0].set_ylabel(r'$p(t, r, \theta, \phi) \,\,/\,\, (\rho_0 c \cdot \frac{R}{r})$')
ax[1].set_ylabel(r'level in dB')
ax[1].set_xlabel('time sample index')
ax[0].legend(loc='upper right', labelspacing=0, ncols=1, fontsize='small')
ax[0].grid(True)
ax[1].grid(True)

for i in range(2):
    ax[i].plot(
        [ac_dly_smp+n0, ac_dly_smp+n0], [-100, +100],
        '-', lw=1, color=col_hro)
    ax[i].plot(
        [ac_dly_smp-n0, ac_dly_smp-n0], [-100, +100],
        '-', lw=1, color=col_hro)
ax[1].text(ac_dly_smp, -96, 'FIR',
           horizontalalignment='center',
           verticalalignment='bottom',
           color=col_hro)

plt.tight_layout()
plt.savefig(figure_path()+'dodecahedron_pressure_irs.png', dpi=dpi)
