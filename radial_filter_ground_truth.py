# see https://github.com/spatialaudio/daga2026_dode

# paper figure 1

import matplotlib.pyplot as plt
import numpy as np
from util_cap import (figure_path,
                      radial_filter_frequency_response,
                      radial_filter_min_phase_frequency_response)

plt.rcParams.update({'font.size': 10})
figw, figh, dpi = (37/2)/2.54, (37/2)/2.54 * 7.5/16, 600
# 37 cm is the column width for Uni Rostock DIN A0 poster
# we go for 37/2 cm with 600 dpi and font size 10
# and use \includegraphics[width=37cm] in tex
# this handling looks nicer than png renderings
# with original size 37cm and font sizes >= 16

fs = 32000  # Hz
c = 343  # m/s

R = 0.2  # m
r = 2  # m
print(r, R)
n = [0, 1, 2, 3, 4, 5, 6, 7]

color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
f = np.logspace(np.log10(2), np.log10(20000), num=2**9)  # Hz
w = 2*np.pi*f  # rad/s
dB_norm = 20*np.log10(R/r)

fig, axs = plt.subplots(2, 2)
fig.set_figwidth(figw)
fig.set_figheight(figh)
cnt = 0
for order in n:

    H_rad = radial_filter_frequency_response(
        order, w, r, R, c, farfield_flag=False)
    H_min = radial_filter_min_phase_frequency_response(
        order, w, r, R, c, farfield_flag=False)
    H_rad_far = radial_filter_frequency_response(
        order, w, r, R, c, farfield_flag=True)
    H_min_far = radial_filter_min_phase_frequency_response(
        order, w, r, R, c, farfield_flag=True)

    # always compensate for the R/r-fraction with dB_norm
    # we don't need to compensate phases in the dB domain
    if order == 1:
        axs[0, 0].semilogx(f, 20*np.log10(np.abs(H_rad))-dB_norm,
                           color=color[cnt], label='general')
        axs[0, 0].semilogx(f, 20*np.log10(np.abs(H_rad_far))-dB_norm,
                           '-.', lw=0.75,
                           color=color[cnt], label='far approx')
    else:
        axs[0, 0].semilogx(f, 20*np.log10(np.abs(H_rad))-dB_norm,
                           color=color[cnt])
        axs[0, 0].semilogx(f, 20*np.log10(np.abs(H_rad_far))-dB_norm,
                           '-.', lw=0.75,
                           color=color[cnt])
    axs[0, 1].semilogx(f, 20*np.log10(np.abs(-H_min)),
                       color=color[cnt],
                       label='n='+str(order))
    # compensate the time delay to get 1+Hn and thus get the correct min-phase:
    axs[1, 0].semilogx(f,
                       180 / np.pi * np.unwrap(
                           np.angle(H_rad * np.exp(+1j*w/c*(r-R)))),
                       color=color[cnt])
    # this is already min-phase because the
    # radial_filter_min_phase_frequency_response() intentionally returns it
    axs[1, 1].semilogx(f,
                       180 / np.pi * np.unwrap(
                           np.angle(-H_min)),
                       color=color[cnt])

    cnt += 1

for i in range(2):
    axs[1, i].set_xlabel('frequency in Hz')
axs[0, 0].set_ylabel('level in dB')
axs[1, 0].set_ylabel('phase in deg')
for i in range(2):
    for ii in range(2):
        axs[i, ii].grid(True)
        axs[i, ii].set_xlim([2, 20000])
axs[0, 0].set_yticks(np.arange(-48*3, 12, 48))
axs[0, 1].set_yticks(np.arange(-48, 24, 12))
axs[1, 0].set_yticks(np.arange(0, 540+90, 90))
axs[1, 1].set_yticks(np.arange(-90, 90+45, 45))
axs[0, 0].set_ylim([-48*3, 6])
axs[0, 1].set_ylim([-48, 12])
axs[1, 0].set_ylim([0, 540])
axs[1, 1].set_ylim([-90, 90])
axs[0, 0].set_title(r'min-phase high-pass  $1+\hat{H}_n(s,r,R)$')
axs[0, 1].set_title(r'min-phase low-pass    $-\hat{H}_n(s,r,R)$')
axs[0, 0].legend(loc='lower right', labelspacing=0, ncols=1, fontsize='small')
axs[0, 1].legend(loc='lower left', labelspacing=0, ncols=2, fontsize='small')

plt.tight_layout()
plt.savefig(figure_path()+'radial_filter_ground_truth.png',  dpi=dpi)
