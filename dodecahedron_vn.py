# see https://github.com/spatialaudio/daga2026_dode

# paper figure 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from spharpy.samplings import t_design, equiangular, lebedev
from util_cap import figure_path, vnm_dodecahedron


col_hro = '#004a99'  # uni color
plt.rcParams.update({'font.size': 10})
figw, figh, dpi = (37/2)/2.54, (37/2)/2.54 * 10/16, 600
# 37 cm is the column width for Uni Rostock DIN A0 poster
# we go for 37/2 cm with 600 dpi and font size 10
# and use \includegraphics[width=37cm] in tex
# this handling looks nicer than png renderings
# with original size 37cm and font sizes >= 16


alpha_cap = 20 * np.pi/180
beta_kaiser = 0
N = 30

fig, axs = plt.subplots(2, 1)
fig.set_figwidth(figw)
fig.set_figheight(figh)
for i in range(2):
    for n in range(N+1):
        axs[i].plot([n, n], [-8*12, 12], '-', color='silver', lw=0.5)
    for db in np.arange(-8*12, 12+12, 12):
        axs[i].plot([-1, N+1], [db, db], '-', color='silver', lw=0.5)

# we want a t-design sampling of the sphere with about 1000 points
sph_sampling = t_design(n_max=22, criterion='const_angular_spread')
# sph_sampling = equiangular(n_max=15)  # check with another grid
# sph_sampling = lebedev(n_max=26)  # check with another grid
Theta = np.array([sph_sampling.colatitude, sph_sampling.azimuth]).T
print(Theta.shape)  # we get 1059 sampling points on the sphere

# uniform caps
gl = np.ones(12)
vnm = vnm_dodecahedron(N, beta_kaiser, gl, alpha_cap)
vn = np.zeros((Theta.shape[0], N+1), dtype='complex128')
cnt = 0
for n in range(N+1):
    for m in range(-n, n+1):
        vn[:, n] += vnm[cnt] * sph_harm_y(n, m,
                                          Theta[:, 0], Theta[:, 1])
        cnt += 1
vn_dB = 20*np.log10(np.abs(vn))
# tf...top figure
tf = axs[0].violinplot(vn_dB, np.arange(N+1),
                       widths=1,
                       showmeans=False,
                       showmedians=True,
                       showextrema=False,
                       bw_method=0.5)  # explicitly use 0.5 dB bandwidth

# level dithered caps
dB_span = 2
np.random.seed(3)  # for reproducible/consistent output
gl = 10**(np.random.uniform(-dB_span, +dB_span, 12)/20)
vnm = vnm_dodecahedron(N, beta_kaiser, gl, alpha_cap)
vn = np.zeros((Theta.shape[0], N+1), dtype='complex128')
cnt = 0
for n in range(N+1):
    for m in range(-n, n+1):
        vn[:, n] += vnm[cnt] * sph_harm_y(n, m,
                                          Theta[:, 0], Theta[:, 1])
        cnt += 1
vn_dB = 20*np.log10(np.abs(vn))
# bf...bottom figure
bf = axs[1].violinplot(vn_dB, np.arange(N+1),
                       widths=1,
                       showmeans=False,
                       showmedians=True,
                       showextrema=False,
                       bw_method=0.5)

# figure housekeeping
for pc in tf['bodies']:
    pc.set_facecolor(col_hro)
    pc.set_edgecolor(col_hro)
    pc.set_linewidth(1)
    pc.set_alpha(0.3)
tf['cmedians'].set_color('black')
for pc in bf['bodies']:
    pc.set_facecolor(col_hro)
    pc.set_edgecolor(col_hro)
    pc.set_linewidth(1)
    pc.set_alpha(0.3)
bf['cmedians'].set_color('black')

axs[0].text(30, 0.5, 'all caps have unit-gain',
            horizontalalignment='right')
axs[1].text(30, 0.5, r'caps level with $\pm$ %2.1f dB uniform-PDF dither'
            % dB_span,
            horizontalalignment='right')

axs[1].set_xticks(np.arange(N+1))
axs[0].set_xticks([0, 6, 10, 12, 16, 18, 20, 22, 24, 26, 28, 30],
                  labels=['0', '6', '10', '12', '16', '18', '20', '22', '24',
                          '26', '28', '30'])
for i in range(2):
    axs[i].set_yticks(np.arange(-8*12, 12+12, 12))
    axs[i].set_xlim([-1, N+1])
    axs[i].set_ylim([-8*12, 12])
    axs[i].set_ylabel(r'level of $v_n(\mathbf{\theta})$ in dB')
axs[1].set_xlabel('modal order n')

plt.tight_layout()
plt.savefig(figure_path()+'dodecahedron_vn.png', dpi=dpi)
