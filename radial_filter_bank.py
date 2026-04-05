# see https://github.com/spatialaudio/daga2026_dode

# paper figures 2(a) and 2(b)

import matplotlib.pyplot as plt
import numpy as np
from util_cap import (figure_path,
                      filterbank,
                      filterbank_impulse_responses,
                      filterbank_frequency_responses,
                      filterbank_via_ifft_impulse_responses,
                      filterbank_via_ifft_frequency_responses,
                      )


col_hro = '#004a99'  # uni color
plt.rcParams.update({'font.size': 10})
# figw, figh, dpi = (37/2)/2.54, (37/2)/2.54 * 8/16, 600
# 37 cm is the column width for Uni Rostock DIN A0 poster
# we go for 37/2 cm with 600 dpi and font size 10
# and use \includegraphics[width=37cm] in tex
# this handling looks nicer than png renderings
# with original size 37cm and font sizes >= 16

res_type = 'ls_residual'  # 'ls_residual' or 'scipy_residual'
farfield_flag = False
flag_pfe_only = False

L = 2**12  # samples to represent / calc IRs
# make sure that ac_dly_smp >> n0 and ac_dly_smp << L
fs = 32000  # Hz
c = 343  # m/s
# rho0 = 1.2041  # kg/m^3
R = 0.4 / 2  # m
r = 200 / fs * c + R  # m, r should be chosen such that
# (r-R)/c*fs is int -> then we don't have
# an additional error due to fractional delay issues
# the current time alignment implementation in the filter design
# relies on int !!! fractional delay TBD
ac_dly_smp = int(np.round((r-R) / c * fs))  # should be == (r-R)/c*fs
print((r-R)/c*fs, ac_dly_smp)
dB_norm = 20*np.log10(R/r)
print(r, R, dB_norm)

f = np.logspace(np.log10(2), np.log10(fs//2), num=2**10)  # Hz
w = 2*np.pi*f  # rad/s
# w_c = w/c  # rad / m
# s = 1j*w  # rad/s
# kr = w_c * r  # rad
# kR = w_c * R  # rad
# Z0 = rho0*c  # kg/m^3 m/s

# impulse responses ###########################################################
figw, figh, dpi = (37/2)/2.54, (37/2)/2.54 * 8/16, 600
N = 4

h_ifft = filterbank_via_ifft_impulse_responses(
    N, fs, L, r, R, c, farfield_flag, flag_pfe_only)

Filters_corr = filterbank(N, w, r, R, c=c, fs=fs,
                          farfield_flag=farfield_flag,
                          res_type=res_type,
                          ii_type='DCIIM')
h_corr_modal = filterbank_impulse_responses(
    Filters_corr, L, ac_dly_smp, r, R, flag_pfe_only)

Lfir, n0, num_fc = 21, 10, 27  # is this the best heuristic choice?
Filters_ls = filterbank(N, w, r, R, c=c, fs=fs,
                        farfield_flag=farfield_flag,
                        res_type=res_type,
                        ii_type='BLIIM',
                        Lfir=Lfir, n0=n0,
                        fc_min=fs//2**9, fc_max=fs//2**1, num_fc=num_fc)
h_ls_modal = filterbank_impulse_responses(
    Filters_ls, L, ac_dly_smp, r, R, flag_pfe_only)

# get rid of log10(0)
tmp = 10**(-200/20)
h_corr_modal[h_corr_modal == 0] = tmp
h_ls_modal[h_ls_modal == 0] = tmp
h_ifft[h_ifft == 0] = tmp

# compensate for amplitude decay by taking inverse term r/R
h_ifft *= r/R
h_corr_modal *= r/R
h_ls_modal *= r/R
# for dB the radial fraction is then already compensated
H_ifft = 20*np.log10(np.abs(h_ifft))
H_corr_modal = 20*np.log10(np.abs(h_corr_modal))
H_ls_modal = 20*np.log10(np.abs(h_ls_modal))

# get correct samples w.r.t acoustic propagation time
# all IRs are already properly time aligned
# so only a np.arange is needed for sample index array
t = np.arange(L)
n0_corr = Filters_corr[0][2]
n0_ls = Filters_ls[0][2]

fig, axs = plt.subplots(2, 3)
fig.set_figwidth(figw)
fig.set_figheight(figh)
axs[0, 0].plot(t, h_ifft.T, lw=1)
axs[0, 1].plot(t, h_corr_modal.T, lw=1)
axs[0, 2].plot(t, h_ls_modal.T, lw=1)
axs[1, 0].plot(t, H_ifft.T, lw=1)
axs[1, 1].plot(t, H_corr_modal.T, lw=1)
axs[1, 2].plot(t, H_ls_modal.T, lw=1)

for i in range(2):
    for ii in range(3):
        # axs[1, ii].plot(
        #     [ac_dly_smp, ac_dly_smp], [-100, +100],
        #     '-', lw=0.5, color=col_hro)
        axs[i, ii].grid(True)
        axs[0, ii].set_xlim(ac_dly_smp-20, ac_dly_smp+20)
        axs[0, ii].set_xticks(np.arange(180, 220+10, 10))
        axs[1, ii].set_xlim(ac_dly_smp-50, ac_dly_smp+150)
        axs[1, ii].set_xticks(np.arange(150, 350+50, 50))

for i in range(2):
    axs[i, 2].plot(
        [ac_dly_smp+n0_ls, ac_dly_smp+n0_ls], [-100, +100],
        '-', lw=1, color=col_hro)
    axs[i, 2].plot(
        [ac_dly_smp-n0_ls, ac_dly_smp-n0_ls], [-100, +100],
        '-', lw=1, color=col_hro)

axs[0, 0].text(181, 0.95, 'K=%d' % L,
               horizontalalignment='left',
               verticalalignment='top')
axs[1, 2].text(ac_dly_smp, 0, 'FIR',
               horizontalalignment='center',
               verticalalignment='bottom',
               color=col_hro)

for i in range(3):
    axs[1, i].set_yticks(np.arange(-8*12, 0+12, 12))
    axs[1, i].set_ylim([-12*8, 0])
    axs[0, i].set_yticks(np.arange(-1, 1+0.5, 0.5))
    axs[0, i].set_ylim([-.55, 1.05])

axs[1, 1].set_xlabel('time sample index')
axs[0, 0].set_ylabel('linear')
axs[1, 0].set_ylabel('level in dB')
axs[0, 0].set_title('IDFT')
axs[0, 1].set_title('DCIIM')
axs[0, 2].set_title('LS-BLIIM')

plt.tight_layout()
plt.savefig(figure_path()+'radial_filter_bank_impulse_responses.png', dpi=600)

# frequency responses #########################################################
figw, figh, dpi = (37/2)/2.54, (37/2)/2.54 * 5/16, 600
N = 24

Filters_corr = filterbank(N, w, r, R, c=c, fs=fs,
                          farfield_flag=farfield_flag,
                          res_type=res_type,
                          ii_type='DCIIM')

Filters_ls = filterbank(N, w, r, R, c=c, fs=fs,
                        farfield_flag=farfield_flag,
                        res_type=res_type,
                        ii_type='BLIIM',
                        Lfir=Lfir, n0=n0,
                        fc_min=fs//2**9, fc_max=fs//2**1, num_fc=num_fc)

H_ifft = filterbank_via_ifft_frequency_responses(
    N, f, r, R, c, farfield_flag, flag_pfe_only)
H_corr_modal = filterbank_frequency_responses(
    Filters_corr, f, fs, ac_dly_smp, r, R, flag_pfe_only)
H_ls_modal = filterbank_frequency_responses(
    Filters_ls, f, fs, ac_dly_smp, r, R, flag_pfe_only)

# get rid of log10(0)
tmp = 10**(-200/20)
H_ifft[H_ifft == 0] = tmp
H_corr_modal[H_corr_modal == 0] = tmp
H_ls_modal[H_ls_modal == 0] = tmp

H_ls_modal[H_ls_modal == 0] = 1e-16
H_corr_modal[H_corr_modal == 0] = 1e-16

fig, axs = plt.subplots(1, 1)
fig.set_figwidth(figw)
fig.set_figheight(figh)

axs.semilogx(f, 20*np.log10(np.abs(H_ifft.T[:, 0]))-dB_norm,
             ':', lw=2, label=r'$1+\hat{H}_n(s,r,R)$')
plt.gca().set_prop_cycle(None)  # reset colorcycle
axs.semilogx(f, 20*np.log10(np.abs(H_corr_modal.T[:, 0]))-dB_norm,
             lw=0.334, label='DCIIM')
plt.gca().set_prop_cycle(None)  # reset colorcycle
axs.semilogx(f, 20*np.log10(np.abs(H_ls_modal.T[:, 0]))-dB_norm,
             lw=0.75, label='LS-BLIIM')
plt.gca().set_prop_cycle(None)  # reset colorcycle
axs.semilogx(f, 20*np.log10(np.abs(H_ifft.T))-dB_norm, ':', lw=2)
plt.gca().set_prop_cycle(None)  # reset colorcycle
axs.semilogx(f, 20*np.log10(np.abs(H_corr_modal.T))-dB_norm, lw=0.334)
plt.gca().set_prop_cycle(None)  # reset colorcycle
axs.semilogx(f, 20*np.log10(np.abs(H_ls_modal.T))-dB_norm, lw=0.75)

axs.set_yticks(np.arange(-48*3, 0+24, 24))
axs.set_ylim([-48*3, 6])
axs.set_xlim([2, 20000])
if False:  # zoom into high frequencies
    axs.set_yticks(np.arange(-6, 6+1, 1))
    axs.set_ylim([-6, 6])
    axs.set_xlim([5000, 10000])
axs.set_xlabel('frequency in Hz')
axs.set_ylabel('level in dB')
axs.grid(True)
axs.legend()
plt.tight_layout()
plt.savefig(figure_path()+'radial_filter_bank_frequency_responses.png',
            dpi=600)
