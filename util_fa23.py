# see https://github.com/spatialaudio/daga2026_dode

# Nara Hahn (https://github.com/narahahn) is the author of the code below
# which originates from the project
# N. Hahn, F. Schultz, S. Spors (2023):
# Accurate time-domain simulation of spherical microphone arrays.
# Forum Acusticum, Torino, pp. 599–606.
# https://dael.euracoustics.org/confs/fa2023/data/articles/001255.pdf
# https://www.doi.org/10.61782/fa.2023.1255

# only the relevant code parts were adopted

import numpy as np
from scipy.signal import freqz
from scipy.special import spherical_jn, spherical_yn


def spherical_hn2(n, z, derivative=False):
    return spherical_jn(n, z, derivative) - 1j * spherical_yn(n, z, derivative)


def hn2_poly(n, derivative=False):
    """Bessel polynomial of n-th order.
    Polynomial that characterizes the spherical Hankel functions.
    The coefficients are computed by using the recurrence relation.
    The returned array has a length of n+1. The first coefficient is always 1.

    Parameters
    ----------

    n : int
        Bessel polynomial order.

    """
    if derivative:
        return derivative_hn2_poly(n)
    else:
        beta = np.zeros(n + 1)
        beta[n] = 1
        for k in range(n-1, -1, -1):
            beta[k] = beta[k+1] * (2*n-k) * (k+1) / (n-k) / 2
        return beta


def decrease_hn2_poly_order_by_one(beta):
    """Bessel polynomial of order decreased by 1.
    """
    n = len(beta)-1
    alpha = np.zeros(n)
    for k in range(n-1):
        alpha[k] = beta[k+1] * (k+1) / (2*n-k-1)
    alpha[-1] = 1
    return alpha


def increase_hn2_poly_order_by_one(beta):
    """Bessel polynomial of order increased by 1.
    """
    n = len(beta)
    alpha = np.zeros(n+1)
    for k in range(n):
        alpha[k+1] = beta[k] * (2*n-k-1) / (k+1)
    alpha[0] = alpha[1]
    return alpha


def derivative_hn2_poly(n):
    """Polynomial characterizing the derivative of the spherical Hankel func.
    """
    gamma = hn2_poly(n+1)
    gamma[:-1] -= n * decrease_hn2_poly_order_by_one(gamma)
    return gamma


def derivative_hn2_poly_2(n):
    """
    Polynomial characterizing the derivative of the spherical Hankel functions.
    An alternative recurrence relation is used.
    """
    if n == 0:
        gamma = hn2_poly(1)
    else:
        beta0 = hn2_poly(n)
        beta1 = hn2_poly(n-1)
        gamma = np.zeros(n+2)
        gamma[0] = (n+1)*beta0[0]
        gamma[1] = (n+1)*beta0[1]
        gamma[-1] = 1
        for k in range(2, n+1):
            gamma[k] = (n+1)*beta0[k] + beta1[k-2]
    return gamma


def s_zeros_hn2_poly(n):
    return np.roots(hn2_poly(n)[::-1])


def s_zeros_derivative_hn2_poly(n):
    return np.zeros(n)


def s_poles_derivative_hn2_poly(n):
    return np.roots(derivative_hn2_poly(n)[::-1])


def s_zpk_hn2_poly(n):
    s_zeros = s_zeros_derivative_hn2_poly(n)
    s_poles = s_poles_derivative_hn2_poly(n)
    return s_zeros, s_poles, 1


def s_zpk_ps_rigid_sphere(n):
    s_zeros = s_zeros_hn2_poly(n)
    s_poles = s_poles_derivative_hn2_poly(n)
    return s_zeros, s_poles, 1


def phaseshift_timedelay(delay, w):
    return np.exp(-1j * 2 * np.pi * w * delay)


def phaseshift_sampledelay(n, w, fs):
    return phaseshift_timedelay(delay=n/fs, w=w)


def log_frequency(fmin, fmax, num_f, endpoint=True):
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_f,
                       endpoint=endpoint)


def impulse_invariance(r, p, k, L_fir, n_center, fs, mode, window=None):
    """
    Impulse invariance method.

    Parameters
    ----------
    r : array_like
        Residues.
    p : array_like
        Poles.
    k : float
        Direct throughput.
    L_fir : int
        FIR length.
    n_center : int
        Sample index at which the IIR fitler begins (pre-delay).
    fs : int
        Sampling frequency in Hertz.
    mode : string
        {"uncorrected", "corrected", "dcmatched", "bandlimited",
         "dcmbandlimited"}.
    window : array_like, optional
        Tapering window.

    Returns
    -------
    IIR : list
        List of FOS and SOS section filters.
    FIR : array_like
        FIR coefficients.
    n_center : int
        Sample index at which the IIR fitler begins (pre-delay).

    """
#    p_cplx, p_real = cplxreal(p)
#    num_real = len(p_real)
#    r_cplx, r_real = r[num_real::2], r[:num_real]

    tol = 1e-10
    idx_real = (np.abs(np.imag(p)) < tol)
    p_real = p[idx_real]
    r_real = r[idx_real]
#    num_real = len(p_real)

    p_cplx = p[~idx_real]
    r_cplx = r[~idx_real]
    idx_sort = np.argsort(p_cplx.real)
    p_cplx = p_cplx[idx_sort[::2]]
    r_cplx = r_cplx[idx_sort[::2]]

    filters_real = [fos_real_pole(ri, pi, L_fir, n_center, fs, mode, window)
                    for (ri, pi) in zip(r_real, p_real)]
    filters_cplx = [sos_cplx_poles(ri, pi, L_fir, n_center, fs, mode, window)
                    for (ri, pi) in zip(r_cplx, p_cplx)]

    IIR = []
    FIR = np.zeros(L_fir)
    for filt in filters_real:
        IIR.append((filt[0], filt[1]))
        FIR += filt[2]
    for filt in filters_cplx:
        IIR.append((filt[0], filt[1]))
        FIR += filt[2]
    if len(k) == 1:
        FIR[n_center] += k[0]
    return IIR, FIR, n_center


def impulse_invariance_least_square(
        residues, poles, gain, Lfir, n0, f_control, fs):
    """
    Band-limited impulse invariance method using least squares fit.
    """
    T = 1/fs
    w = 2*np.pi*f_control
    jw = 1j*w
    # z1 = np.exp(-1j*w*T)

    # IIR part: computed by using the conventional impulse invariance method
    IIR, _, _ = impulse_invariance(
        residues, poles, gain, 1, 0, fs, mode='uncorrected')
    H_iir = np.zeros_like(f_control, dtype=complex)
    for iir in IIR:
        H_iir += freqz(iir[0], iir[1], worN=f_control, fs=fs)[1]
    H_iir *= phaseshift_sampledelay(n0, f_control, fs)

    H_ref = np.zeros_like(f_control, dtype=complex)
    for (r, p) in zip(residues, poles):
        H_ref += r/(jw - p)
    H_ref *= phaseshift_sampledelay(n0, f_control, fs)
    a = H_ref - H_iir  # difference between the target spectrum and IIR model

    # Discrete-Time Fourier transform matrix
    W = np.exp(-jw[:, np.newaxis]*np.arange(Lfir)*T)

    # FIR coefficients
    FIR = np.linalg.inv((np.conjugate(W.T)@W).real) \
        @ (np.conjugate(W.T)@a).real
    # FIR = np.linalg.lstsq(W, a)
    return IIR, FIR, n0


def fos_real_pole(r, p, L_fir, n_center, fs, mode, window=None):
    """
    Impulse invariance method applied to a first-order section filter
    with a real pole.

    Parameters
    ----------
    r : float
        residue.
    p : float
        pole.
    L_fir : int
        FIR length.
    n_center : int
        Sample index where the IIR fitler begins (pre-delay).
    fs : int
        Sampling frequency in Hertz.
    mode : string
        {"uncorrected", "corrected", "dcmatched", "bandlimited",
         "dcmbandlimited"}.
    window : array_like, optional
        Tapering window.

    Returns
    -------
    list
        Numerator coefficients.
    list
        Denominator coefficients.
    FIR : array_like
        FIR coefficients.

    """
    T = 1/fs
    rd = r.real*T
    pd = np.exp(p.real*T)

    # First-order section
    b0 = rd
    a0 = 1.
    a1 = -pd

    # if mode in {'bandlimited', 'dcmbandlimited'}:
    #     n = np.arange(L_fir) - n_center
    #     blexres = bandlimited_decaying_exponential(r, p, n/fs, fs,
    #                                                residual=True)
    #     FIR = T * blexres.real
    #     if window is not None:
    #         FIR *= window
    #     if mode == 'dcmbandlimited':
    #         FIR[n_center] -= (r/p + rd/(1-pd)).real + np.sum(FIR)
    if mode == 'corrected':
        FIR = np.zeros(L_fir)
        FIR[n_center] = -0.5 * rd
    elif mode == 'uncorrected':
        FIR = np.zeros(L_fir)
    elif mode == 'dcmatched':
        FIR = np.zeros(L_fir)
        FIR[n_center] = -(r/p + rd/(1-pd)).real
    elif mode == 'nyquistmatched':
        FIR = np.zeros(L_fir)
        FIR[n_center] = np.abs(r/(1j*np.pi*fs-p)) - rd/(1+pd)
    else:
        FIR = np.zeros(L_fir)
    return [b0], [a0, a1], FIR


def sos_cplx_poles(r, p, L_fir, n_center, fs, mode, window=None):
    """
    Impulse invariance method applied to a second-order section filter
    with complex conjugate poles.

    Parameters
    ----------
    r : complex
        One from the complex conjugate pole. This must corresponds to the
        pole given in the second argument.
    r: array_like
        One from the complex conjugate residue. This must corresponds to the
        residue given in the first argument.
    L_fir : int
        FIR length.
    n_center : int
        Sample index where the IIR fitler begins (pre-delay).
    fs : int
        Sampling frequency in Hertz.
    mode : string
        {"uncorrected", "corrected", "dcmatched", "bandlimited",
         "dcmbandlimited"}.
    window : array_like, optional
        Tapering window.

    Returns
    -------
    list
        Numerator coefficients.
    list
        Denominator coefficients.
    FIR : array_like
        FIR coefficients.

    """
    T = 1/fs
    rd = r*T
    pd = np.exp(p*T)

    # Second-order section
    b0 = 2 * rd.real
    b1 = -2 * (rd.conj() * pd).real
    a0 = 1.
    a1 = -2 * np.exp(p.real*T) * np.cos(p.imag*T)
#    a2 = np.exp(2 * p.real * T)
    a2 = (pd * pd.conj()).real

    # if mode in {'bandlimited', 'dcmbandlimited'}:
    #     n = np.arange(L_fir) - n_center
    #     blexres = bandlimited_decaying_sinusoid(r, p, n/fs, fs,
    #                                             residual=True)
    #     FIR = T * blexres
    #     if window is not None:
    #         FIR *= window
    #     if mode == 'dcmbandlimited':
    #         FIR[n_center] -= 2 * (r/p + rd/(1-pd)).real + np.sum(FIR)
    if mode == 'corrected':
        FIR = np.zeros(L_fir)
        FIR[n_center] = - rd.real
    elif mode == 'uncorrected':
        FIR = np.zeros(L_fir)
    elif mode == 'dcmatched':
        FIR = np.zeros(L_fir)
        FIR[n_center] = -2 * (r/p + rd/(1-pd)).real
    elif mode == 'nyquistmatched':
        FIR = np.zeros(L_fir)
        FIR[n_center] = \
            (np.abs(r/(1j*np.pi*fs-p) + r.conj()/(1j*np.pi*fs-p.conj()))
             - rd/(1+pd) - rd.conj()/(1+pd.conj())).real
    return [b0, b1], [a0, a1, a2], FIR
