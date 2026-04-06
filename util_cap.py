# see https://github.com/spatialaudio/daga2026_dode

# Frank Schultz (https://github.com/fs446) is the author of the code below
# which accompanies the paper
# Frank Schultz, Nara Hahn, Sascha Spors
# Time-Domain Radiation Model of a Cap on a Rigid Sphere
# DAGA 2026 - 52. Jahrestagung für Akustik, 23.-26. März 2026, Dresden
# https://2026.daga-tagung.de/

# needs spharpy >= 1.0.0


import numpy as np
from scipy.signal import freqz, lfilter, residue, unit_impulse
from scipy.signal.windows import kaiser
from scipy.special import assoc_legendre_p, sph_harm_y
from spharpy.samplings import dodecahedron
from util_fa23 import (derivative_hn2_poly, phaseshift_sampledelay,
                       spherical_hn2, hn2_poly, log_frequency,
                       impulse_invariance,
                       impulse_invariance_least_square,)


# Useful references:
#
# [Raf19] Boaz Rafaely (2019): Fundamentals of Spherical Array Processing,
# Springer, 2nd Ed., https://doi.org/10.1007/978-3-319-99561-8
#
# [ZF19] Franz Zotter, Matthias Frank (2019): Ambisonics, A Practical
# 3D Audio Theory for Recording, Studio Production, Sound Reinforcement,
# and Virtual Reality, Springer, 1st Ed.,
# https://doi.org/10.1007/978-3-030-17207-7
#
# [MRZ15] Hai Morgenstern, Boaz Rafaely, Franz Zotter (2015): Theory and
# investigation of acoustic multiple-input multiple-output systems based
# on spherical arrays in a room, J. Acoust. Soc. Am. 138(5):2998-3009,
# https://doi.org/10.1121/1.4934555
#
# [Kle20] Johannes Christian Klein (2020): Directional Room Impulse Response
# Measurement, doctoral thesis, RWTH Aachen
#
# [Pom08] Hannes Pomberger (2008): Angular and Radial Directivity Control
# for Spherical Loudspeaker Arrays, diploma thesis, IEM/KUG Graz
#
# [HSS23] Nara Hahn, Frank Schultz, Sascha Spors (2023): Accurate time-domain
# simulation of spherical microphone arrays, Proc. Forum Acusticum,
# Torino, September 2023, 599-606
#
# [HSS22] Nara Hahn, Frank Schultz, Sascha Spors (2022): Band Limited
# Impulse Invariance Method. 30th EUSIPCO, Belgrade, pp. 209-213.
#
# [Bad2010] N. Baddour, "Operational and convolution properties of
# three-dimensional fourier transforms in spherical
# polar coordinates," J. Opt. Soc. Am. A, vol. 27, no. 10,
# pp. 2144–2155, Oct. 2010. doi: 10.1364/JOSAA.27.002144
# https://opg.optica.org/josaa/abstract.cfm?URI=josaa-27-10-2144


def figure_path():
    fig_path = ''
    return fig_path


def N_to_Nacn(N):
    """
    N: ...spherical harmonics expansion order
    Nacn...number of modes, needed for ACN for-loops
    """
    Nacn = N*(N+1)+N+1
    return Nacn


def Nacn_to_N(Nacn):
    """
    Nacn: number of modes, needed for ACN for-loops
    N...spherical harmonics expansion order
    """
    if Nacn == 1:
        N = 0
    else:
        N = np.roots([1, 2, 1 - Nacn])
        N = int(N[N > 0][0])
    return N


def vn0_northpole_cap(N, alpha_cap, beta_kaiser=0):
    """modal coefficients for spherical cap piston on north-pole

    N: spherical expansion order
    alpha_cap: half-opening angle in rad
    beta_kaiser: we add a Kaiser-Bessel window for spatial lowpass filtering
    beta_kaiser=0 yields the original weights == hard truncated cap spectrum

    we implement [Raf19] eq. (1.63)
    cf. also [ZF19] Ch. 7.3, note the sign typo in (A.61)
    cf. also [MRZ15] eq. (10)
    cf. also [Kle20] Ch. 3.3.7.
    """
    if np.mod(N, 2) != 0:
        # for convenient kaiser window handling with sym=True
        print('error: N must be even !')
        return (-1)
    window = kaiser(2*(N+1), beta_kaiser, sym=True)[N+1:]
    vn0 = np.zeros(N+1)
    for n in range(N+1):
        if n == 0:
            # we need this n==0 handling as assoc_legendre_p()
            # does not handle negative Legendre orders?!
            # in theory the equation for case 'else' is valid
            # also for the case 'n==0'
            vn0[n] = np.sqrt(np.pi) * (1-np.cos(alpha_cap))
        else:
            vn0[n] = np.sqrt(np.pi/(2*n+1)) *\
                  (+ assoc_legendre_p(n-1, 0, np.cos(alpha_cap))[0]
                   - assoc_legendre_p(n+1, 0, np.cos(alpha_cap))[0])
    return vn0 * window  # spatially lowpass-filtered velocity spectrum
    # of north-pole cap


def vnm_spherical_cap(N, alpha_cap, theta_cap, phi_cap, beta_kaiser=0):
    """modal coefficients for spherical cap piston facing to
    theta_cap, phi_cap

    N: spherical harmonic expansion order
    alpha_cap: half-opening angle in rad
    theta_cap: polar angle for normal direction of the cap in rad
    phi_cap: azimuth angle for normal direction of the cap in rad
    beta_kaiser: weight for Kaiser-Bessel window

    get vnm from vn0:
    this is multiplication in modal domain, which originates
    from the spherical convolution theorem:
    convolution of reference north-pole cap v(theta, phi)
    with bandlimited dirac(theta_cap, phi_cap)
    this rotates the cap to the desired normal direction
    we implement [Raf19] eq. (1.86)
    cf. also [ZF19] p.190ff and Ch. 7.3
    cf. [Raf19] eq. (1.58) for bandlimited Dirac spherical spectrum
    """
    vn0 = vn0_northpole_cap(N, alpha_cap, beta_kaiser=beta_kaiser)
    vnm = np.zeros(N_to_Nacn(N), dtype='complex128')
    cnt = 0
    for n in range(N+1):
        for m in range(-n, n+1):
            # check [Bad2010] for full 3D rotational convolution Eq. (82)
            # note that [Bad2010] uses the Condon-Shortly Phase exclusively
            # in her equations, but not included in the associated Legendre
            # polynomial, as we do here
            vnm[cnt] = np.sqrt(4*np.pi / (2*n+1)) * \
                np.conjugate(sph_harm_y(n, m, theta_cap, phi_cap)) * vn0[n]
            cnt += 1
            # note: the 2 pi in [Raf19] eq. (1.86) is not correct
            # cf. [Kle20] eq. (3.100) with a rather weird explanation to fix
            # for that mismatch
            # [MRZ15] is mismatched by 2 pi as well, but otherwise
            # fine with our stuff here (note that they are using
            # the Hankel 1 convention)
    return vnm


def vnm_dodecahedron(N, beta_kaiser=5, gl=np.ones(12),
                     alpha_cap=20*np.pi/180):
    """modal coefficients for a reference dodecahedron with
    caps driven by gain vector gl

    N: spherical harmonic expansion order
    beta_kaiser: weight for Kaiser-Bessel window
    gl: gain vector (linear values) for the caps
    alpha_cap: half-opening angle in rad
    """
    dode = dodecahedron()
    thetal, phil = dode.colatitude, dode.azimuth
    # get modal spectrum for dodecahedron,
    # i.e. superposition of all spherical cap pistons
    vnm = np.zeros(N_to_Nacn(N), dtype='complex128')
    for t, p, g in zip(thetal, phil, gl):
        vnm += g * vnm_spherical_cap(N, alpha_cap,
                                     t, p,
                                     beta_kaiser=beta_kaiser)
    return vnm


def surface_velocity(vnm, Theta):
    """calculate the velocity on the unit-sphere
    This is actually an iSHT of vnm

    vnm: modal coefficients
    Theta: matrix with [polar angle theta, azimuth angle phi] in rad
    """
    N = Nacn_to_N(vnm.shape[0])
    angular = np.zeros(Theta.shape[0], dtype='complex128')
    # angular expansion
    cnt = 0  # go through ACN
    for n in range(N+1):
        for m in range(-n, n+1):
            angular += vnm[cnt] * sph_harm_y(n, m,
                                             Theta[:, 0],
                                             Theta[:, 1])
            cnt += 1
    return angular


def pressure_with_hn2(vnm, f, Theta, r, R, c=343, rho0=1.2041):
    """Pressure expansion for a given velocity modal spectrum vnm.

    This models the radiation from a rigid spherical loudspeaker array
    using the radial filter Hn = -j hn2(kr) / hn2'(kR)
    and Z0 = rho0 * c mapping from velocity to pressure

    vnm:    velocity modal spectrum, array with ACN
    f:      frequency vector, Hz, array
    Theta:  angle array [polar, azimuth], dim: N angles x 2
    r:      exterior radius, m, scalar
    R:      spherical loudspeaker array's radius, m, scalar
    c:      speed of sound, m/s, scalar
    rho0:   density of air, kg/m^3, scalar
    """
    N = Nacn_to_N(vnm.shape[0])
    w = 2*np.pi*f  # rad/s
    # w_c = w / c  # rad / m
    # kr = w_c * r  # rad
    # kR = w_c * R  # rad

    angular = np.zeros((Theta.shape[0], N+1), dtype='complex128')  # phi x n
    radial_exact = np.zeros((N+1, f.shape[0]), dtype='complex128')  # n x f
    radial_far = np.zeros((N+1, f.shape[0]), dtype='complex128')  # n x f

    # angular expansion
    cnt = 0  # go through ACN
    for n in range(N+1):
        for m in range(-n, n+1):
            angular[:, n] += vnm[cnt] * sph_harm_y(n, m,
                                                   Theta[:, 0],
                                                   Theta[:, 1])
            cnt += 1
    # radial expansion
    for n in range(N+1):
        # Hden_closed = spherical_hn2(n, kR, derivative=True)
        # Hnum_exact = spherical_hn2(n, kr, derivative=False)
        # Hnum_far = (1j)**(n+1) / (kr) * np.exp(-1j*kr)  # NIST (eq. 10.52.4)
        # radial_exact[n, :] = -1j * Hnum_exact / Hden_closed
        # radial_far[n, :] = -1j * Hnum_far / Hden_closed
        radial_exact[n, :] = radial_filter_frequency_response(
            n, w, r, R, c, farfield_flag=False)
        radial_far[n, :] = radial_filter_frequency_response(
            n, w, r, R, c, farfield_flag=True)
    # sum via linear combinations and
    # weight it with air's acoustic impedance Z0 = rho0 c
    # to get the pressure in Pa
    # polarity -1j in radial filter is needed such that we get
    # positive pressure peak from positive velocity peak
    # with exp(+1i w t) time convention
    # cf. an omni-breathing sphere, i.e. the cap spans the whole sphere,
    # and only n=0, m=0 mode is needed
    p_exact_matrix = rho0 * c * angular @ radial_exact
    p_far_matrix = rho0 * c * angular @ radial_far
    return p_exact_matrix, p_far_matrix


def pressure_impulse_response_with_filterbank(
        vnm, f, Theta, r, R, L,
        c=343, rho0=1.2041, fs=32000,
        farfield_flag=False, flag_pfe_only=False,
        Lfir=21, n0=10, num_fc=27):
    """Pressure expansion for a given velocity modal spectrum vnm.

    vnm:    velocity modal spectrum, array with ACN
    f:      frequency vector, Hz, array
    Theta:  angle array [polar, azimuth], dim: N angles x 2
    r:      exterior radius, m, scalar
    R:      spherical loudspeaker array's radius, m, scalar
    L:      length of impulse response, i.e. number of time-domain samples
            please do not confuse L with the amount of caps/loudspeakers
            as used in the paper
    c:      speed of sound, m/s, scalar
    rho0:   density of air, kg/m^3, scalar
    fs:     sampling frequency, Hz, scalar
    farfield_flag:
            False=exact radial filter
            True=farfield approx for hn2(w/c r->oo)
    flag_pfe_only:
            False=complete radial filter, i.e. highpass
            True=parallel filter path only, i.e. lowpass
    Lfir:   number of FIR taps for band-limited IIM least squares FIR filter
            we always use odd integer
    n0:     time-domain sample index for FIR peak
            we always use (Lfir-1)/2
    num_fc: number of LS control frequencies
    note that fc_min and fc_max are further parameters which are hard coded
    for this function

    This models the radiation from a rigid spherical loudspeaker array
    using the IIR/FIR filterbank impulse responses for the
    radial filter Hn = -j hn2(kr) / hn2'(kR)
    and adds the Z0 = rho0 * c weighting for mapping from velocity to pressure

    """
    N = Nacn_to_N(vnm.shape[0])

    # angular expansion
    angular = np.zeros((Theta.shape[0], N+1), dtype='complex128')  # phi x n
    cnt = 0  # go through ACN
    for n in range(N+1):
        for m in range(-n, n+1):
            angular[:, n] += vnm[cnt] * sph_harm_y(n, m,
                                                   Theta[:, 0],
                                                   Theta[:, 1])
            cnt += 1
    # radial expansion
    Filters = filterbank(N, 2*np.pi*f, r, R, c=c, fs=fs,
                         farfield_flag=farfield_flag,
                         res_type='ls_residual',
                         ii_type='BLIIM',
                         Lfir=Lfir, n0=n0,
                         fc_min=fs//2**9, fc_max=fs//2**1, num_fc=num_fc)
    acoustic_delay_samples = int(np.round((r-R)/c*fs))  # TBD: get rid of int !
    # TBD: proper fractional delay handling
    # print(acoustic_delay_samples)
    h_radial = filterbank_impulse_responses(Filters, L, acoustic_delay_samples,
                                            r, R, flag_pfe_only)
    p = rho0 * c * angular @ h_radial
    return p


def radial_filter_frequency_response(
        order, w, r, R, c, farfield_flag):
    """Calculate frequency response for n-th order radial filter
    Hn = -j hn2(kr) / hn2'(kR)
    using spherical Hankel functions from scipy.special

    order:          spherical expansion order
    w:              angular frequency, in rad/s, array
    r:              radius as distance from source's origin, in m, scalar
    R:              source's radius, in m, scalar
    c:              speed of sound, in m/s, scalar
    farfield_flag:
                    False=exact radial filter
                    True=farfield approx for hn2(w/c r->oo)

    this frequency response corresponds to an (n+1) zeros / (n+1) poles system

    the filter is a highpass filter with some PEQ & shelving characteristics
    in the region of the cut-off frequency

    with the chosen conventions, the filter has positive polarity as dictated
    by the physics

    cf. calculus for DAGA 2026 Dresden, Eq. (2)
    """
    H = np.zeros_like(w, dtype='complex128')
    w_c = w / c
    kr, kR = w_c * r, w_c * R
    Hden_closed = spherical_hn2(order, kR, derivative=True)
    H = -1j * 1/Hden_closed  # -1j for correct polarity in the final
    # impulse response when using exp(+1j w t) time convention
    if farfield_flag:
        H *= (1j)**(order+1) / (kr) * np.exp(-1j*kr)  # NIST (eq. 10.52.4)
    else:
        H *= spherical_hn2(order, kr, derivative=False)
    return H


def radial_filter_min_phase_frequency_response(
        order, w, r, R, c, farfield_flag):
    """Calculate frequency response for minimum phase radial filter.

    order:          spherical expansion order
    w:              angular frequency, in rad/s, array
    r:              radius as distance from source's origin, in m, scalar
    R:              source's radius, in m, scalar
    c:              speed of sound, in m/s, scalar
    farfield_flag:
                    False=exact radial filter
                    True=farfield approx for hn2(w/c r->oo)

    get minimum phase part of the n-th order radial filter
    Hn = -j hn2(kr) / hn2'(kR)
    by calculating Hn using spherical Hankel functions from scipy.special
    and then compensating for r, R, the propagation delay
    and finally subtracting the parallel, unit-gain, pass-thru system

    this frequency response corresponds to an n zeros / n+1 poles system
    which is a proper rational function suitable for a partial fraction
    expansion (PFE), in the DAGA paper this is hat{H}_n(s,r,R)

    the filter is a min-phase lowpass filter with some PEQ & shelving
    characteristics in the region of the cut-off frequency

    the filter has negative polarity such that we have a parallel
    filter bank: 1 + polarity inverted lowpass

    cf. calculus for DAGA 2026 Dresden, Eq. (4), (5)
    """
    H = np.zeros_like(w, dtype='complex128')
    n = order
    w_c = w / c
    kr, kR = w_c * r, w_c * R
    Hden_closed = spherical_hn2(n, kR, derivative=True)
    H = (-1j * 1/Hden_closed) * r/R * np.exp(+1j*w_c*(r-R))
    if farfield_flag:
        H *= (1j)**(n+1) / (kr) * np.exp(-1j*kr)  # NIST (eq. 10.52.4)
    else:
        H *= spherical_hn2(n, kr, derivative=False)
    H += -1  # get rid of the parallel, unit-gain pass-thru system

    if w[0] == 0:  # handle exact DC
        H[0] = -1  # all filters exhibit -1 at DC
    else:  # or copy 2nd entry assuming that this frequency works
        H[0] = H[1]

    return H


def radial_filter_coeff(
        order, r, R=0.39/2, c=343, farfield_flag=True):
    """Laplace transfer function b,a-coeff for
    radial filter Hn = -j hn2(kr) / hn2'(kR)
    system has (n+1) poles, (n+1) zeros
    we explicitly added one zero in the origin
    the series expansion itself is very similar
    (besides the polarity it should be actually identical?!)
    to the series expansion in [HSS23]

    order: mode n
    r: distance from radiator's origin in m
    R: radiator's radius in m
    c: speed of sound in m/s
    farfield_flag: True...uses far-field/high-frequency approximation
    for hn2(w/c r-->oo)
    False...otherwise use the exact Hankel function

    cf. Schultz et al. DAGA 2026, left side of Eq. (4), (5)
    """
    tau_r, tau_R = r/c, R/c
    b = hn2_poly(order)[::-1] * (1/tau_r)**np.arange(order+1)
    b = np.append(b, 0)  # add the additional zero in origin
    a = derivative_hn2_poly(order)[::-1] * (1/tau_R)**np.arange(order+2)
    if farfield_flag:  # Hankels's approx for hn2(kr-->oo) yields
        b = np.zeros_like(a)  # number of poles == number of zeros
        b[0] = 1  # and all zeros are in the origin
        # this resembles a plane wave expansion
        # i.e. wave with no wavefront curvature far away from the radiator
    return b, a  # don't add potential prefactors (such as from acoustics) to b


def pfe(order, w, r, R=0.39/2, c=343,
        farfield_flag=True,
        res_type='scipy_residual'):
    """partial fraction expansion of radial filter

    order: mode n
    w: angular frequency array in rad/s
    r: distance from radiator's origin in m
    R: radiator's radius in m
    c: speed of sound in m/s
    farfield_flag:
    True...uses far-field/high-frequency approximation for hn2(w/c r-->oo)
    False...otherwise use the exact Hankel function
    res_type: ls_residual (more robust for large n) or scipy_residual
    """
    if res_type == 'ls_residual':
        pole, res, gain = pfe_rigid_sphere_ps(
            order, w, r, R, c, farfield_flag)
        return pole, res, gain
    elif res_type == 'scipy_residual':
        pole, res, gain = radial_filter_pfe(
            order, r, R, c, farfield_flag)
        return pole, res, gain
    else:
        print('unknown type_flag')


def pfe_rigid_sphere_ps(order, w, r, R=0.39/2, c=343, farfield_flag=True):
    """partial fraction expansion of radial filter
    using Nara Hahn's LS approach to find more robust residue.

    order:          spherical expansion order
    w:              angular frequency, in rad/s, array
    r:              radius as distance from source's origin, in m, scalar
    R:              source's radius, in m, scalar
    c:              speed of sound, in m/s, scalar
    farfield_flag:
                    False=exact radial filter
                    True=farfield approx for hn2(w/c r->oo)

    original code in util.py from Nara Hahn for the paper [HSS23]
    here modified for the DAGA 2026 project by calling
    radial_filter_coeff() instead of system_rigid_sphere_ps()

    TBD: hard coded fs/2 !!!

    see radial_filter_pfe()
    for an alternative, but numerically less stable implementation
    which uses scipy's residue()

    see pfe() for parameter descriptions
    """
    ba = radial_filter_coeff(
        order, r, R, c, farfield_flag)

    pole = np.roots(ba[1])
    pole = pole[np.argsort(pole.real)]
    if (order % 2) == 0:
        num_real = 1
    else:
        num_real = 0
    pole_real = pole[:num_real]
    pole_cplx = pole[num_real:]

    # TBD: get rid of hard coded fs/2:
    wmin, wmax, num_w = 1e-6, 2*np.pi*16000, 96  # arbitrarily chosen
    # larger num_x isn't necessarily better
    w = log_frequency(wmin, wmax, num_w)
    w[0] = 0
    Ht = radial_filter_min_phase_frequency_response(
        order, w, r, R, c, farfield_flag)  # target spectrum

    # least squares fit
    A = np.zeros((num_w, order+1), dtype=complex)  # response/plant matrix
    w1 = 1j*w
    w2 = -w**2
    for i, p_i in enumerate(pole_real):
        a1 = 1
        a0 = -p_i
        A[:, i] = 1/(a1*w1 + a0)
    for i, p_i in enumerate(pole_cplx[::2]):  # for complex conjugate pairs
        a2 = 1
        a1 = -2*p_i.real
        a0 = np.abs(p_i)**2
        A[:, num_real+2*i+1] = 1 / (a2*w2 + a1*w1 + a0)
        A[:, num_real+2*i] = w1 * A[:, num_real+2*i+1]
    A = np.concatenate((A.real, A.imag))
    y = np.concatenate((Ht.real, Ht.imag))
    b = np.linalg.pinv(A) @ y

    # Compute residue
    residue = np.zeros_like(pole)
    for i in range(num_real):
        residue[i] = b[i]
    for i, p_i in enumerate(pole_cplx[::2]):
        b1 = b[num_real+2*i]
        b0 = b[num_real+2*i+1]
        r_i = (b1*p_i + b0) / (2j*p_i.imag)
        rconj_i = -(b1*p_i.conj() + b0) / (2j*p_i.imag)
        residue[num_real+2*i] = r_i
        residue[num_real+2*i+1] = rconj_i
    # we are just interested in
    # the n+1 poles and their residues
    # the parallel, unit-gain, pass-thru system (Dirac) is managed
    # in modal_impulse_response(), modal_frequency_response()
    return residue, pole, np.asarray([0])


def radial_filter_pfe(
        order, r, R=0.39/2, c=343, farfield_flag=True):
    """partial fraction expansion of radial filter
    using scipy.signal.residue()

    order:          spherical expansion order
    r:              radius as distance from source's origin, in m, scalar
    R:              source's radius, in m, scalar
    c:              speed of sound, in m/s, scalar
    farfield_flag:
                    False=exact radial filter
                    True=farfield approx for hn2(w/c r->oo)

    see pfe_rigid_sphere_ps()
    for alternative more robust numerical implementation

    see pfe() for parameter descriptions
    """
    b, a = radial_filter_coeff(
        order=order, r=r, R=R, farfield_flag=farfield_flag, c=c)
    res, pole, _ = residue(b, a)
    # we are just interested in
    # the n+1 poles and their residues
    # the parallel, unit-gain, pass-thru system (Dirac) is managed
    # in modal_impulse_response(), modal_frequency_response()
    return res, pole, np.asarray([0])


def filterbank_single_n(n, w, r, R,
                        c=343,
                        fs=32000,
                        farfield_flag=False,
                        res_type='ls_residual',
                        ii_type='BLIIM',
                        Lfir=21, n0=10,
                        fc_min=32000//2**9, fc_max=32000//2**1, num_fc=4*21):
    """ get filter bank for a single mode

    n:              spherical mode of order n
    w:              angular frequency, in rad/s, array
    r:              radius as distance from source's origin, in m, scalar
    R:              source's radius, in m, scalar

    for other parameters see the respective functions that use them
    or cf. filterbank()
    """
    filters_rpk = [pfe(n, w, r, R, c, farfield_flag, res_type)]
    if ii_type == 'DCIIM':
        fbank = [impulse_invariance(*rpk, 1, 0, fs, 'dcmatched')
                 for rpk in filters_rpk]
    elif ii_type == 'BLIIM':
        f_control = log_frequency(fc_min, fc_max, num_fc, endpoint=True)
        fbank = [impulse_invariance_least_square(*rpk, Lfir, n0, f_control, fs)
                 for rpk in filters_rpk]
    else:
        print('unknown ii_type')
    return fbank


def filterbank(N, w, r, R,
               c=343,
               fs=32000,
               farfield_flag=False,
               res_type='ls_residual',
               ii_type='BLIIM',
               Lfir=21, n0=10,
               fc_min=32000//2**9, fc_max=32000//2**1, num_fc=4*21):
    """
    get filter bank for all modes up to N

    N:      spherical expansion order
    w:      angular frequency, in rad/s, array
    r:      radius as distance from source's origin, in m, scalar
    R:      source's radius, in m, scalar
    c:      speed of sound, in m/s
    fs:     sampling frequency, in Hz
    farfield_flag:
            False=exact radial filter
            True=farfield approx for hn2(w/c r->oo)
    res_type:
            ls_residual -> least squares based calc of residuals
            scipy_residual -> scipy's standard function
    ii_type:
            BLIIM -> band-limited impulse invariance method
            DCIIM -> DC matched impulse invariance method
    Lfir:   number of FIR taps for band-limited IIM least squares FIR filter
            we always use odd integer
    n0:     time-domain sample index for FIR peak
            we always use (Lfir-1)/2
    fc_min: min control frequency for BLIIM
    fc_max: max control frequency for BLIIM
    num_fc: number of log-spaced LS control frequencies

    TBD: check defaults
    """
    filters_rpk = [pfe(n, w, r, R, c, farfield_flag, res_type)
                   for n in range(N+1)]
    if ii_type == 'DCIIM':
        fbank = [impulse_invariance(*rpk, 1, 0, fs, 'dcmatched')
                 for rpk in filters_rpk]
    elif ii_type == 'BLIIM':
        f_control = log_frequency(fc_min, fc_max, num_fc, endpoint=True)
        fbank = [impulse_invariance_least_square(*rpk, Lfir, n0, f_control, fs)
                 for rpk in filters_rpk]
    else:
        print('unknown ii_type')
    return fbank


def filterbank_impulse_responses(
        filterbank, L, acoustic_delay_samples, r, R, flag_pfe_only):
    """
    get impulse responses from a filter bank

    filterbank: filter bank dict
                either from filterbank() or filterbank_single_n()
    L:          length of impulse response, i.e. number of time-domain samples
                please do not confuse L with the amount of caps/loudspeakers
                as used in the paper
    acoustic_delay_samples: integer delay that represents the sound
                            propagation duration
    r:          radius as distance from source's origin, in m, scalar
    R:          source's radius, in m, scalar
    flag_pfe_only:
                False=complete radial filter, i.e. highpass
                True=parallel filter path only, i.e. lowpass
    """
    # we make sure that n0 is constant in all filters:
    tmp = []
    for fb in filterbank:
        tmp.append(fb[2])
    if len(np.unique(np.array(tmp))) == 1:
        n0 = filterbank[0][2]

    # get a properly shifted Dirac
    # with acoustic propagation time and compensated for FIR delay
    x = unit_impulse(L, acoustic_delay_samples-n0)
    # TBD: fractional delay handling if acoustic_delay_samples is not int

    # get modal IIRs and FIR
    # the parallel, unit-gain, pass-thru system i.e. Dirac is in the FIR
    h_modal_iir, h_modal_fir = modal_impulse_response(
        filterbank, x, flag_pfe_only)

    # add parallel filters and apply radius fraction
    h_modal = R/r * (h_modal_iir + h_modal_fir)
    # h_modal = R/r * (h_modal_iir)

    return h_modal


def filterbank_frequency_responses(
        filterbank, f, fs, acoustic_delay_samples, r, R, flag_pfe_only):
    """
    get frequency responses from a filter bank

    see filterbank_impulse_responses() for parameters

    f:      frequency vector in Hz
    fs:     sampling frequency in Hz
    """
    # we make sure that n0 is constant in all filters:
    tmp = []
    for fb in filterbank:
        tmp.append(fb[2])
    if len(np.unique(np.array(tmp))) == 1:
        n0 = int(filterbank[0][2])

    # get frequency response of a properly shifted Dirac
    # with acoustic propagation time and compensated for FIR delay
    X = phaseshift_sampledelay(acoustic_delay_samples-n0, f, fs)

    # get modal IIRs and FIR
    # the parallel, unit-gain, pass-thru system i.e. Dirac is in the FIR
    H_modal_iir, H_modal_fir = modal_frequency_response(
        filterbank, X, f, fs, flag_pfe_only)

    # add parallel filters and apply radius fraction
    H_modal = R/r * (H_modal_iir + H_modal_fir)
    # H_modal = R/r * (H_modal_iir)

    return H_modal


def filterbank_via_ifft_impulse_responses(N, fs, L, r, R, c,
                                          farfield_flag, flag_pfe_only):
    """
    get impulse responses for IDFT approach

    N:      spherical expansion order
    fs:     sampling frequency in Hz
    L:      DFT block length, i.e. number of frequencies 0...fs
            number of time-domain samples, even integer!
    r:      radius as distance from source's origin, in m, scalar
    R:      source's radius, in m, scalar
    c:      speed of sound, in m/s
    farfield_flag:
            False=exact radial filter
            True=farfield approx for hn2(w/c r->oo)
    flag_pfe_only:
            False=complete radial filter, i.e. highpass
            True=parallel filter path only, i.e. lowpass
    """
    # L even!
    w_ifft = 2 * np.pi * np.arange(L) * (fs / L)
    H_ifft = np.zeros((N+1, L), dtype='complex128')
    for order in range(N+1):
        if flag_pfe_only:
            H_ifft[order, :] = radial_filter_min_phase_frequency_response(
                order, w_ifft, r, R, c, farfield_flag) * \
                    np.exp(-1j*w_ifft/c*(r-R)) * R/r
            # we apply the acoustic propagation time and radial fraction R/r
            # for consistence
        else:
            H_ifft[order, :] = radial_filter_frequency_response(
                order, w_ifft, r, R, c, farfield_flag)
        H_ifft[order, 0] = np.real(np.abs(H_ifft[order, 1]))
        H_ifft[order, L//2] = np.real(np.abs(H_ifft[order, L//2]))
        H_ifft[order, L//2+1:] = np.flip(np.conj(H_ifft[order, 1:L//2]))
    h_ifft = np.fft.ifft(H_ifft, axis=1)
    print(np.max(np.imag(h_ifft)), np.min(np.imag(h_ifft)))
    h_ifft = np.real(h_ifft)
    return h_ifft


def filterbank_via_ifft_frequency_responses(N, f, r, R, c,
                                            farfield_flag, flag_pfe_only):
    """
    get frequency responses for the IDFT approach, i.e.
    do just the frequency sampling at specified frequencies

    N:      spherical expansion order
    f:      frequency vector in Hz
    r:      radius as distance from source's origin, in m, scalar
    R:      source's radius, in m, scalar
    c:      speed of sound, in m/s
    farfield_flag:
            False=exact radial filter
            True=farfield approx for hn2(w/c r->oo)
    flag_pfe_only:
            False=complete radial filter, i.e. highpass
            True=parallel filter path only, i.e. lowpass
    """
    w_ifft = 2 * np.pi * f
    H_ifft = np.zeros((N+1, f.shape[0]), dtype='complex128')
    for order in range(N+1):
        if flag_pfe_only:
            H_ifft[order, :] = radial_filter_min_phase_frequency_response(
                order, w_ifft, r, R, c, farfield_flag) * \
                    np.exp(-1j*w_ifft/c*(r-R)) * R/r
            # we apply the acoustic propagation time and radial fraction
            # for consistence
        else:
            H_ifft[order, :] = radial_filter_frequency_response(
                order, w_ifft, r, R, c, farfield_flag)
    return H_ifft


def modal_impulse_response(Filters, x, flag_pfe_only=True):
    """Calculate impulse response from filter bank.

    Filters:    filter bank dict
                either from filterbank() or filterbank_single_n()
    x: input signal, we use shifted dirac
    flag_pfe_only:
            False=complete radial filter, i.e. highpass
            True=parallel filter path only, i.e. lowpass

    original code in fa2023-mkfig-all-results.py
    from Nara Hahn for the paper [HSS23]
    here modified to add a parallel, unit-gain, pass-thru system
    by 'if not flag_pfe_only:' used for Schultz et al. DAGA 26 project
    default case 'flag_pfe_only=True' resembles Nara's original function
    """
    L = len(x)
    h_iir = []
    h_fir = []
    for Filter in Filters:
        Lfir = len(Filter[1])
        n0 = Filter[2]
        h_iir_temp = np.zeros(L)
        h_fir_temp = np.zeros(L)
        for ba in Filter[0]:
            h_iir_temp[n0:] += lfilter(*ba, x[:L-n0])
        h_fir_temp[:Lfir+L-1] = lfilter(np.asarray(Filter[1]), 1, x)
        if not flag_pfe_only:  # apply parallel dirac system
            # Dirac needs to be time-aligend with FIRs group delay n0
            h_fir_temp += np.roll(x, n0)  # TBD: use shift instead of roll
            # for the case that x does contain a non-impulse like signal
        h_iir.append(h_iir_temp)
        h_fir.append(h_fir_temp)
    return np.asarray(h_iir), np.asarray(h_fir)


def modal_frequency_response(Filters, X, f, fs, flag_pfe_only=True):
    """Calculate frequency response from filter bank.

    Filters:    filter bank dict
                either from filterbank() or filterbank_single_n()
    X:          input spectrum, we use unit magnitude and
                with appropriate phase (pre-delay, acoustic delay)
    f:          frequency vector, in Hz
    fs:         sampling frequency, in Hz
    flag_pfe_only:
            False=complete radial filter, i.e. highpass
            True=parallel filter path only, i.e. lowpass

    original code in fa2023-mkfig-all-results.py
    from Nara Hahn for the paper [HSS23]
    here modified to add a parallel, unit-gain, pass-thru system
    by 'if not flag_pfe_only:' used for Schultz et al. DAGA 26 project
    default case 'flag_pfe_only=True' resembles Nara's original function
    """
    H_iir = []
    H_fir = []
    for Filter in Filters:
        n0 = Filter[2]
        H_iir_temp = np.zeros_like(f, dtype=complex)
        for ba in Filter[0]:
            H_iir_temp += freqz(*ba, worN=f, fs=fs)[1]
        H_iir_temp *= phaseshift_sampledelay(n0, f, fs) * X
        H_fir_temp = freqz(Filter[1], 1, worN=f, fs=fs)[1] * X
        if not flag_pfe_only:  # apply the parallel dirac system
            # Dirac needs to be time-aligend with FIRs group delay n0
            H_fir_temp += phaseshift_sampledelay(n0, f, fs) * X
        H_iir.append(H_iir_temp)
        H_fir.append(H_fir_temp)
    return np.asarray(H_iir), np.asarray(H_fir)
