"""
TRIDENT CAFE model engine.

This module ports the Oregon State CAFE C-code structure to Python for a
single pixel. It follows the same 10-step structure and input list:

PAR, chl, mld, lat, yd, aph_443, adg_443, bbp_443, bbp_s, sst

Output:
    daily vertically integrated NPP [mg C m^-2 day^-1]

Notes:
- This is intentionally written first as a transparent single-pixel reference
  implementation, not a highly optimized vectorized implementation.
- Once validated against OSU CAFE output, this can be parallelized/vectorized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


WV = np.array([
    400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
    550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700
], dtype=float)

AW = np.array([
    0.00663, 0.00473, 0.00454, 0.00495, 0.00635, 0.00922, 0.00979, 0.0106,
    0.0127, 0.015, 0.0204, 0.0325, 0.0409, 0.0434, 0.0474, 0.0565, 0.0619,
    0.0695, 0.0896, 0.1351, 0.2224, 0.2644, 0.2755, 0.2916, 0.3108, 0.34,
    0.41, 0.439, 0.465, 0.516, 0.624
], dtype=float)

A_BRICAUD = np.array([
    0.0241, 0.0287, 0.0328, 0.0359, 0.0378, 0.0350, 0.0328, 0.0309, 0.0281,
    0.0254, 0.0210, 0.0162, 0.0126, 0.0103, 0.0085, 0.0070, 0.0057, 0.0050,
    0.0051, 0.0054, 0.0052, 0.0055, 0.0061, 0.0066, 0.0071, 0.0078, 0.0108,
    0.0174, 0.0161, 0.0069, 0.0025
], dtype=float)

E_BRICAUD = np.array([
    0.6877, 0.6834, 0.6664, 0.6478, 0.6266, 0.5993, 0.5961, 0.5970, 0.5890,
    0.6074, 0.6529, 0.7212, 0.7939, 0.8500, 0.9036, 0.9312, 0.9345, 0.9298,
    0.8933, 0.8589, 0.8410, 0.8548, 0.8704, 0.8638, 0.8524, 0.8155, 0.8233,
    0.8138, 0.8284, 0.9255, 1.0286
], dtype=float)

PAR_SPECTRUM = np.array([
    0.00227, 0.00218, 0.00239, 0.00189, 0.00297, 0.00348, 0.00345, 0.00344,
    0.00373, 0.00377, 0.00362, 0.00364, 0.00360, 0.00367, 0.00354, 0.00368,
    0.00354, 0.00357, 0.00363, 0.00332, 0.00358, 0.00357, 0.00359, 0.00340,
    0.00350, 0.00332, 0.00342, 0.00347, 0.00342, 0.00290, 0.00314
], dtype=float)


@dataclass
class CafeDiagnostics:
    npp: float
    zeu: float
    kdpar: float
    eu: float
    valid: bool
    reason: str = "ok"


def betasw_zhh2009(lambda_nm: float, S: float, Tc: float) -> float:
    """Total scattering coefficient of pure seawater following Zhang et al. 2009.

    Ported from the Oregon CAFE code. Returns bsw.
    """
    Na = 6.0221417930e23
    Kbz = 1.3806503e-23
    Tk = Tc + 273.15
    M0 = 18e-3
    delta = 0.039

    n_air = 1.0 + (
        5792105.0 / (238.0185 - 1 / (lambda_nm / 1e3) ** 2)
        + 167917.0 / (57.362 - 1 / (lambda_nm / 1e3) ** 2)
    ) / 1e8

    n0 = 1.31405; n1 = 1.779e-4; n2 = -1.05e-6; n3 = 1.6e-8; n4 = -2.02e-6
    n5 = 15.868; n6 = 0.01155; n7 = -0.00423; n8 = -4382; n9 = 1.1455e6

    nsw = (
        n0 + (n1 + n2 * Tc + n3 * Tc**2) * S + n4 * Tc**2
        + (n5 + n6 * S + n7 * Tc) / lambda_nm
        + n8 / lambda_nm**2 + n9 / lambda_nm**3
    ) * n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc**2 + n6 / lambda_nm) * n_air

    kw = 19652.21 + 148.4206 * Tc - 2.327105 * Tc**2 + 1.360477e-2 * Tc**3 - 5.155288e-5 * Tc**4
    g0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc**2 - 6.167e-5 * Tc**3
    g1 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc**2
    Ks = kw + g0 * S + g1 * S**1.5
    IsoComp = (1 / Ks * 1e-5)

    a0 = 8.24493e-1; a1 = -4.0899e-3; a2 = 7.6438e-5; a3 = -8.2467e-7
    a4 = 5.3875e-9; a5 = -5.72466e-3; a6 = 1.02270e-4; a7 = -1.6546e-6; a8 = 4.8314e-4
    b0 = 999.842594; b1 = 6.793952e-2; b2 = -9.09529e-3; b3 = 1.001685e-4
    b4 = -1.120083e-6; b5 = 6.536332e-9

    density = b0 + b1 * Tc + b2 * Tc**2 + b3 * Tc**3 + b4 * Tc**4 + b5 * Tc**5
    density += ((a0 + a1 * Tc + a2 * Tc**2 + a3 * Tc**3 + a4 * Tc**4) * S +
                (a5 + a6 * Tc + a7 * Tc**2) * S**1.5 + a8 * S**2)

    dlnawds = (
        -5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc**2 + 2.40808e-11 * Tc**3
        + 1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * Tc**2 - 1.39872e-11 * Tc**3) * S**0.5
        + 2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * Tc**2) * S
    )

    n_wat2 = nsw**2
    base = (nsw / 3 - (1.0 / 3.0) / nsw)
    DFRI = (n_wat2 - 1) * (1 + (2.0 / 3.0) * (n_wat2 + 2) * base**2)

    beta_df = (math.pi**2 / 2 * (lambda_nm * 1e-9) ** -4 * Kbz * Tk * IsoComp * DFRI**2
               * (6 + 6 * delta) / (6 - 7 * delta))
    flu_con = S * M0 * dnswds**2 / density / (-1 * dlnawds) / Na
    beta_cf = (2 * math.pi**2 * (lambda_nm * 1e-9) ** -4 * nsw**2 * flu_con
               * (6 + 6 * delta) / (6 - 7 * delta))
    beta90sw = beta_df + beta_cf
    bsw = 8 * math.pi / 3 * beta90sw * (2 + delta) / (1 + delta)
    return bsw


def _is_valid_inputs(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def opp_cafe_pixel(
    PAR: float,
    chl: float,
    mld: float,
    lat: float,
    yd: int,
    aph_443: float,
    adg_443: float,
    bbp_443: float,
    bbp_s: float,
    sst: float,
    return_diagnostics: bool = False,
) -> float | CafeDiagnostics:
    """Compute CAFE daily NPP for one pixel.

    The implementation preserves the Oregon code's core equations and integration
    structure. Invalid or nonphysical inputs return NaN.
    """
    vals = (PAR, chl, mld, lat, aph_443, adg_443, bbp_443, bbp_s, sst)
    if not _is_valid_inputs(*vals):
        d = CafeDiagnostics(np.nan, np.nan, np.nan, np.nan, False, "non-finite input")
        return d if return_diagnostics else np.nan
    if PAR <= 0 or chl <= 0 or mld < 0 or aph_443 < 0 or adg_443 < 0 or bbp_443 < 0 or bbp_s < 0:
        d = CafeDiagnostics(np.nan, np.nan, np.nan, np.nan, False, "nonphysical input")
        return d if return_diagnostics else np.nan

    # Step 2: derive IOPs at 10 nm increments from 400 to 700 nm.
    bbw = np.array([betasw_zhh2009(wv, 32.5, sst) / 2 for wv in WV], dtype=float)
    denom = 0.03711 * chl ** 0.61479
    if denom <= 0:
        d = CafeDiagnostics(np.nan, np.nan, np.nan, np.nan, False, "bad chl denominator")
        return d if return_diagnostics else np.nan

    aphi = aph_443 * A_BRICAUD * chl ** E_BRICAUD / denom
    a = AW + aphi + adg_443 * np.exp(-0.018 * (WV - 443.0))
    bb = bbw + bbp_443 * (443.0 / WV) ** bbp_s

    if np.any(a <= 0) or np.any(bb < 0):
        d = CafeDiagnostics(np.nan, np.nan, np.nan, np.nan, False, "invalid spectral optics")
        return d if return_diagnostics else np.nan

    # Step 3: absorbed energy.
    absorbed_photons = 0.0
    for w in range(30):
        absorbed_photons += 0.5 * 10 * (
            PAR_SPECTRUM[w + 1] * aphi[w + 1] / a[w + 1]
            + PAR_SPECTRUM[w] * aphi[w] / a[w]
        )
    absorbed_photons *= PAR * 0.95

    # Step 4: Kd, KdPAR, Zeu.
    decl = 23.5 * math.cos(2 * math.pi * (yd - 172) / 365.0) * math.pi / 180.0
    dl_arg = -1 * math.tan(lat * math.pi / 180.0) * math.tan(decl)
    dl_arg = max(min(dl_arg, 1.0), -1.0)
    DL = math.acos(dl_arg) / math.pi
    if DL <= 0:
        d = CafeDiagnostics(np.nan, np.nan, np.nan, np.nan, False, "zero daylength")
        return d if return_diagnostics else np.nan

    solzen = 90.0 - math.asin(
        math.sin(lat * math.pi / 180.0) * math.sin(decl)
        - math.cos(lat * math.pi / 180.0) * math.cos(decl) * math.cos(math.pi)
    ) * 180.0 / math.pi
    m0 = math.sqrt((1 + 0.005 * solzen) * (1 + 0.005 * solzen))
    kd = m0 * a + 4.18 * (1 - 0.52 * np.exp(-10.8 * a)) * bb
    if kd[9] <= 0:
        d = CafeDiagnostics(np.nan, np.nan, np.nan, np.nan, False, "invalid kd490")
        return d if return_diagnostics else np.nan
    kdpar = 0.0665 + (0.874 * kd[9]) - (0.00121 / kd[9])
    if kdpar <= 0:
        d = CafeDiagnostics(np.nan, np.nan, kdpar, np.nan, False, "invalid kdpar")
        return d if return_diagnostics else np.nan
    zeu = -1.0 * math.log(0.1 / (PAR * 0.95)) / kdpar
    if not np.isfinite(zeu) or zeu <= 0:
        d = CafeDiagnostics(np.nan, zeu, kdpar, np.nan, False, "invalid zeu")
        return d if return_diagnostics else np.nan

    # Step 5: scalar irradiance conversion.
    tseq = np.arange(51, dtype=float) / 50.0
    zseq = np.arange(101, dtype=float) / 100.0 * math.ceil(zeu)
    delz = zseq[1] - zseq[0]
    PAR_noon = math.pi / 2.0 * PAR * 0.95 * PAR_SPECTRUM

    E_tzw = np.zeros((51, 101, 31), dtype=float)
    AP_tzw = np.zeros((51, 101, 31), dtype=float)
    for t in range(51):
        sin_term = math.sin(math.pi * tseq[t])
        for z in range(101):
            E_tzw[t, z, :] = PAR_noon * sin_term * np.exp(-1 * kd * zseq[z])
            AP_tzw[t, z, :] = E_tzw[t, z, :] * aphi

    E_tz = np.zeros((51, 101), dtype=float)
    AP_tz = np.zeros((51, 101), dtype=float)
    for z in range(101):
        for t in range(51):
            E_tz[t, z] = np.trapezoid(E_tzw[t, z, :], WV)
            AP_tz[t, z] = np.trapezoid(AP_tzw[t, z, :], WV)

    AP_z = np.zeros(101, dtype=float)
    for z in range(101):
        for t in range(50):
            AP_z[z] += 0.02 * (AP_tz[t + 1, z] + AP_tz[t, z]) / 2.0

    AP = 0.0
    for z in range(100):
        AP += delz * (AP_z[z] + AP_z[z + 1]) / 2.0
    if AP <= 0:
        d = CafeDiagnostics(np.nan, zeu, kdpar, np.nan, False, "zero absorbed photons")
        return d if return_diagnostics else np.nan
    Eu = absorbed_photons / AP
    E_tz *= Eu
    AP_tz *= Eu

    # Step 6: Ek through depth.
    IML = (PAR * 0.95 / (DL * 24.0)) * math.exp(-0.5 * kdpar * mld)
    Ek = np.full(101, 19 * math.exp(0.038 * (PAR * 0.95 / (DL * 24.0)) ** 0.45 / kdpar), dtype=float)
    Ek[Ek < 10] = 10

    if mld < zeu:
        Eg_mld = (PAR / DL) * math.exp(-1 * kdpar * mld)
        scale = (1 + math.exp(-0.15 * (0.95 * PAR / (DL * 24.0)))) / (1 + math.exp(-3 * IML))
        for z in range(101):
            Ek[z] *= scale
            if zseq[z] > mld:
                Eg = (PAR / DL) * math.exp(-1 * kdpar * zseq[z])
                denom2 = Eg_mld - 0.1
                if abs(denom2) > 1e-12:
                    Ek[z] = 10 + (Ek[z] - 10) / denom2 * (Eg - 0.1)
            if Ek[z] < 10:
                Ek[z] = 10
    Ek *= 0.0864

    # Step 7: KPUR.
    mean_aphi = float(np.mean(aphi))
    if mean_aphi <= 0:
        d = CafeDiagnostics(np.nan, zeu, kdpar, Eu, False, "zero mean aphi")
        return d if return_diagnostics else np.nan
    KPUR = np.zeros(101, dtype=float)
    for z in range(101):
        numerator = np.trapezoid(AP_tzw[24, z, :], WV)
        denominator = np.trapezoid(E_tzw[24, z, :], WV)
        if denominator <= 0 or numerator <= 0:
            KPUR[z] = np.nan
        else:
            KPUR[z] = Ek[z] / (numerator / (denominator * mean_aphi) / 1.3)

    # Step 8: phimax.
    phirange = np.array([0.018, 0.030], dtype=float)
    Ekrange = np.array([150 * 0.086400, 10 * 0.0864], dtype=float)
    slope = (phirange[1] - phirange[0]) / (Ekrange[1] - Ekrange[0])
    phimax = phirange[1] + (Ek - Ekrange[1]) * slope
    phimax = np.clip(phimax, phirange[0], phirange[1])

    # Step 9: aphi scaling beneath MLD.
    AP_tz2 = np.zeros_like(AP_tz)
    if mld < zeu:
        aphi_fact = np.ones(101, dtype=float)
        for z in range(101):
            if zseq[z] > mld and Ek[z] > 0:
                aphi_fact[z] = 1 + Ek[0] / Ek[z] * 0.15
        for t in range(51):
            AP_tzw[t, 0, :] = E_tzw[t, 0, :] * aphi
            for z in range(1, 101):
                a_mod = AW + aphi * aphi_fact[z] + adg_443 * np.exp(-0.018 * (WV - 443.0))
                kd_mod = m0 * a_mod + 4.18 * (1 - 0.52 * np.exp(-10.8 * a_mod)) * bb
                E_tzw[t, z, :] = E_tzw[t, z - 1, :] * np.exp(-1 * kd_mod * delz)
                AP_tzw[t, z, :] = E_tzw[t, z, :] * aphi * aphi_fact[z]
        for z in range(101):
            for t in range(51):
                AP_tz2[t, z] = np.trapezoid(AP_tzw[t, z, :], WV) * Eu
    else:
        AP_tz2[:] = AP_tz

    # Step 10: NPP.
    NPP_tz = np.zeros((51, 101), dtype=float)
    for t in range(51):
        for z in range(101):
            if E_tz[t, z] > 0 and np.isfinite(KPUR[z]):
                NPP_tz[t, z] = phimax[z] * AP_tz2[t, z] * math.tanh(KPUR[z] / E_tz[t, z]) * 12000.0
            else:
                NPP_tz[t, z] = 0.0

    NPP_z = np.zeros(101, dtype=float)
    for z in range(101):
        for t in range(50):
            NPP_z[z] += 0.02 * (NPP_tz[t + 1, z] + NPP_tz[t, z]) / 2.0

    NPP = 0.0
    for z in range(100):
        NPP += delz * (NPP_z[z] + NPP_z[z + 1]) / 2.0

    if not np.isfinite(NPP) or NPP < 0:
        NPP = np.nan
    diag = CafeDiagnostics(float(NPP), float(zeu), float(kdpar), float(Eu), np.isfinite(NPP), "ok")
    return diag if return_diagnostics else diag.npp

@dataclass
class CafeResult:
    npp: float
    zeu: float
    kdpar: float


def cafe_pixel(par, chl, mld, lat, yd, aph443, adg443, bbp443, bbp_s, sst):
    """Compatibility wrapper used by the TRIDENT app/workflows.

    Returns a compact CafeResult while using the full validated CAFE pixel
    implementation above. This replaces the older lightweight approximation
    that caused seasonal amplitudes to diverge from the OSU archived product.
    """
    res = opp_cafe_pixel(par, chl, mld, lat, yd, aph443, adg443, bbp443, bbp_s, sst, return_diagnostics=True)
    return CafeResult(float(res.npp), float(res.zeu), float(res.kdpar))
