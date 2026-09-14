#!/usr/bin/env python3
"""
run_pipeline.py
================

Production HPC pipeline for HOD-based mock galaxy catalogue generation
and finite-difference wp(rp) derivatives (Smith et al. 2024 BGS HOD,
F_spline central occupation).

This is a direct, non-scientific-changing port of the validated
development notebook. All HOD math, F_spline, NFW satellite placement,
Corrfunc wp calculation, and the derivative/convergence/multi-seed
averaging logic are reproduced verbatim from the notebook. Only the
execution shell (config, logging, CLI, file I/O, orchestration) is new.

Usage
-----
    python run_pipeline.py --seed 10 --mr -18 --realizations 15

Submit via PBS:
    qsub submit_hod_derivatives.pbs
"""

import sys
import os
import gc
import time
import logging
import argparse
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf

from Corrfunc.theory.wp import wp as corrfunc_wp


# ============================================================
# SECTION 1 — CONFIG (all magic numbers live here)
# ============================================================

class Config:
    """Container for every configurable quantity in the pipeline."""

    # ---- Run mode ----
    LOCAL_RUN = False  # toggle for local testing vs Pegasus/Sahyadri

    # ---- Paths (Pegasus / Sahyadri defaults; overridden if LOCAL_RUN) ----
    if LOCAL_RUN:
        POSTPROCESS_PATH = Path("/path/to/sahyadri-sandbox/scripts/post-process")
        HALO_CAT_M200B = Path("/path/to/mock_v2_M200b_corrected.fits")
    else:
        POSTPROCESS_PATH = Path(
            "/mnt/home/project/cgowari.aditya/sahyadri-codes/"
            "sahyadri-sandbox/scripts/post-process"
        )
        HALO_CAT_M200B = Path(
            "/mnt/home/project/cgowari.aditya/sahyadri-codes/"
            "mock_v2_M200b_corrected.fits"
        )

    # Pegasus Paths monkey-patch values — verbatim from run_pipeline.py L699-710
    PEGASUS_HOME_PATH = "/mnt/home/project/cgowari.aditya/sahyadri-codes/sahyadri-sandbox/"
    PEGASUS_SCRATCH_PATH = "/data/project/hpc2502016/data/"

    # ---- Science parameters (CLI-overridable defaults) ----
    MR_THRESH = -18.0     # Luminosity threshold (r-band absolute magnitude)
    SEED = 10             # Global RNG seed — shared by baseline + all perturbation pairs
    PIMAX = 40.0           # Line-of-sight integration limit for wp [h^-1 Mpc]
    N_RP = 50              # Number of rp bins
    RP_MIN = 0.5            # Minimum rp [h^-1 Mpc]
    RP_MAX = 30.0           # Maximum rp [h^-1 Mpc]
    NTHREADS = 4            # Corrfunc threads
    MASSDEF = 'm200b'       # Halo mass definition used throughout
    N_REALIZATIONS = 15     # number of random seeds to average over (noisy params)

    # ---- Sahyadri simulation identifiers (verbatim from run_pipeline.py) ----
    SIM_STEM = 'sahyadri/default2048'
    REAL = 1
    SNAP = 82

    # ---- Validated simulation header values — used in header assert checks ----
    EXPECTED_N_HALOS = 343_275
    EXPECTED_LBOX = 200.0          # h^-1 Mpc
    EXPECTED_OMEGA_M = 0.3137721

    # ---- Derivative engine ----
    HOD_PARAM_NAMES = ['Mcut', 'M1', 'M0', 'sigma', 'alpha']
    DELTAS = [0.050, 0.025]
    DELTA_STABLE = 0.025
    CONVERGENCE_TOL = 0.15
    STABILIZATION_WINDOW = 5
    STABILIZATION_TOL = 0.05
    STABILIZATION_BIN_IDX = 5

    # ---- Physical constants (verbatim from Mocker.__init__) ----
    RHOC = 2.7754e11   # (Msun/h) / (Mpc/h)^3 — critical density at z=0

    # ---- HOD calibration coefficients — Smith et al. (2024) Table 1,
    #      AbacusSummit c000/ph000. Verbatim from HODFits_BGS.__init__.
    #      These are physics constants, NOT tunable config — never change.
    A_min, B_min, C_min, D_min = -0.1163295942035853792, -0.5759066407274642252, 0.1437085418067726994, -0.01908335449679073731
    A_sig, B_sig, C_sig, D_sig = 0.01832666435246161490, 0.7584865035087475782, 1.103103765754579246, 0.3277787214124110449
    A_0, B_0 = -0.5468895455660652827, -1.673562380257160864
    A_1, B_1, C_1, D_1 = 1.121497360473139082, -0.4725625165007949491, 0.09200038944813740405, -0.01141762462160326314
    A_al, B_al, C_al = 1.111866664500087198, 4.473949866395862784, -4.353309870091147893

    # ---- Reference check values (Ch 4 point check at Mr=-20.5) ----
    REF_MR = -20.5
    REF_EXPECTED = dict(Mcut=1.62e12, M1=2.41e13, M0=1.95e11, sigma=0.42, alpha=1.11)

    # ---- Output directories ----
    OUTPUT_DIR = Path('outputs')
    CATALOGUE_DIR = OUTPUT_DIR / 'catalogues'
    WP_DIR = OUTPUT_DIR / 'wp'
    DERIV_DIR = OUTPUT_DIR / 'derivatives'
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    LOG_DIR = Path('logs')

    # ---- matplotlib style ----
    MPL_RCPARAMS = {
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 120,
    }

    @classmethod
    def apply_cli_overrides(cls, args: argparse.Namespace) -> None:
        """Apply parsed CLI arguments on top of the class defaults."""
        cls.SEED = args.seed
        cls.MR_THRESH = args.mr
        cls.N_REALIZATIONS = args.realizations
        cls.PIMAX = args.pimax
        cls.NTHREADS = args.nthreads


def build_arg_parser() -> argparse.ArgumentParser:
    """Command-line interface. Defaults match the notebook exactly."""
    p = argparse.ArgumentParser(
        description="HOD derivative pipeline (Smith et al. 2024 BGS, F_spline)."
    )
    p.add_argument('--seed', type=int, default=Config.SEED,
                    help='Global RNG seed (default: %(default)s)')
    p.add_argument('--mr', type=float, default=Config.MR_THRESH,
                    help='r-band luminosity threshold Mr (default: %(default)s)')
    p.add_argument('--realizations', type=int, default=Config.N_REALIZATIONS,
                    help='Number of seeds for multi-seed averaging (default: %(default)s)')
    p.add_argument('--pimax', type=float, default=Config.PIMAX,
                    help='Line-of-sight pi_max for wp [h^-1 Mpc] (default: %(default)s)')
    p.add_argument('--nthreads', type=int, default=Config.NTHREADS,
                    help='Corrfunc thread count (default: %(default)s)')
    return p


# ============================================================
# SECTION 2 — UTILITIES (no scientific calculations)
# ============================================================

def make_directories() -> None:
    """Create every output/log directory needed by the pipeline."""
    for d in (Config.OUTPUT_DIR, Config.CATALOGUE_DIR, Config.WP_DIR,
              Config.DERIV_DIR, Config.FIGURE_DIR, Config.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logger() -> logging.Logger:
    """Configure a logger writing to logs/run.log and stdout."""
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('hod_pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(Config.LOG_DIR / 'run.log', mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


@contextmanager
def timer(logger: logging.Logger, stage_name: str):
    """Context manager that logs the wall-clock runtime of a pipeline stage."""
    t0 = time.time()
    logger.info(f"--- START stage: {stage_name} ---")
    yield
    dt = time.time() - t0
    logger.info(f"--- END stage: {stage_name}  (runtime: {dt:.2f} s) ---")


def save_catalogue(mock: dict, path: Path, logger: logging.Logger) -> None:
    """
    Save a generate_mock() output dict to a compressed .npz file and
    verify the reload is bit-identical.

    np.savez_compressed is used (not pickle) because all fields are
    plain numpy arrays / scalars — npz is portable across numpy
    versions and avoids pickle's arbitrary-code-execution risk on
    shared HPC filesystems.
    """
    np.savez_compressed(
        path,
        x=mock['x'], y=mock['y'], z=mock['z'],
        halo_idx=mock['halo_idx'], is_central=mock['is_central'],
        ncen_total=np.int64(mock['ncen_total']),
        nsat_total=np.int64(mock['nsat_total']),
    )
    logger.info(f"Saved catalogue -> {path}  ({path.stat().st_size / 1e6:.2f} MB)")

    loaded = np.load(path, allow_pickle=False)
    for field in ['x', 'y', 'z', 'halo_idx', 'is_central']:
        assert np.array_equal(mock[field], loaded[field]), \
            f"Catalogue reload mismatch in field '{field}'"
    assert int(loaded['ncen_total']) == mock['ncen_total'], "ncen_total mismatch on reload"
    assert int(loaded['nsat_total']) == mock['nsat_total'], "nsat_total mismatch on reload"
    logger.info("Catalogue reload verified bit-identical.")


def load_catalogue(path: Path) -> dict:
    """Load a catalogue saved by save_catalogue() back into a mock-style dict."""
    d = np.load(path, allow_pickle=False)
    return {
        'x': d['x'], 'y': d['y'], 'z': d['z'],
        'halo_idx': d['halo_idx'], 'is_central': d['is_central'],
        'ncen_total': int(d['ncen_total']), 'nsat_total': int(d['nsat_total']),
    }


def save_wp(rp: np.ndarray, wp: np.ndarray, path: Path, logger: logging.Logger,
            **extra) -> None:
    """Save an rp/wp pair (plus optional extra arrays) to a compressed npz."""
    np.savez_compressed(path, rp=rp, wp=wp, **extra)
    logger.info(f"Saved wp -> {path}")


def save_derivatives(save_dict: dict, path: Path, logger: logging.Logger) -> None:
    """Save the final derivative dictionary to a compressed npz."""
    np.savez(path, **save_dict)
    logger.info(f"Saved {len(save_dict)} arrays -> {path}")


def savefig(fig, path: Path, logger: logging.Logger) -> None:
    """Save a matplotlib figure and close it (headless-safe)."""
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure -> {path}")


# ============================================================
# SECTION 3 — SCIENCE (verbatim from the validated notebook)
#             No algorithm, formula, or RNG call below this
#             point has been modified from the notebook.
# ============================================================

def _M_function(Mr, A, B, C, D):
    """Cubic polynomial in x = Mr + 20.  Returns log10(mass)."""
    x = Mr + 20
    return (A + 12) + B * x + C * x ** 2 + D * x ** 3


def _M0_function(Mr, A, B):
    """Linear in x = Mr + 20.  Returns log10(M0)."""
    x = Mr + 20
    return (A + 11) + B * x


def _sigma_function(Mr, A, B, C, D):
    """Sigmoid in x = Mr + 20.  Returns sigma (dimensionless)."""
    x = Mr + 20
    return A + (B - A) / (1.0 + np.exp(C * (x + D)))


def _alpha_function(Mr, A, B, C):
    """Power-law in x = Mr + 20.  Returns alpha (dimensionless)."""
    x = Mr + 20
    return A + B ** (-x + C)


def get_hod_params(Mr):
    """
    Return Smith et al. (2024) DESI BGS HOD parameters for luminosity
    threshold Mr (scalar or array).

    Parameters
    ----------
    Mr : float or array-like
        r-band absolute magnitude threshold (e.g. -20.0).
        Valid calibration range: -22 <= Mr <= -18.

    Returns
    -------
    dict with keys: Mcut, M1, M0, sigma, alpha, kappa
        Masses in h^-1 Msun.  sigma, alpha, kappa dimensionless.
    """
    C = Config
    Mcut = 10 ** _M_function(Mr, C.A_min, C.B_min, C.C_min, C.D_min)
    M1 = 10 ** _M_function(Mr, C.A_1, C.B_1, C.C_1, C.D_1)
    M0 = 10 ** _M0_function(Mr, C.A_0, C.B_0)
    sigma = _sigma_function(Mr, C.A_sig, C.B_sig, C.C_sig, C.D_sig)
    alpha = _alpha_function(Mr, C.A_al, C.B_al, C.C_al)
    kappa = M0 / Mcut
    return dict(Mcut=Mcut, M1=M1, M0=M0, sigma=sigma, alpha=alpha, kappa=kappa)


def F_spline(x):
    """
    Analytic CDF of the spline kernel — Smith et al. (2024) arXiv:2312.08792v2.

    Replaces erf() in the BGS central occupation:
        <Ncen>(M) = 0.5 * (1 + F_spline(x))
        x = (log10 M - log10 Mcut) / sigma

    Piecewise in t = x / sqrt(6):
        |t| <= 0.5 : cubic piece  (inner region)
        |t| <= 1.0 : quartic piece (outer region)
        |t| >  1   : +-1  (saturated — compact support)
    """
    x = np.asarray(x, dtype=float)
    t = x / np.sqrt(6)
    at = np.abs(t)

    piece1 = np.sign(t) * ((8 / 3) * at - (16 / 3) * at ** 3 + 4 * at ** 4)
    piece2 = np.sign(t) * (1 - (4 / 3) * (1 - at) ** 4)
    piece3 = np.sign(t) * 1.0

    return np.where(at <= 0.5, piece1,
           np.where(at <= 1.0, piece2, piece3))


def fcen_thresh(lgm, hod_params):
    """
    BGS central occupation: cumulative probability P(Lcen > L | M).

    <Ncen>(M) = 0.5 * (1 + F_spline(x))
    x = (log10 M - log10 Mcut) / sigma
    """
    x = (np.asarray(lgm) - np.log10(hod_params['Mcut'])) / hod_params['sigma']
    return 0.5 * (1.0 + F_spline(x))


def Nsat_thresh(lgm, hod_params):
    """
    Mean satellite count <Nsat>(M > L_thresh).

    <Nsat>(M) = ((M - M0) / M1)^alpha   for M > M0
                0                        otherwise

    Note: the ncen-gating factor is applied in generate_mock(), not here,
    so this function returns the *unconditional* satellite mean.
    """
    mhalo = 10 ** np.asarray(lgm, dtype=float)
    Dm = (mhalo - hod_params['M0']) / hod_params['M1']
    Dm = np.where(Dm < 0.0, 0.0, Dm)
    return Dm ** hod_params['alpha']


def EHub(z, Om):
    """Dimensionless Hubble parameter E(z) = H(z)/H0 for flat LCDM."""
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def compute_rvir_con(mass, rs_kpc, z, Om):
    """
    Compute virial radius and NFW concentration for each halo.

    The virial radius is defined via the M200b density threshold:
        M = (4*pi/3) * Delta_vir * rho_c(z) * R_vir^3
    where Delta_vir = 200 * Omega_m (background density convention).

    Parameters
    ----------
    mass   : ndarray  m200b [h^-1 Msun]
    rs_kpc : ndarray  NFW scale radius [comoving kpc/h]  ('rs' column)
    z      : float    redshift
    Om     : float    Omega_matter

    Returns
    -------
    rvir : ndarray  [comoving Mpc/h]
    con  : ndarray  concentration c = Rvir / rs  [dimensionless]
    """
    dvir = 200.0 * Om
    rhoc_z = Config.RHOC * EHub(z, Om) ** 2
    rvir_phys = (3.0 * mass / (4.0 * np.pi * dvir * rhoc_z)) ** (1.0 / 3.0)
    rvir_com = rvir_phys * (1.0 + z)   # physical -> comoving
    rs_com = rs_kpc * 1e-3              # kpc/h -> Mpc/h
    con = rvir_com / rs_com
    return rvir_com, con


def Mencl_nfw(x):
    """
    NFW enclosed mass (dimensionless):
        M(<r) / M_total = [ln(1+x) - x/(1+x)] / [ln(1+c) - c/(1+c)]

    Here we return the unnormalised numerator; the caller divides by
    Mencl_nfw(c). x = r / r_s
    """
    return np.log(1.0 + x) - x / (1.0 + x)


def gen_rsamp(Nsat, cvir, Rvir, rng):
    """
    Sample Nsat radial distances [Mpc/h] from the NFW CDF via
    inverse-transform sampling on a fine grid.
    """
    xmax = 2.0 * cvir
    xfine = np.linspace(0.0, xmax, 100_000)
    Px = Mencl_nfw(xfine)

    # Taylor series near x=0 for numerical stability (avoids 0/0)
    ind_small = xfine < 1e-3
    if ind_small.any():
        xs = xfine[ind_small]
        Px[ind_small] = (xs ** 2 / 2.0 - 2 * xs ** 3 / 3.0
                         + 3 * xs ** 4 / 4.0 - 4 * xs ** 5 / 5.0)

    Px /= Mencl_nfw(cvir)          # normalise to [0, 1]
    rs = Rvir / cvir                # scale radius [Mpc/h]
    rsamp = np.interp(rng.rand(Nsat), Px, xfine) * rs
    return rsamp


def gen_NFW_profile(Nsat, cvir, Rvir, rng):
    """
    Generate Nsat 3-d satellite offsets [Mpc/h] from halo centre,
    distributed isotropically following the NFW profile.
    """
    phi = 2.0 * np.pi * rng.rand(Nsat)
    cos_theta = 2.0 * rng.rand(Nsat) - 1.0
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))
    r = gen_rsamp(Nsat, cvir, Rvir, rng)

    dx = r * sin_theta * np.sin(phi)
    dy = r * sin_theta * np.cos(phi)
    dz = r * cos_theta
    return np.column_stack([dx, dy, dz])


def draw_centrals(lgm, hod_params, rng):
    """
    Draw central galaxy boolean flags via Bernoulli sampling.
    Uses the BGS F_spline occupation via fcen_thresh().

    Returns
    -------
    ncen     : bool ndarray (N_halos,)   True = halo has a central galaxy
    mean_cen : float ndarray (N_halos,)  <Ncen>(M) — the occupation probability
    """
    mean_cen = fcen_thresh(lgm, hod_params)          # BGS: F_spline, NOT erf
    u = rng.rand(lgm.size)
    ncen = mean_cen >= u                               # Bernoulli draw
    return ncen, mean_cen


def draw_satellites(lgm, hod_params, rng, ncen):
    """
    Draw satellite galaxy counts via ncen-gated Poisson sampling.

    The Poisson mean is:
        lambda(M) = Ncen(M) * Nsat_thresh(M)
    """
    mean_sat_raw = Nsat_thresh(lgm, hod_params)   # unconditional mean
    mean_sat = ncen.astype(float) * mean_sat_raw   # ncen gate
    nsat = rng.poisson(mean_sat)                     # Poisson sample
    return nsat, mean_sat


def generate_mock(halos, params, lbox, massdef, seed, logger=None, verbose=True):
    """
    Generate a threshold mock galaxy catalogue.

    Design: one RandomState, created fresh from `seed`, is passed
    sequentially into draw_centrals -> draw_satellites -> gen_NFW_profile.
    Changing `params` changes HOD-driven selection but keeps the
    underlying random draws in the same stream — this is the
    seed-matching strategy for finite-difference derivatives.

    Returns
    -------
    dict with keys: x, y, z, is_central, halo_idx, ncen_total, nsat_total
    """
    rng = np.random.RandomState(seed=seed)

    lgm = np.log10(halos[massdef])
    rvir, con = compute_rvir_con(halos[massdef], halos['rs'],
                                  z=generate_mock.REDSHIFT, Om=generate_mock.OMEGA_M)

    # -- Step 1: central occupation (BGS spline via draw_centrals) --
    ncen, _ = draw_centrals(lgm, params, rng)

    # -- Step 2: satellite counts (Poisson, ncen-gated) --
    nsat, _ = draw_satellites(lgm, params, rng, ncen)

    # -- Step 3: central positions = halo centres --
    idx_cen = np.where(ncen)[0]
    cen_x = halos['x'][idx_cen].copy()
    cen_y = halos['y'][idx_cen].copy()
    cen_z = halos['z'][idx_cen].copy()

    # -- Step 4: satellite positions from NFW profile --
    nsat_total = int(nsat.sum())
    sat_x = np.empty(nsat_total)
    sat_y = np.empty(nsat_total)
    sat_z = np.empty(nsat_total)
    sat_hidx = np.empty(nsat_total, dtype=int)

    ptr = 0
    indsel = np.where(nsat > 0)[0]
    for i in indsel:
        ns = int(nsat[i])
        pos = gen_NFW_profile(ns, con[i], rvir[i], rng)
        sat_x[ptr:ptr + ns] = (pos[:, 0] + halos['x'][i]) % lbox
        sat_y[ptr:ptr + ns] = (pos[:, 1] + halos['y'][i]) % lbox
        sat_z[ptr:ptr + ns] = (pos[:, 2] + halos['z'][i]) % lbox
        sat_hidx[ptr:ptr + ns] = i
        ptr += ns

    # -- Step 5: stack centrals + satellites --
    all_x = np.concatenate([cen_x, sat_x])
    all_y = np.concatenate([cen_y, sat_y])
    all_z = np.concatenate([cen_z, sat_z])

    ncen_total = idx_cen.size
    is_central = np.zeros(ncen_total + nsat_total, dtype=bool)
    is_central[:ncen_total] = True
    halo_idx_all = np.concatenate([idx_cen, sat_hidx])

    if verbose and logger is not None:
        ntot = ncen_total + nsat_total
        fsat = nsat_total / ntot if ntot > 0 else 0
        logger.info(f"Mock generated (seed={seed}): "
                    f"centrals={ncen_total:,}  satellites={nsat_total:,}  "
                    f"total={ntot:,}  f_sat={fsat:.4f}")

    return {
        'x': all_x, 'y': all_y, 'z': all_z,
        'is_central': is_central, 'halo_idx': halo_idx_all,
        'ncen_total': ncen_total, 'nsat_total': nsat_total,
    }


def compute_wp(mock, lbox, pimax, rp_bins, nthreads):
    """
    Compute the projected correlation function w_p(r_p) for a mock catalogue.

    Returns
    -------
    rp : ndarray (N_RP,)  bin-averaged r_p [h^-1 Mpc]
    wp : ndarray (N_RP,)  projected correlation function [h^-1 Mpc]
    """
    x = mock['x'].astype(np.float64)
    y = mock['y'].astype(np.float64)
    z = mock['z'].astype(np.float64)

    results = corrfunc_wp(
        lbox, pimax, nthreads,
        binfile=rp_bins,
        X=x, Y=y, Z=z,
        output_rpavg=True,
    )
    rp_arr = np.array([r['rpavg'] for r in results])
    wp_arr = np.array([r['wp'] for r in results])
    return rp_arr, wp_arr


class LuminosityAssigner:
    """
    Assign absolute magnitudes Mr to central and satellite galaxies
    via nested-HOD inverse-transform sampling.

    Scientific basis: Smith et al. (2024) BGS HOD.
        P(Lcen > L | M) is proportional to fcen_thresh(lgm, hod_params_at_L)
        P(Lsat > L | M) is proportional to Nsat_thresh(lgm, hod_params_at_L)

    The central occupation uses F_spline (BGS model), NOT erf.
    HOD equations are NOT re-implemented here — this class delegates
    entirely to fcen_thresh() and Nsat_thresh().
    """

    def __init__(self, Mrmax, seed=42, dMr=0.001):
        self.Mrmax = Mrmax
        self.rng = np.random.RandomState(seed=seed)

        nMr = int((23.5 + Mrmax) / dMr)
        self.Mrvals = np.linspace(-23.5, Mrmax, nMr)
        self.dMr = self.Mrvals[1] - self.Mrvals[0]

        self._hod_grid = get_hod_params(self.Mrvals)
        self._hod_max = get_hod_params(self.Mrmax)

    def assign_central_luminosity(self, lgm_halos):
        """Assign Mr to each central galaxy via inverse-transform sampling."""
        n = lgm_halos.size
        Mr_cen = np.zeros(n, dtype=float)
        u = self.rng.rand(n)
        nmax = self.Mrvals.size - 1

        Pnorm = fcen_thresh(lgm_halos, self._hod_max)

        for h in range(n):
            PcenL = fcen_thresh(lgm_halos[h], self._hod_grid) / Pnorm[h]
            idx = min(np.searchsorted(PcenL, u[h], side='right'), nmax)
            Mr_cen[h] = self.Mrvals[idx]

        return Mr_cen

    def assign_satellite_luminosities(self, centrals, h):
        """Assign Mr to all satellites in halo h via inverse-transform sampling."""
        Nsat_h = centrals['Nsat'][h]
        lgmhalo = centrals['lgm'][h]

        PsatL = Nsat_thresh(lgmhalo, self._hod_grid)
        norm = Nsat_thresh(lgmhalo, self._hod_max)

        if norm <= 0.0:
            return np.full(Nsat_h, self.Mrmax)

        PsatL /= norm
        u = self.rng.rand(Nsat_h)
        idx = np.clip(np.searchsorted(PsatL, u, side='left'), 0, self.Mrvals.size - 1)
        return self.Mrvals[idx]


def perturb_params(hod_params, param_name, factor):
    """
    Return a new HOD parameter dict with one parameter multiplied by `factor`.
    Does NOT mutate the input dict.
    """
    new_params = dict(hod_params)
    new_params[param_name] = hod_params[param_name] * factor
    return new_params


# ============================================================
# SECTION 4 — PIPELINE STAGES (orchestration; calls science
#             functions above, contains no science itself)
# ============================================================

def load_halo_catalogue(logger):
    """Load the Sahyadri halo catalogue via HaloReader, with header validation."""
    if str(Config.POSTPROCESS_PATH) not in sys.path:
        sys.path.insert(0, str(Config.POSTPROCESS_PATH))

    from readers import HaloReader
    from utilities import Paths

    if not Config.LOCAL_RUN:
        def _pegasus_paths_init(self):
            self.home_path = Config.PEGASUS_HOME_PATH
            self.scratch_path = Config.PEGASUS_SCRATCH_PATH
            self.config_path = self.home_path + "config/"
            self.python_path = self.home_path + "scripts/post-process/"
            self.sim_path = self.scratch_path + "sims/"
            self.halo_path = self.scratch_path + "halos/"
            self.gal_path = self.scratch_path + "galaxies/"
            self.config_transfer_path = self.config_path + "transfer/"
            self.config_sim_path = self.config_path + "sims/"
            self.config_halo_path = self.config_path + "halos/"
        Paths.__init__ = _pegasus_paths_init

    logger.info("Loading halo catalogue...")
    hr = HaloReader(sim_stem=Config.SIM_STEM, real=Config.REAL, snap=Config.SNAP,
                     read_header=True)
    hpos, halos = hr.prep_halos(
        va=False, massdef=Config.MASSDEF, Npmin=1000, QE=0.5,
        sorthalos=False, keep_subhalos=False,
    )

    lbox, omega_m, hubble, redshift = hr.Lbox, hr.Om, hr.hubble, hr.redshift

    logger.info(f"N_halos={halos.size:,}  Lbox={lbox}  Omega_m={omega_m}  "
                f"h={hubble}  redshift={redshift:.4f}")

    checks = {
        'N_halos': (halos.size, Config.EXPECTED_N_HALOS),
        'Lbox': (lbox, Config.EXPECTED_LBOX),
        'Omega_m': (omega_m, Config.EXPECTED_OMEGA_M),
    }
    all_ok = True
    for name, (got, expected) in checks.items():
        ok = bool(np.isclose(got, expected, rtol=1e-4))
        all_ok &= ok
        logger.info(f"Header check {name}: got={got}  expected={expected}  ok={ok}")
    assert all_ok, "Catalogue header mismatch — aborting."

    return halos, lbox, omega_m, hubble, redshift


def validate_hod_reference_point(logger):
    """Point-check get_hod_params() against the notebook's reference values."""
    ref_params = get_hod_params(Config.REF_MR)
    all_ok = True
    for k, exp_v in Config.REF_EXPECTED.items():
        got_v = ref_params[k]
        ok = np.isclose(got_v, exp_v, rtol=0.02)
        all_ok &= ok
        logger.info(f"HOD ref check {k}: got={got_v:.4e}  expected={exp_v:.4e}  ok={ok}")
    assert all_ok, "HOD parameter mismatch > 2% at reference point — check coefficients."


def validate_fspline_and_occupation(halos, base_hod, logger):
    """F_spline endpoint/linearisation checks + occupation sanity checks."""
    assert np.isclose(F_spline(-100.0), -1.0), "F_spline(-100) != -1"
    assert np.isclose(F_spline(0.0), 0.0), "F_spline(0) != 0"
    assert np.isclose(F_spline(100.0), +1.0), "F_spline(100) != +1"

    x_small = np.linspace(-0.1, 0.1, 21)
    rel_err = np.max(np.abs(F_spline(x_small) - erf(x_small)) / (np.abs(erf(x_small)) + 1e-12))
    assert rel_err < 0.05, f"F_spline deviates from erf by {rel_err:.2%} near x=0"

    lgm_test = np.linspace(11.0, 15.0, 200)
    fcen_test = fcen_thresh(lgm_test, base_hod)
    assert np.all(np.diff(fcen_test) >= -1e-12), "fcen_thresh is not monotone with mass!"
    assert np.all(fcen_test >= 0.0) and np.all(fcen_test <= 1.0), "<Ncen> out of [0,1]"

    nsat_test = Nsat_thresh(lgm_test, base_hod)
    assert np.all(nsat_test >= 0.0), "<Nsat> < 0 detected"

    logger.info(f"F_spline/occupation checks passed. "
                f"<Ncen> range=[{fcen_test.min():.4f}, {fcen_test.max():.4f}]  "
                f"max <Nsat>={nsat_test.max():.2f}")


def validate_rvir_nfw(halos, redshift, omega_m, logger):
    """Validate compute_rvir_con and gen_NFW_profile on synthetic/real inputs."""
    rvir_all, con_all = compute_rvir_con(halos[Config.MASSDEF], halos['rs'],
                                          z=redshift, Om=omega_m)
    assert np.all(rvir_all > 0), 'negative rvir detected'
    assert np.all(con_all > 0), 'negative concentration detected'
    frac_highc = np.sum(con_all > 100) / con_all.size
    assert frac_highc < 0.05, f'Too many high-c halos ({100*frac_highc:.1f}%)'
    logger.info(f"rvir/concentration validated. median c={np.median(con_all):.2f}  "
                f"c>100 frac={frac_highc:.2%}")

    _rng_test = np.random.RandomState(0)
    _pos_test = gen_NFW_profile(1000, cvir=10.0, Rvir=1.0, rng=_rng_test)
    _r_test = np.sqrt((_pos_test ** 2).sum(axis=1))
    assert _pos_test.shape == (1000, 3), 'shape mismatch'
    assert np.all(_r_test > 0) and np.all(_r_test < 2.0), 'NFW sampler out of range'
    logger.info("gen_NFW_profile validated.")

    return rvir_all, con_all


def build_baseline_mock(halos, base_hod, lbox, redshift, omega_m, logger):
    """Generate + validate the baseline mock galaxy catalogue."""
    generate_mock.REDSHIFT = redshift
    generate_mock.OMEGA_M = omega_m

    mock_base = generate_mock(halos, base_hod, lbox=lbox, massdef=Config.MASSDEF,
                               seed=Config.SEED, logger=logger, verbose=True)

    ntot = mock_base['ncen_total'] + mock_base['nsat_total']
    fsat = mock_base['nsat_total'] / ntot

    for coord in ['x', 'y', 'z']:
        assert mock_base[coord].min() >= 0.0, f'{coord} < 0'
        assert mock_base[coord].max() < lbox, f'{coord} >= LBOX'
    assert mock_base['is_central'].sum() == mock_base['ncen_total'], 'is_central mismatch'
    assert (~mock_base['is_central']).sum() == mock_base['nsat_total'], 'is_satellite mismatch'
    assert 0.10 < fsat < 0.40, f'f_sat={fsat:.3f} outside expected range [0.10, 0.40]'

    logger.info(f"Baseline mock validated. N_gal={ntot:,}  f_sat={fsat:.4f}  "
                f"centrals={mock_base['ncen_total']:,}  satellites={mock_base['nsat_total']:,}")
    return mock_base


def build_luminosity_catalogue(halos, mock_base, logger):
    """Assign Mr to centrals and satellites via LuminosityAssigner."""
    mm = LuminosityAssigner(Mrmax=Config.MR_THRESH, seed=42)

    cen_halo_idx = mock_base['halo_idx'][mock_base['is_central']]
    centrals = np.zeros(cen_halo_idx.size, dtype=[
        ('haloid', 'int64'), ('lgm', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('Mr', 'f8'), ('Nsat', 'i4'),
    ])
    centrals['haloid'] = cen_halo_idx
    centrals['lgm'] = np.log10(halos[Config.MASSDEF][cen_halo_idx])
    centrals['x'] = mock_base['x'][mock_base['is_central']]
    centrals['y'] = mock_base['y'][mock_base['is_central']]
    centrals['z'] = mock_base['z'][mock_base['is_central']]

    nsat_per_halo = np.zeros(halos.size, dtype=int)
    sat_hidx = mock_base['halo_idx'][~mock_base['is_central']]
    np.add.at(nsat_per_halo, sat_hidx, 1)
    centrals['Nsat'] = nsat_per_halo[cen_halo_idx]

    centrals['Mr'] = mm.assign_central_luminosity(centrals['lgm'])

    Nsat_tot = int(centrals['Nsat'].sum())
    satellites = np.zeros(Nsat_tot, dtype=[
        ('haloid', 'int64'), ('lgm', 'f8'),
        ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
        ('Mr', 'f8'),
    ])
    sat_mask = ~mock_base['is_central']
    satellites['x'] = mock_base['x'][sat_mask]
    satellites['y'] = mock_base['y'][sat_mask]
    satellites['z'] = mock_base['z'][sat_mask]
    satellites['haloid'] = mock_base['halo_idx'][sat_mask]
    satellites['lgm'] = np.log10(halos[Config.MASSDEF][satellites['haloid'].astype(int)])

    s_lo = 0
    for h in range(centrals.size):
        s_hi = s_lo + centrals['Nsat'][h]
        if centrals['Nsat'][h] > 0:
            satellites['Mr'][s_lo:s_hi] = mm.assign_satellite_luminosities(centrals, h)
        s_lo = s_hi

    cen_ok = np.all((centrals['Mr'] >= -23.5) & (centrals['Mr'] <= Config.MR_THRESH))
    sat_ok = np.all((satellites['Mr'] >= -23.5) & (satellites['Mr'] <= Config.MR_THRESH))
    assert cen_ok, 'Central Mr out of range'
    assert sat_ok, 'Satellite Mr out of range'

    logger.info(f"Luminosity catalogue built. centrals={centrals.size:,}  "
                f"satellites={satellites.size:,}")
    return centrals, satellites


def run_derivative_engine(halos, base_hod, lbox, rp_bins, redshift, omega_m, logger):
    """
    Run the finite-difference derivative loop over HOD_PARAM_NAMES at both
    DELTAS step sizes, then the convergence check identifying NOISY_PARAMS.
    """
    derivs = {p: {} for p in Config.HOD_PARAM_NAMES}

    for param in Config.HOD_PARAM_NAMES:
        for delta in Config.DELTAS:
            p_plus = perturb_params(base_hod, param, 1.0 + delta)
            p_minus = perturb_params(base_hod, param, 1.0 - delta)

            mock_plus = generate_mock(halos, p_plus, lbox=lbox, massdef=Config.MASSDEF,
                                       seed=Config.SEED, logger=logger, verbose=False)
            mock_minus = generate_mock(halos, p_minus, lbox=lbox, massdef=Config.MASSDEF,
                                        seed=Config.SEED, logger=logger, verbose=False)

            rp_p, wp_p = compute_wp(mock_plus, lbox, Config.PIMAX, rp_bins, Config.NTHREADS)
            rp_m, wp_m = compute_wp(mock_minus, lbox, Config.PIMAX, rp_bins, Config.NTHREADS)

            dtheta = base_hod[param] * 2.0 * delta
            dwp = (wp_p - wp_m) / dtheta
            derivs[param][delta] = (rp_p, dwp)

            logger.info(f"Derivative {param} delta={delta}: computed.")

            del mock_plus, mock_minus
            gc.collect()

    NOISY_PARAMS = []
    all_converged = True
    for param in Config.HOD_PARAM_NAMES:
        _, d5 = derivs[param][0.050]
        _, d25 = derivs[param][0.025]
        denom = np.maximum(np.abs(d5), np.abs(d25))
        mask = denom > 1e-8
        rel_diff = np.max(np.abs(d5[mask] - d25[mask]) / denom[mask]) if mask.any() else 0.0
        ok = rel_diff < Config.CONVERGENCE_TOL
        if not ok:
            NOISY_PARAMS.append(param)
        all_converged &= ok
        logger.info(f"Convergence {param}: rel_diff={rel_diff:.4f}  converged={ok}")

    logger.info(f"NOISY_PARAMS={NOISY_PARAMS}")
    return derivs, NOISY_PARAMS


def run_multiseed_averaging(halos, base_hod, lbox, rp_bins, noisy_params, derivs, logger):
    """Multi-seed averaging + stabilization check for NOISY_PARAMS."""
    if not noisy_params:
        logger.info("No noisy parameters — skipping multi-seed averaging.")
        return {}, {}

    wp_plus_stack_all = {p: [] for p in noisy_params}
    wp_minus_stack_all = {p: [] for p in noisy_params}

    for real_seed in range(Config.N_REALIZATIONS):
        for param in noisy_params:
            p_plus = perturb_params(base_hod, param, 1.0 + Config.DELTA_STABLE)
            p_minus = perturb_params(base_hod, param, 1.0 - Config.DELTA_STABLE)

            mock_plus = generate_mock(halos, p_plus, lbox=lbox, massdef=Config.MASSDEF,
                                       seed=real_seed, logger=logger, verbose=False)
            mock_minus = generate_mock(halos, p_minus, lbox=lbox, massdef=Config.MASSDEF,
                                        seed=real_seed, logger=logger, verbose=False)

            _, wp_p = compute_wp(mock_plus, lbox, Config.PIMAX, rp_bins, Config.NTHREADS)
            _, wp_m = compute_wp(mock_minus, lbox, Config.PIMAX, rp_bins, Config.NTHREADS)

            wp_plus_stack_all[param].append(wp_p)
            wp_minus_stack_all[param].append(wp_m)

            del mock_plus, mock_minus
            gc.collect()
        logger.info(f"Realization {real_seed + 1}/{Config.N_REALIZATIONS} complete.")

    derivs_averaged = {}
    for param in noisy_params:
        wp_plus_mean = np.mean(wp_plus_stack_all[param], axis=0)
        wp_minus_mean = np.mean(wp_minus_stack_all[param], axis=0)
        wp_plus_std = np.std(wp_plus_stack_all[param], axis=0, ddof=1)
        wp_minus_std = np.std(wp_minus_stack_all[param], axis=0, ddof=1)

        dtheta = base_hod[param] * 2.0 * Config.DELTA_STABLE
        dwp_mean = (wp_plus_mean - wp_minus_mean) / dtheta
        dwp_err = np.sqrt(wp_plus_std ** 2 + wp_minus_std ** 2) / (
            dtheta * np.sqrt(Config.N_REALIZATIONS))

        rp_ref, _ = derivs[param][Config.DELTA_STABLE]
        derivs_averaged[param] = (rp_ref, dwp_mean, dwp_err)
        logger.info(f"Averaged derivative for {param} complete.")

    # Stabilization check
    convergence_summary = {}
    for param in noisy_params:
        p_plus_arr = np.array(wp_plus_stack_all[param])
        p_minus_arr = np.array(wp_minus_stack_all[param])
        dtheta = base_hod[param] * 2.0 * Config.DELTA_STABLE

        running = []
        for n in range(1, Config.N_REALIZATIONS + 1):
            avg_plus = np.mean(p_plus_arr[:n], axis=0)
            avg_minus = np.mean(p_minus_arr[:n], axis=0)
            deriv = (avg_plus - avg_minus) / dtheta
            running.append(deriv[Config.STABILIZATION_BIN_IDX])
        running = np.array(running)

        n_final = Config.N_REALIZATIONS
        n_ref = max(1, n_final - Config.STABILIZATION_WINDOW)
        val_final = running[n_final - 1]
        val_ref = running[n_ref - 1]
        rel_change = abs(val_final - val_ref) / max(abs(val_final), 1e-30)
        is_conv = rel_change < Config.STABILIZATION_TOL
        convergence_summary[param] = (rel_change, is_conv, running)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, Config.N_REALIZATIONS + 1), running,
                marker='o', lw=1.8, color='teal', ms=6, label='Running average')
        ax.axhline(val_final, color='red', ls='--', alpha=0.7, lw=1.5,
                   label=f'Final (N={n_final}): {val_final:.4f}')
        ax.axvspan(n_ref, n_final, color='gray', alpha=0.15,
                   label=f'Last {Config.STABILIZATION_WINDOW} realizations')
        ax.set_xlabel('Number of realizations (N)')
        ax.set_ylabel(f'd(wp)/d({param})  (bin {Config.STABILIZATION_BIN_IDX})')
        ax.set_title(f'Stabilisation: {param}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        savefig(fig, Config.FIGURE_DIR / f'stabilization_{param}.png', logger)

        status = 'CONVERGED' if is_conv else 'STILL STABILISING'
        logger.info(f"Stabilization {param}: rel_change={rel_change:.4f}  status={status}")

    return derivs_averaged, convergence_summary


def assemble_final_derivatives(base_hod, derivs, noisy_params, derivs_averaged, logger):
    """Choose averaged vs single-realization derivative per parameter."""
    FINAL_DERIVS = {}
    FINAL_METHOD = {}
    rp_ref = None

    for param in Config.HOD_PARAM_NAMES:
        if param in noisy_params and param in derivs_averaged:
            rp_ref, dwp_mean, _ = derivs_averaged[param]
            FINAL_DERIVS[param] = dwp_mean
            FINAL_METHOD[param] = f'averaged N={Config.N_REALIZATIONS} delta=+-{100*Config.DELTA_STABLE:.1f}%'
        else:
            rp_ref, dwp_single = derivs[param][0.025]
            FINAL_DERIVS[param] = dwp_single
            FINAL_METHOD[param] = 'single-realization delta=+-2.5%'
        logger.info(f"Final method for {param}: {FINAL_METHOD[param]}")

    return FINAL_DERIVS, FINAL_METHOD, rp_ref


# ============================================================
# SECTION 5 — MAIN ORCHESTRATION
# ============================================================

def main():
    args = build_arg_parser().parse_args()
    Config.apply_cli_overrides(args)

    make_directories()
    logger = setup_logger()

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("HOD DERIVATIVE PIPELINE START")
    logger.info(f"Start time     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Seed           : {Config.SEED}")
    logger.info(f"Mr threshold   : {Config.MR_THRESH}")
    logger.info(f"Realizations   : {Config.N_REALIZATIONS}")
    logger.info(f"pimax          : {Config.PIMAX}")
    logger.info(f"nthreads       : {Config.NTHREADS}")
    logger.info("=" * 70)

    mpl.rcParams.update(Config.MPL_RCPARAMS)
    np.random.seed(Config.SEED)

    try:
        # ---- Stage: load halos ----
        with timer(logger, "load_halo_catalogue"):
            halos, lbox, omega_m, hubble, redshift = load_halo_catalogue(logger)
        logger.info(f"Number of halos: {halos.size:,}")

        # ---- Stage: baseline HOD + validation ----
        with timer(logger, "baseline_hod"):
            validate_hod_reference_point(logger)
            base_hod = get_hod_params(Config.MR_THRESH)
            logger.info(f"Baseline HOD params: {base_hod}")

        # ---- Stage: F_spline / occupation validation ----
        with timer(logger, "fspline_occupation_validation"):
            validate_fspline_and_occupation(halos, base_hod, logger)

        # ---- Stage: rvir/NFW validation ----
        with timer(logger, "rvir_nfw_validation"):
            validate_rvir_nfw(halos, redshift, omega_m, logger)

        # ---- Stage: baseline mock generation + save ----
        with timer(logger, "generate_baseline_mock"):
            mock_base = build_baseline_mock(halos, base_hod, lbox, redshift, omega_m, logger)
            cat_path = Config.CATALOGUE_DIR / f"mock_seed{Config.SEED}_Mr{abs(Config.MR_THRESH):.0f}.npz"
            save_catalogue(mock_base, cat_path, logger)

        # ---- Stage: baseline wp ----
        rp_bins = np.logspace(np.log10(Config.RP_MIN), np.log10(Config.RP_MAX), Config.N_RP + 1)
        with timer(logger, "compute_baseline_wp"):
            rp_base, wp_base = compute_wp(mock_base, lbox, Config.PIMAX, rp_bins, Config.NTHREADS)
            assert np.all(wp_base > 0), 'wp has non-positive values'
            assert wp_base[0] > wp_base[-1], 'wp not decreasing — clustering problem'
            wp_path = Config.WP_DIR / f"wp_seed{Config.SEED}.npz"
            save_wp(rp_base, wp_base, wp_path, logger)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.loglog(rp_base, wp_base, 'ko-', ms=4, lw=1.8, label=f'Baseline (seed={Config.SEED})')
            ax.set_xlabel(r'$r_p\;[h^{-1}\mathrm{Mpc}]$')
            ax.set_ylabel(r'$w_p(r_p)\;[h^{-1}\mathrm{Mpc}]$')
            ax.set_title(f'Projected Correlation Function  Mr<{Config.MR_THRESH}')
            ax.legend(fontsize=9)
            savefig(fig, Config.FIGURE_DIR / 'wp_baseline.png', logger)

        # ---- Stage: luminosity catalogue ----
        with timer(logger, "build_luminosity_catalogue"):
            centrals, satellites = build_luminosity_catalogue(halos, mock_base, logger)

        # ---- Stage: derivative engine ----
        with timer(logger, "derivative_engine"):
            derivs, noisy_params = run_derivative_engine(
                halos, base_hod, lbox, rp_bins, redshift, omega_m, logger)

            for param in Config.HOD_PARAM_NAMES:
                fig, ax = plt.subplots(figsize=(7, 4))
                for delta, ls, col in zip(Config.DELTAS, ['-', '--'], ['steelblue', 'darkorange']):
                    rp, dwp = derivs[param][delta]
                    ax.plot(rp, dwp, ls=ls, color=col, lw=1.8, label=f'delta=+-{100*delta:.1f}%')
                ax.axhline(0, color='gray', lw=0.8, ls=':')
                ax.set_xscale('log')
                ax.set_xlabel(r'$r_p\;[h^{-1}\mathrm{Mpc}]$')
                ax.set_ylabel(r'$\partial w_p / \partial\theta$')
                ax.set_title(f'd(wp)/d({param})  Mr<{Config.MR_THRESH}  seed={Config.SEED}')
                ax.legend(fontsize=9)
                savefig(fig, Config.FIGURE_DIR / f'derivative_{param}.png', logger)

        # ---- Stage: multi-seed averaging for noisy params ----
        with timer(logger, "multiseed_averaging"):
            derivs_averaged, convergence_summary = run_multiseed_averaging(
                halos, base_hod, lbox, rp_bins, noisy_params, derivs, logger)

        # ---- Stage: assemble + save final derivatives ----
        with timer(logger, "assemble_and_save_final_derivatives"):
            FINAL_DERIVS, FINAL_METHOD, rp_ref = assemble_final_derivatives(
                base_hod, derivs, noisy_params, derivs_averaged, logger)

            outfile = Config.DERIV_DIR / f"derivatives_seed{Config.SEED}.npz"
            save_dict = {
                'rp': rp_ref, 'wp_base': wp_base, 'rp_base': rp_base,
                'seed': np.int64(Config.SEED),
                'Mr_thresh': np.float64(Config.MR_THRESH),
                'pimax': np.float64(Config.PIMAX),
                'Lbox': np.float64(lbox),
            }
            for pname in Config.HOD_PARAM_NAMES:
                save_dict[f'base_{pname}'] = np.float64(base_hod[pname])
                save_dict[f'dwp_d{pname}_FINAL'] = FINAL_DERIVS[pname]
                save_dict[f'dwp_d{pname}_method'] = FINAL_METHOD[pname]
                for delta in Config.DELTAS:
                    rp_d, dwp_d = derivs[pname][delta]
                    save_dict[f'dwp_d{pname}_delta{int(1000*delta):03d}'] = dwp_d
            save_derivatives(save_dict, outfile, logger)

            for param in Config.HOD_PARAM_NAMES:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(rp_ref, FINAL_DERIVS[param], color='crimson', lw=2.2)
                ax.axhline(0, color='gray', lw=0.8, ls=':')
                ax.set_xscale('log')
                ax.set_xlabel(r'$r_p\;[h^{-1}\mathrm{Mpc}]$')
                ax.set_ylabel(r'$\partial w_p / \partial\theta$')
                ax.set_title(f'FINAL d(wp)/d({param})\n({FINAL_METHOD[param]})')
                ax.grid(True, alpha=0.3)
                savefig(fig, Config.FIGURE_DIR / f'derivative_{param}_FINAL.png', logger)

        # ---- Summary ----
        ntot = mock_base['ncen_total'] + mock_base['nsat_total']
        runtime = time.time() - t_start
        logger.info("=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info(f"  Halo catalogue   : {Config.SIM_STEM}, {halos.size:,} halos")
        logger.info(f"  Mr threshold     : {Config.MR_THRESH}")
        logger.info(f"  Galaxies (base)  : {ntot:,}  "
                    f"(centrals={mock_base['ncen_total']:,}, satellites={mock_base['nsat_total']:,})")
        logger.info(f"  Parameters       : {Config.HOD_PARAM_NAMES}")
        logger.info(f"  Catalogue file   : {cat_path}")
        logger.info(f"  wp file          : {wp_path}")
        logger.info(f"  Derivatives file : {outfile}")
        logger.info(f"  Total runtime    : {runtime:.2f} s")
        logger.info("FINISHED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception:
        logger.exception("Pipeline failed with an exception:")
        raise


if __name__ == "__main__":
    main()
