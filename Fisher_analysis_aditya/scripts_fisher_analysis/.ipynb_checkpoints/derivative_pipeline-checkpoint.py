"""
derivative_pipeline.py — HOD-parameter derivatives of wp(rp), using the
RSD mock pipeline (generate_mock_rsd / compute_wp from hod_pipeline.py).

Preserves the exact derivative logic from redo_derivative.ipynb
(perturb_params, get_dtheta, the finite-difference formula) — only the
mock-generation call is swapped from the old hand-rolled generate_mock()
to generate_mock_rsd().
"""
import numpy as np
from pathlib import Path

from config import DERIVATIVES_DIR, SEED
from hod_pipeline import generate_mock_rsd, compute_wp

HOD_PARAM_NAMES = ['Mcut', 'M1', 'M0', 'sigma', 'alpha']
MASS_PARAMS     = ['Mcut', 'M1', 'M0']   # derivative taken w.r.t. log10(mass)
DELTAS          = [0.050, 0.025]         # ±5%, ±2.5%

_DELTA_TAG = {0.050: '5', 0.025: '2p5'}  # for filenames: Mcut_plus5.npz, Mcut_plus2p5.npz


def perturb_params(hod_params, param_name, factor):
    """Return a new HOD dict with one parameter multiplied by `factor`. Does not mutate input."""
    new_params = dict(hod_params)
    new_params[param_name] = hod_params[param_name] * factor
    return new_params


def get_dtheta(param, delta, base_hod_params):
    """
    Finite-difference denominator for a ±delta relative perturbation.
    Mass params: derivative w.r.t. log10(mass) — base-independent.
    Non-mass params: derivative w.r.t. the linear value.
    """
    if param in MASS_PARAMS:
        return np.log10(1.0 + delta) - np.log10(1.0 - delta)
    else:
        return base_hod_params[param] * 2.0 * delta


def generate_perturbed_mock(base_hod_params, param, factor, seed, Lbox, Omega_m):
    """
    perturb_params(...) -> generate_mock_rsd(...).
    kappa = M0/Mcut is recomputed fresh inside generate_mock_rsd from
    whatever perturbed dict is passed in — no special-casing needed here
    for Mcut or M0 (see confirmed reasoning: perturbing Mcut alone still
    changes kappa correctly since M0 stays at baseline in the dict, and
    vice versa).
    """
    hod_params = perturb_params(base_hod_params, param, factor)
    np.random.seed(seed)   # explicit — simulator.py's random_seed arg alone does not seed
    mock = generate_mock_rsd(hod_params, seed=seed, rsd=True, Lbox=Lbox, Omega_m=Omega_m)
    return mock, hod_params


def compute_derivative(base_hod_params, param, delta, seed, Lbox, Omega_m,
                        save_dir=DERIVATIVES_DIR, verbose=True):
    """
    Runs +delta and -delta mocks (same seed for both), computes wp for
    each, saves each wp result to save_dir, computes dwp/dtheta, returns
    (rp, dwp, dtheta).
    """
    save_dir = Path(save_dir)
    tag = _DELTA_TAG[delta]

    if verbose:
        print(f'\n── {param}  δ = ±{100*delta:.1f}% ──')

    mock_plus,  hod_plus  = generate_perturbed_mock(base_hod_params, param, 1.0 + delta, seed, Lbox, Omega_m)
    mock_minus, hod_minus = generate_perturbed_mock(base_hod_params, param, 1.0 - delta, seed, Lbox, Omega_m)

    if verbose:
        n_p = mock_plus['ncen_total']  + mock_plus['nsat_total']
        n_m = mock_minus['ncen_total'] + mock_minus['nsat_total']
        print(f'  {param}: {base_hod_params[param]:.6g} -> '
              f'+{100*delta:.1f}%: {hod_plus[param]:.6g}  |  -{100*delta:.1f}%: {hod_minus[param]:.6g}')
        print(f'  N_gal: +{100*delta:.1f}% = {n_p:,}   -{100*delta:.1f}% = {n_m:,}')

    rp_p, wp_p = compute_wp(mock_plus,  lbox=Lbox)
    rp_m, wp_m = compute_wp(mock_minus, lbox=Lbox)

    # save individual +/- wp results
    plus_path  = save_dir / f'{param}_plus{tag}.npz'
    minus_path = save_dir / f'{param}_minus{tag}.npz'
    np.savez(plus_path,  rp=rp_p, wp=wp_p, param=param, delta=delta, factor=1.0 + delta,
             hod_param_value=hod_plus[param], seed=seed)
    np.savez(minus_path, rp=rp_m, wp=wp_m, param=param, delta=delta, factor=1.0 - delta,
             hod_param_value=hod_minus[param], seed=seed)
    if verbose:
        print(f'  Saved -> {plus_path.name}, {minus_path.name}')

    dtheta = get_dtheta(param, delta, base_hod_params)
    dwp = (wp_p - wp_m) / dtheta

    return rp_p, dwp, dtheta


def run_all_derivatives(base_hod_params, seed=SEED, Lbox=None, Omega_m=None,
                         param_names=HOD_PARAM_NAMES, deltas=DELTAS,
                         save_dir=DERIVATIVES_DIR, verbose=True):
    """
    Loops over param_names x deltas, calls compute_derivative for each,
    collects results, saves the bundled derivatives.npz.

    Returns: derivs dict, structured as derivs[param][delta] = (rp, dwp)
    """
    assert Lbox is not None and Omega_m is not None, "Lbox and Omega_m must be provided"

    derivs = {p: {} for p in param_names}
    rp_ref = None

    for param in param_names:
        for delta in deltas:
            rp, dwp, dtheta = compute_derivative(
                base_hod_params, param, delta, seed, Lbox, Omega_m,
                save_dir=save_dir, verbose=verbose,
            )
            derivs[param][delta] = (rp, dwp)
            if rp_ref is None:
                rp_ref = rp

    # bundle into one file
    bundle = {'rp': rp_ref, 'seed': seed, 'Mr_thresh': None,  # caller can overwrite Mr_thresh if needed
              'param_names': np.array(param_names), 'deltas': np.array(deltas),
              'mass_params': np.array(MASS_PARAMS)}
    for param in param_names:
        for delta in deltas:
            tag = _DELTA_TAG[delta]
            _, dwp = derivs[param][delta]
            bundle[f'dwp_{param}_{tag}'] = dwp

    bundle_path = Path(save_dir) / 'derivatives.npz'
    np.savez(bundle_path, **bundle)
    if verbose:
        print(f'\nSaved bundled derivatives -> {bundle_path}')

    return derivs