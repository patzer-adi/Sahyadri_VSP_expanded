"""
hod_pipeline.py — halo loading, HOD-mocker (via simulator.py), wp(rp)
computation, and save/load helpers.
"""
import sys
import numpy as np
import fitsio as F
import importlib.util
from pathlib import Path
from astropy.table import Table
from Corrfunc.theory.wp import wp as corrfunc_wp

from config import (
    POSTPROCESS_PATH, SPECTROPHOTO_PATH, MODIFIED_POISSON_PATH,
    SIM_STEM, REAL, SNAP, MASSDEF,
    EXPECTED_N_HALOS, EXPECTED_LBOX, EXPECTED_OMEGA_M,
    HALO_FITS_BOLSHOI_PATH,
    RP_MIN, RP_MAX, N_RP, PIMAX, NTHREADS,
)

RP_BINS = np.logspace(np.log10(RP_MIN), np.log10(RP_MAX), N_RP + 1)


# ---------------------------------------------------------------------
# 1. Load Sahyadri halos (once)
# ---------------------------------------------------------------------
def load_halos():
    if str(POSTPROCESS_PATH) not in sys.path:
        sys.path.insert(0, str(POSTPROCESS_PATH))
    from readers import HaloReader
    from utilities import Paths

    def _pegasus_paths_init(self):
        self.home_path    = "/mnt/home/project/cgowari.aditya/sahyadri-codes/sahyadri-sandbox/"
        self.scratch_path = "/data/project/hpc2502016/data/"
        self.config_path  = self.home_path + "config/"
        self.python_path  = self.home_path + "scripts/post-process/"
        self.sim_path     = self.scratch_path + "sims/"
        self.halo_path    = self.scratch_path + "halos/"
        self.gal_path     = self.scratch_path + "galaxies/"
        self.config_transfer_path = self.config_path + "transfer/"
        self.config_sim_path      = self.config_path + "sims/"
        self.config_halo_path     = self.config_path + "halos/"
    Paths.__init__ = _pegasus_paths_init

    hr = HaloReader(sim_stem=SIM_STEM, real=REAL, snap=SNAP, read_header=True)
    hpos, halos = hr.prep_halos(
        va=False, massdef=MASSDEF, Npmin=1000, QE=0.5,
        sorthalos=False, keep_subhalos=False,
    )

    checks = {
        'N_halos': (halos.size, EXPECTED_N_HALOS),
        'Lbox':    (hr.Lbox,    EXPECTED_LBOX),
        'Omega_m': (hr.Om,      EXPECTED_OMEGA_M),
    }
    for name, (got, expected) in checks.items():
        assert np.isclose(got, expected, rtol=1e-4), f"{name} mismatch: got {got}, expected {expected}"

    return halos, hr.Lbox, hr.Om, hr.hubble, hr.redshift


# ---------------------------------------------------------------------
# 2. Write halos in fits_Bolshoi schema (once)
# ---------------------------------------------------------------------
def write_fits_bolshoi(halos, path=HALO_FITS_BOLSHOI_PATH):
    fits_dtype = np.dtype([
        ("x", "f8"), ("y", "f8"), ("z", "f8"),
        ("vx", "f8"), ("vy", "f8"), ("vz", "f8"),
        ("M200b", "f8"), ("pid", "i8"), ("upid", "i8"),
        ("Rvir", "f8"), ("rs", "f8"), ("id", "i8"),
    ])
    fits_data = np.zeros(halos.size, dtype=fits_dtype)
    fits_data["x"], fits_data["y"], fits_data["z"] = halos["x"], halos["y"], halos["z"]
    fits_data["vx"], fits_data["vy"], fits_data["vz"] = halos["vx"], halos["vy"], halos["vz"]
    fits_data["M200b"] = halos["m200b"]
    fits_data["pid"]   = -1
    fits_data["upid"]  = -1
    fits_data["Rvir"]  = halos["rvir"]
    fits_data["rs"]    = halos["rs"]
    fits_data["id"]    = np.arange(halos.size)

    path = Path(path)
    if path.exists():
        path.unlink()
    with F.FITS(str(path), "rw", clobber=True) as fout:
        fout.write(fits_data)
    print(f"Wrote {fits_data.size} halos -> {path}")
    return path


# ---------------------------------------------------------------------
# 3. HOD meta-parameters — Smith et al. (2024) Eq. 5-9
# ---------------------------------------------------------------------
def _M_function(magnitude, A, B, C, D):
    return (A + 12) + B*(magnitude+20) + C*(magnitude+20)**2 + D*(magnitude+20)**3

def _M0_function(magnitude, A, B):
    return (A + 11) + B*(magnitude + 20)

def _sigma_function(magnitude, A, B, C, D):
    return A + (B - A) / (1. + np.exp(C*(magnitude + 20 + D)))

def _alpha_function(magnitude, A, B, C):
    return A + B**(-magnitude - 20 + C)

_PARAMS = np.array([
    -1.163295942035853792e-01, -5.759066407274642252e-01,
     1.437085418067726994e-01, -1.908335449679073731e-02,
     1.832666435246161490e-02,  7.584865035087475782e-01,
     1.103103765754579246e+00,  3.277787214124110449e-01,
    -5.468895455660652827e-01, -1.673562380257160864e+00,
     1.121497360473139082e+00, -4.725625165007949491e-01,
     9.200038944813740405e-02, -1.141762462160326314e-02,
     1.111866664500087198e+00,  4.473949866395862784e+00,
    -4.353309870091147893e+00,
])

def get_hod_params(magnitude):
    Mcut  = 10**_M_function(magnitude, *_PARAMS[0:4])
    M1    = 10**_M_function(magnitude, *_PARAMS[10:14])
    M0    = 10**_M0_function(magnitude, *_PARAMS[8:10])
    sigma = _sigma_function(magnitude, *_PARAMS[4:8])
    alpha = _alpha_function(magnitude, *_PARAMS[14:17])
    return {'Mcut': Mcut, 'M1': M1, 'M0': M0, 'sigma': sigma, 'alpha': alpha}


# ---------------------------------------------------------------------
# 4. simulator.py loader
# ---------------------------------------------------------------------
_simulator = None

def get_simulator():
    global _simulator
    if _simulator is not None:
        return _simulator
    if str(MODIFIED_POISSON_PATH) not in sys.path:
        sys.path.insert(0, str(MODIFIED_POISSON_PATH))
    spec = importlib.util.spec_from_file_location("simulator", str(SPECTROPHOTO_PATH))
    simulator = importlib.util.module_from_spec(spec)
    sys.modules["simulator"] = simulator
    spec.loader.exec_module(simulator)
    _simulator = simulator
    return simulator


# ---------------------------------------------------------------------
# 5. Mock generation
#    NOTE: central_model="BGS" and kappa=M0/Mcut are NOT YET VERIFIED
#    against Chetan's current pipeline (an older notebook of his used
#    central_model="erf"). Do not treat as settled.
# ---------------------------------------------------------------------
def generate_mock_rsd(hod_params, seed, rsd=True, Lbox=None, Omega_m=None,
                       halo_fits_path=HALO_FITS_BOLSHOI_PATH):
    simulator = get_simulator()
    kappa = hod_params['M0'] / hod_params['Mcut']

    np.random.seed(seed)   # explicit — simulator.py's random_seed arg does not itself seed anything
    mock = simulator.default_mock(
        halo_file=str(halo_fits_path), halo_ftype="fits_Bolshoi",
        central_model="BGS", satellite_model="power_law",
        Mcut=hod_params['Mcut'], M1=hod_params['M1'],
        sigma=hod_params['sigma'], alpha=hod_params['alpha'],
        kappa=kappa, pmax=1.0,
        radial_profile="NFW", angular_distribution="spherical",
        geometry="cubic", rsd=rsd, gravz=False,
        Omega_m=Omega_m, Lbox=Lbox,
        return_full_obj=True, random_seed=seed, verbose=False,
    )

    x = np.concatenate([mock.cen_xyz[:, 0], mock.sat_xyz[:, 0]])
    y = np.concatenate([mock.cen_xyz[:, 1], mock.sat_xyz[:, 1]])
    z = np.concatenate([mock.cen_xyz[:, 2], mock.sat_xyz[:, 2]])
    n_cen = mock.cen_xyz.shape[0]

    return {'x': x, 'y': y, 'z': z,
            'ncen_total': n_cen, 'nsat_total': mock.sat_xyz.shape[0]}


# ---------------------------------------------------------------------
# 6. compute_wp
# ---------------------------------------------------------------------
def compute_wp(mock, lbox=None, pimax=PIMAX, rp_bins=RP_BINS, nthreads=NTHREADS):
    x = mock['x'].astype(np.float64)
    y = mock['y'].astype(np.float64)
    z = mock['z'].astype(np.float64)
    results = corrfunc_wp(lbox, pimax, nthreads, binfile=rp_bins, X=x, Y=y, Z=z, output_rpavg=True)
    rp_arr = np.array([r['rpavg'] for r in results])
    wp_arr = np.array([r['wp']    for r in results])
    return rp_arr, wp_arr


# ---------------------------------------------------------------------
# 7. Save / load — minimal schema (x, y, z only)
# ---------------------------------------------------------------------
def save_mock_fits(mock, centrals_path, satellites_path):
    n_cen = mock['ncen_total']
    cen_table = Table({'x': mock['x'][:n_cen], 'y': mock['y'][:n_cen], 'z': mock['z'][:n_cen]})
    sat_table = Table({'x': mock['x'][n_cen:], 'y': mock['y'][n_cen:], 'z': mock['z'][n_cen:]})

    for p in (Path(centrals_path), Path(satellites_path)):
        if p.exists():
            p.unlink()
    cen_table.write(centrals_path, format='fits')
    sat_table.write(satellites_path, format='fits')
    print(f'Saved {len(cen_table):,} centrals -> {centrals_path}')
    print(f'Saved {len(sat_table):,} satellites -> {satellites_path}')
    return centrals_path, satellites_path


def load_mock_fits(centrals_path, satellites_path):
    cen = Table.read(centrals_path)
    sat = Table.read(satellites_path)
    x = np.concatenate([np.asarray(cen['x']), np.asarray(sat['x'])])
    y = np.concatenate([np.asarray(cen['y']), np.asarray(sat['y'])])
    z = np.concatenate([np.asarray(cen['z']), np.asarray(sat['z'])])
    print(f'Loaded {len(cen):,} centrals + {len(sat):,} satellites = {len(x):,} galaxies')
    return {'x': x, 'y': y, 'z': z, 'ncen_total': len(cen), 'nsat_total': len(sat)}


def save_jackknife_catalog(positions_with_region, path):
    """Schema (x, y, z, jn_region) confirmed directly from JackKnife.add_jackknife_regions()."""
    t = Table({
        'x':         positions_with_region[:, 0],
        'y':         positions_with_region[:, 1],
        'z':         positions_with_region[:, 2],
        'jn_region': positions_with_region[:, 3].astype(int),
    })
    path = Path(path)
    if path.exists():
        path.unlink()
    t.write(path, format='fits')
    print(f'Saved {len(t):,} rows, {len(np.unique(t["jn_region"]))} regions -> {path}')
    return path


def load_jackknife_catalog(path):
    """Read back x,y,z,jn_region as a plain (N,4) array — for JackknifeCovariance to consume."""
    t = Table.read(path)
    data = np.column_stack([
        np.asarray(t['x']), np.asarray(t['y']), np.asarray(t['z']), np.asarray(t['jn_region']),
    ])
    print(f'Loaded {len(t):,} rows, {len(np.unique(data[:,3]))} regions <- {path}')
    return data


def save_wp_fiducial(rp, wp, path):
    np.savez(path, rp=rp, wp=wp)
    print(f'Saved fiducial wp -> {path}')
    return path


def save_wp_jackknife(rp, wp_mean, cov, diag_err, path, **extra_meta):
    np.savez(path, rp=rp, wp_mean=wp_mean, cov=cov, diag_err=diag_err, **extra_meta)
    print(f'Saved jackknife wp + covariance -> {path}')
    return path