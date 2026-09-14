"""
config.py — single source of truth for constants used across the
derivative and Fisher/jackknife notebooks.
"""
from pathlib import Path

# ---- Halo catalogue ----
SIM_STEM = 'sahyadri/default2048'
REAL     = 1
SNAP     = 82
MASSDEF  = 'm200b'   # NOT YET VERIFIED against Chetan's actual current pipeline

EXPECTED_N_HALOS = 343_275
EXPECTED_LBOX    = 200.0
EXPECTED_OMEGA_M = 0.3137721

# ---- HOD / mock ----
MR_THRESH = -18.0
SEED      = 10
RSD       = True

# ---- wp(rp) ----
PIMAX    = 40.0
N_RP     = 50
RP_MIN   = 0.5
RP_MAX   = 30.0
NTHREADS = 4

# ---- Jackknife ----
NJN = 100
LOS = 1

# ---- Paths (Pegasus HPC) ----
POSTPROCESS_PATH = Path(
    "/mnt/home/project/cgowari.aditya/sahyadri-codes/"
    "sahyadri-sandbox/scripts/post-process"
)
SPECTROPHOTO_PATH = Path(
    "/mnt/home/project/cgowari.aditya/Programs/spectrophotogroup/Simulator/simulator.py"
)
MODIFIED_POISSON_PATH = Path(
    "/mnt/home/project/cgowari.aditya/Programs/modified_poisson"
)
HALO_FITS_BOLSHOI_PATH = Path(
    "/mnt/home/project/cgowari.aditya/sahyadri-codes/sahyadri_r1_snap100_bolshoi_schema.fits"
)

# ---- Output files — flat results/ directory, no CATALOG_DIR subfolder ----
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CENTRALS_PATH          = OUTPUT_DIR / 'centrals.fits'
SATELLITES_PATH        = OUTPUT_DIR / 'satellites.fits'
JACKKNIFE_CATALOG_PATH = OUTPUT_DIR / f'gpos_jackknife_njn{NJN}_los{LOS}.fits'
WP_FIDUCIAL_PATH       = OUTPUT_DIR / 'wp_fiducial.npz'
WP_JACKKNIFE_PATH      = OUTPUT_DIR / f'wp_jackknife_njn{NJN}.npz'

DERIVATIVES_DIR = OUTPUT_DIR / 'derivatives'
DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)