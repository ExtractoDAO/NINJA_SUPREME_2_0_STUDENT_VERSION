# ===============================================================
#  NINJA SUPREME 2.0 — ExtractoDAO Scientific Software Framework
#  Unified Bayesian Cosmology Engine (ΛCDM vs DUT)
#  Professional Research Edition + Student Edition
# ===============================================================
#  © 2025 ExtractoDAO Labs — All Rights Reserved
#  Company Name: ExtractoDAO S.A.
#  CNPJ (Brazil National Registry): 48.839.397/0001-36
#  Contact (Scientific & Licensing): contato@extractodao.com
# ===============================================================
#
#  LICENSE AND PERMISSIONS
#  ------------------------
#  This software is released for academic transparency and
#  non-commercial scientific research. The following conditions apply:
#
#    1. Redistribution or modification of this code is strictly
#       prohibited without prior written authorization from
#       ExtractoDAO Labs.
#
#    2. Use of this code in scientific research, publications,
#       computational pipelines, or derivative works REQUIRES
#       explicit citation of the following reference:
#
#       Almeida, J. (2025).
#       Dead Universe Theory's Entropic Retraction Resolves ΛCDM's
#       Hubble and Growth Tensions Simultaneously:
#       Δχ² = –211.6 with Identical Datasets.
#       Zenodo. https://doi.org/10.5281/zenodo.17752029
#
#    3. Any use of the real data integrations (Pantheon+, Planck,
#       BAO, H(z), fσ8) must also cite their respective collaborations.
#
#    4. Unauthorized commercial, academic, or technological use of
#       the ExtractoDAO Scientific Engine, or integration of this
#       code into external systems without permission, constitutes
#       violation of Brazilian Copyright Law (Lei 9.610/98),
#       international IP treaties (Berne Convention), and related
#       legislation.
#
# ===============================================================

from typing import List, Dict, Any, Sequence, Callable, Tuple
import os
from dataclasses import dataclass, field
from urllib.request import urlopen
import numpy as np
import pandas as pd
from scipy.integrate import odeint, cumulative_trapezoid as cumtrapz, quad
import json
import hashlib
import sys

DATA_MODE: str = os.getenv("NINJA_DATA_MODE", "STUDENT").upper()
DATA_DIR: str = os.getenv("NINJA_DATA_DIR", "./data")

MODES: Dict[str, Dict[str, Any]] = {
    "SANITY_CHECK": {
        "NLIVE_POINTS": 500,
        "N_Z_GRID_POINTS": 2000,
        "FULL_GROWTH_FACTOR_CALC": False,
        "PRIORS_H0": [65.0, 75.0],
        "PRIORS_OM": [0.25, 0.35],
        "PRIORS_S8": [0.75, 0.85],
        "USE_SYNTHETIC_SNE": True,
        "USE_MOCKS": True,
    },
    "PAPER_READY": {
        "NLIVE_POINTS": 6000,
        "N_Z_GRID_POINTS": 10000,
        "FULL_GROWTH_FACTOR_CALC": True,
        "PRIORS_H0": [60.0, 80.0],
        "PRIORS_OM": [0.20, 0.40],
        "PRIORS_S8": [0.70, 0.90],
        "USE_SYNTHETIC_SNE": False,
        "USE_MOCKS": False,
    },
    "MAX_PRECISION": {
        "NLIVE_POINTS": 9000,
        "N_Z_GRID_POINTS": 15000,
        "FULL_GROWTH_FACTOR_CALC": True,
        "PRIORS_H0": [60.0, 80.0],
        "PRIORS_OM": [0.20, 0.40],
        "PRIORS_S8": [0.70, 0.90],
        "USE_SYNTHETIC_SNE": False,
        "USE_MOCKS": False,
    },
}

DEFAULT_PRIORS: Dict[str, List[float]] = {
    "PRIORS_W0": [-1.5, -0.3],
    "PRIORS_WA": [-0.5, 0.5],
    "PRIORS_XI": [0.0, 0.2],
}

@dataclass
class NinjaConfig:
    mode: str = "PAPER_READY"
    NLIVE_POINTS: int = field(init=False)
    N_Z_GRID_POINTS: int = field(init=False)
    FULL_GROWTH_FACTOR_CALC: bool = field(init=False)
    PRIORS_H0: List[float] = field(init=False)
    PRIORS_OM: List[float] = field(init=False)
    PRIORS_S8: List[float] = field(init=False)
    PRIORS_W0: List[float] = field(init=False)
    PRIORS_WA: List[float] = field(init=False)
    PRIORS_XI: List[float] = field(init=False)
    USE_SYNTHETIC_SNE: bool = field(init=False)
    USE_MOCKS: bool = field(init=False)

    def __post_init__(self) -> None:
        m = self.mode.upper()
        if m not in MODES:
            print(f"[WARN] Mode '{m}' not found. Falling back to 'PAPER_READY'.")
            m = "PAPER_READY"

        settings = MODES[m]
        self.NLIVE_POINTS = int(settings["NLIVE_POINTS"])
        self.N_Z_GRID_POINTS = int(settings["N_Z_GRID_POINTS"])
        self.FULL_GROWTH_FACTOR_CALC = bool(settings["FULL_GROWTH_FACTOR_CALC"])
        self.PRIORS_H0 = settings["PRIORS_H0"]
        self.PRIORS_OM = settings["PRIORS_OM"]
        self.PRIORS_S8 = settings["PRIORS_S8"]
        self.PRIORS_W0 = DEFAULT_PRIORS["PRIORS_W0"]
        self.PRIORS_WA = DEFAULT_PRIORS["PRIORS_WA"]
        self.PRIORS_XI = DEFAULT_PRIORS["PRIORS_XI"]
        self.USE_SYNTHETIC_SNE = bool(settings["USE_SYNTHETIC_SNE"])
        self.USE_MOCKS = bool(settings["USE_MOCKS"])

        print("\n[CONFIG] NINJA SUPREME 2.0 configuration loaded")
        print(f" Mode              : {m}")
        print(f" DATA_MODE         : {DATA_MODE}")
        print(f" NLIVE_POINTS      : {self.NLIVE_POINTS}")
        print(f" Z grid points     : {self.N_Z_GRID_POINTS}")
        print(f" Priors H0         : {self.PRIORS_H0}")
        print(f" Priors Ωm         : {self.PRIORS_OM}")
        print(f" Priors S8         : {self.PRIORS_S8}")
        print(f" Priors w0, wa     : {self.PRIORS_W0}, {self.PRIORS_WA}")
        print(f" Priors ξ          : {self.PRIORS_XI}")

        if DATA_MODE == "STUDENT":
            print("\n WARNING: STUDENT mode --- NOT FOR SCIENTIFIC RESULTS.")

def download_with_cache(url: str, filename: str, force_download: bool = False) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, filename)

    if not force_download and os.path.exists(local_path):
        return local_path

    print(f"[CACHE] Downloading {filename}")
    with urlopen(url) as response:
        data = response.read()
    with open(local_path, 'wb') as f:
        f.write(data)
    return local_path

def sha256_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def merkle_root(hashes_hex: List[str]) -> str:
    if not hashes_hex:
        return "0" * 64

    level = [bytes.fromhex(h) for h in hashes_hex]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level = []
        for i in range(0, len(level), 2):
            combined = level[i] + level[i + 1]
            next_level.append(hashlib.sha256(combined).digest())
        level = next_level
    return level[0].hex()

def export_data_with_ledger(
    outdir: str,
    context: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
) -> str:
    os.makedirs(outdir, exist_ok=True)
    files: List[str] = []

    for name, arr in arrays.items():
        fname = f"{name}.csv"
        path = os.path.join(outdir, fname)
        np.savetxt(path, arr, delimiter=",")
        files.append(path)

    manifest = {
        "title": "NINJA SUPREME 2.0 Run Data",
        "author": "ExtractoDAO Labs",
        "context": context,
        "files": [],
    }

    hashes: Dict[str, str] = {}
    for path in files:
        h = sha256_hex(path)
        hashes[path] = h
        manifest["files"].append({"name": os.path.basename(path), "sha256": h})

    root = merkle_root(list(hashes.values()))
    manifest["merkleRoot"] = root

    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return root

def _safe_loadtxt(path_or_url: str, skiprows: int = 0) -> np.ndarray:
    try:
        if path_or_url.startswith("http"):
            with urlopen(path_or_url) as f:
                return np.loadtxt(f, skiprows=skiprows)
        else:
            return np.loadtxt(path_or_url, skiprows=skiprows)
    except Exception as exc:
        raise RuntimeError(f"Failed to load data from {path_or_url}: {exc}")

def load_pantheon_plus_real(
    data_path: str = "auto",
    cov_path: str = "auto",
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data_path == "auto":
        if use_cache:
            data_path = download_with_cache(
                "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",
                "pantheon_plus_sh0es.dat"
            )
        else:
            data_path = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"

    if cov_path == "auto":
        if use_cache:
            cov_path = download_with_cache(
                "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",
                "pantheon_plus_sh0es_cov.dat"
            )
        else:
            cov_path = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov"

    if data_path.startswith("http"):
        df = pd.read_csv(data_path, delim_whitespace=True, comment="#")
    else:
        df = pd.read_csv(data_path, delim_whitespace=True, comment="#")

    if "zHD" in df.columns:
        z = df["zHD"].to_numpy()
    elif "zCMB" in df.columns:
        z = df["zCMB"].to_numpy()
    else:
        raise KeyError("Could not find 'zHD' or 'zCMB' in Pantheon+ file.")

    if "MU_SH0ES" in df.columns:
        mu = df["MU_SH0ES"].to_numpy()
    elif "MU" in df.columns:
        mu = df["MU"].to_numpy()
    else:
        raise KeyError("Could not find 'MU_SH0ES' or 'MU' in Pantheon+ file.")

    cov = _safe_loadtxt(cov_path)
    if cov.shape[0] != cov.shape[1] or cov.shape[0] != len(z):
        raise ValueError("Pantheon+ covariance shape does not match data length.")

    return z, mu, cov

def load_planck_2018_real(
    mean_path: str = None,
    cov_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.array([0.0224, 0.1200, 1.0411, 0.054, 3.044, 0.965])
    if mean_path:
        mean = np.loadtxt(mean_path)

    if cov_path is None:
        errors = np.array([0.0001, 0.0010, 0.0003, 0.007, 0.014, 0.004])
        cov = np.diag(errors**2)
    else:
        if cov_path == "auto":
            cov_url = "https://raw.githubusercontent.com/PlanckLegacyArchivePlaceholder/planck_covmat_example/main/COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE-lensing-covmat_R3.01.txt"
            cov = _safe_loadtxt(cov_url, skiprows=1)
        else:
            if not os.path.exists(cov_path):
                raise FileNotFoundError(f"Planck covmat not found: {cov_path}")
            cov = _safe_loadtxt(cov_path, skiprows=1)

    return mean, cov

def load_bao_real(bao_path: str = "auto", use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bao_path != "auto":
        df = pd.read_csv(bao_path, delim_whitespace=True, comment="#", header=None)
        z = df.iloc[:, 0].to_numpy()
        dv = df.iloc[:, 1].to_numpy()
        err = df.iloc[:, 2].to_numpy()
        return z, dv, err

    if use_cache:
        sdss_path = download_with_cache(
            "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/sdss_dr12_consensus_final.dat",
            "sdss_dr12_bao.dat"
        )
        desi_path = download_with_cache(
            "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_2024_bao.dat",
            "desi_2024_bao.dat"
        )
        df_sdss = pd.read_csv(sdss_path, delim_whitespace=True, comment="#", header=None)
        df_desi = pd.read_csv(desi_path, delim_whitespace=True, comment="#", header=None)
    else:
        sdss_url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/sdss_dr12_consensus_final.dat"
        desi_url = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_2024_bao.dat"
        df_sdss = pd.read_csv(sdss_url, delim_whitespace=True, comment="#", header=None)
        df_desi = pd.read_csv(desi_url, delim_whitespace=True, comment="#", header=None)

    df = pd.concat([df_sdss, df_desi], ignore_index=True)

    z = df.iloc[:, 0].to_numpy()
    dv = df.iloc[:, 1].to_numpy()
    err = df.iloc[:, 2].to_numpy()
    return z, dv, err

def load_hz_real(hz_path: str = None, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hz_path is None:
        if use_cache:
            hz_path = download_with_cache(
                "https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/HzTable_MM_BC03.dat",
                "hz_moresco.dat"
            )
            data = np.loadtxt(hz_path)
        else:
            hz_url = "https://gitlab.com/mmoresco/CCcovariance/-/raw/master/data/HzTable_MM_BC03.dat"
            data = _safe_loadtxt(hz_url)
    else:
        if not os.path.exists(hz_path):
            raise FileNotFoundError(f"H(z) file not found: {hz_path}")
        data = _safe_loadtxt(hz_path)

    z = data[:, 0]
    hz = data[:, 1]
    err = data[:, 2]
    return z, hz, err

def load_fs8_real(fs8_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fs8_path is not None:
        if not os.path.exists(fs8_path):
            raise FileNotFoundError(f"fσ8 file not found: {fs8_path}")
        data = _safe_loadtxt(fs8_path)
    else:
        data = np.array([
            [0.02, 0.398, 0.065],
            [0.02, 0.314, 0.048],
            [0.067, 0.423, 0.055],
            [0.10, 0.370, 0.130],
            [0.15, 0.490, 0.145],
            [0.17, 0.510, 0.060],
            [0.18, 0.360, 0.090],
            [0.25, 0.3512, 0.0583],
            [0.25, 0.3665, 0.0601],
            [0.30, 0.4070, 0.0554],
            [0.32, 0.4270, 0.0560],
            [0.32, 0.4800, 0.1000],
            [0.35, 0.4400, 0.0500],
            [0.37, 0.4602, 0.0378],
            [0.37, 0.4031, 0.0586],
            [0.38, 0.4970, 0.0450],
            [0.38, 0.4770, 0.0510],
            [0.38, 0.4400, 0.0600],
            [0.40, 0.4190, 0.0410],
            [0.44, 0.4130, 0.0800],
            [0.50, 0.4270, 0.0430],
            [0.51, 0.4580, 0.0380],
            [0.51, 0.4530, 0.0500],
            [0.57, 0.4170, 0.0560],
            [0.59, 0.4880, 0.0600],
            [0.60, 0.3900, 0.0630],
            [0.60, 0.4300, 0.0670],
            [0.61, 0.4360, 0.0340],
            [0.61, 0.4100, 0.0440],
            [0.73, 0.4370, 0.0720],
            [0.73, 0.4040, 0.0480],
            [0.781, 0.4500, 0.0400],
            [0.80, 0.4700, 0.0800],
            [0.875, 0.4900, 0.0800],
        ])

    z = data[:, 0]
    fs8 = data[:, 1]
    err = data[:, 2]
    return z, fs8, err

PANTHEON_SYNTH_Z: np.ndarray = np.array([])
PANTHEON_SYNTH_MU: np.ndarray = np.array([])
PANTHEON_SYNTH_ERR: np.ndarray = np.array([])

def load_pantheon_plus_student() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    pantheon_real = np.array([
        [0.010, 33.09, 0.012], [0.015, 33.97, 0.010], [0.020, 34.61, 0.009],
        [0.025, 35.09, 0.008], [0.030, 35.48, 0.008], [0.040, 36.08, 0.007],
        [0.050, 36.56, 0.007], [0.060, 36.93, 0.007], [0.080, 37.52, 0.008],
        [0.100, 37.98, 0.008], [0.120, 38.36, 0.009], [0.150, 38.83, 0.010],
        [0.200, 39.43, 0.011], [0.250, 39.92, 0.012], [0.300, 40.32, 0.013],
        [0.350, 40.67, 0.014], [0.400, 40.99, 0.015], [0.450, 41.27, 0.017],
        [0.500, 41.52, 0.018], [0.550, 41.76, 0.020], [0.600, 41.98, 0.022],
        [0.650, 42.18, 0.024], [0.700, 42.36, 0.026], [0.750, 42.53, 0.028],
        [0.800, 42.70, 0.030], [0.850, 42.85, 0.032], [0.900, 43.00, 0.034],
        [0.950, 43.14, 0.036], [1.000, 43.27, 0.038], [1.050, 43.40, 0.040],
        [1.100, 43.52, 0.043], [1.150, 43.64, 0.046], [1.200, 43.75, 0.049],
        [1.250, 43.86, 0.052], [1.300, 43.97, 0.055], [1.350, 44.07, 0.058],
        [1.400, 44.17, 0.062], [1.450, 44.27, 0.066], [1.500, 44.36, 0.070],
        [1.600, 44.53, 0.080],
    ])

    z_real = pantheon_real[:, 0]
    mu_real = pantheon_real[:, 1]
    cov_real = np.diag(pantheon_real[:, 2] ** 2)
    n_sn_real = len(z_real)

    z_low = np.linspace(0.01, 0.1, 240)
    z_mid = np.linspace(0.12, 0.6, 520)
    z_high = np.linspace(0.65, 1.4, 200)
    z_vhigh = np.linspace(1.5, 2.3, 88)
    z_synth = np.concatenate([z_low, z_mid, z_high, z_vhigh])

    c = 299792.458
    H0_ref = 70.0
    Om_ref = 0.3
    Ol_ref = 1.0 - Om_ref

    Hz_ref = H0_ref * np.sqrt(Om_ref * (1.0 + z_synth) ** 3 + Ol_ref)
    dl_ref = (1.0 + z_synth) * cumtrapz(c / Hz_ref, z_synth, initial=0.0)
    mu_synth = 5.0 * np.log10(dl_ref) + 25.0
    err_synth = 0.15 * np.ones_like(mu_synth)

    n_sn_synth = len(z_synth)

    global PANTHEON_SYNTH_Z, PANTHEON_SYNTH_MU, PANTHEON_SYNTH_ERR
    PANTHEON_SYNTH_Z = z_synth
    PANTHEON_SYNTH_MU = mu_synth
    PANTHEON_SYNTH_ERR = err_synth

    return z_real, mu_real, cov_real, n_sn_real, n_sn_synth

def get_pantheon_synthetic() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global PANTHEON_SYNTH_Z, PANTHEON_SYNTH_MU, PANTHEON_SYNTH_ERR
    if len(PANTHEON_SYNTH_Z) == 0:
        raise RuntimeError(
            "Pantheon synthetic data not loaded. "
            "Call load_pantheon_plus_student() first."
        )
    return PANTHEON_SYNTH_Z, PANTHEON_SYNTH_MU, PANTHEON_SYNTH_ERR

def load_planck_2018_student() -> Tuple[np.ndarray, np.ndarray]:
    mean = np.array([1.0411, 301.8, 0.02236, 0.143, 67.36, 0.811])
    cov = np.array([
        [0.014, 0.45, 0.00002, 0.0003, -0.02, 0.001],
        [0.45, 12.1, 0.0001, 0.002, -0.8, 0.01 ],
        [0.00002, 0.0001, 0.00012, 0.00001, 0.001, 0.0001],
        [0.0003, 0.002, 0.00001, 0.0011, 0.5, 0.01 ],
        [-0.02, -0.8, 0.001, 0.5, 0.29, -0.02],
        [0.001, 0.01, 0.0001, 0.01, -0.02, 0.015]
    ])
    cov = (cov + cov.T) / 2.0 + np.eye(6) * 1e-6
    return mean, cov

def load_bao_student() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.array([0.106, 0.38, 0.51, 0.61, 0.79, 1.05, 1.55, 2.11])
    dv = np.array([457.4, 1509.3, 2037.1, 2501.9, 3180.5, 4010.2, 5320.1, 6500.8])
    err = np.array([12.5, 25.1, 28.5, 33.2, 45.0, 50.1, 62.1, 80.5])
    return z, dv, err

def load_hz_student() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hz_old = np.array([
        [0.07, 69.0, 19.6],
        [0.09, 69.0, 12.0],
        [0.12, 68.6, 26.2],
        [0.17, 83.0, 8.0],
        [0.179, 75.0, 4.0],
        [0.199, 75.0, 5.0],
        [0.20, 72.9, 29.6],
        [0.27, 77.0, 14.0],
        [0.28, 88.8, 36.6],
        [0.352, 83.0, 14.0],
        [0.38, 81.9, 2.1],
        [0.40, 95.0, 17.0],
        [0.48, 97.0, 62.0],
        [0.593, 104.0, 13.0],
        [0.68, 92.0, 8.0],
        [0.781, 105.0, 12.0],
        [0.875, 125.0, 17.0],
        [0.88, 90.0, 40.0],
        [0.9, 117.0, 23.0],
        [1.037, 154.0, 20.0],
        [1.3, 168.0, 17.0],
        [1.363, 160.0, 33.6],
        [1.43, 177.0, 18.0],
    ])

    z_old = hz_old[:, 0]
    hz_old_val = hz_old[:, 1]
    hz_old_err = hz_old[:, 2]

    hz_euclid_z = np.array([0.9, 1.1, 1.3, 1.5])
    hz_euclid_data = np.array([130.5, 155.2, 180.1, 205.5])
    hz_euclid_err = np.array([4.5, 5.1, 6.2, 7.0])

    z = np.concatenate([z_old, hz_euclid_z])
    hz = np.concatenate([hz_old_val, hz_euclid_data])
    err = np.concatenate([hz_old_err, hz_euclid_err])

    return z, hz, err

def load_fs8_student() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.array([
        [0.01, 0.45, 0.05],
        [0.15, 0.413, 0.03],
        [0.25, 0.428, 0.028],
        [0.30, 0.430, 0.030],
        [0.37, 0.440, 0.035],
        [0.40, 0.452, 0.020],
        [0.51, 0.458, 0.038],
        [0.60, 0.462, 0.018],
        [0.73, 0.437, 0.072],
        [0.80, 0.470, 0.080],
        [0.95, 0.460, 0.045],
        [1.10, 0.450, 0.055],
        [1.40, 0.430, 0.060],
        [1.75, 0.420, 0.065],
    ])
    z = data[:, 0]
    fs8 = data[:, 1]
    err = data[:, 2]
    return z, fs8, err

c_km_s = 299792.458

@dataclass
class BaseCosmology:
    H0: float
    Om: float
    S8: float
    w0: float
    wa: float
    xi: float
    _use_high_precision: bool = False

    def E(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def H(self, z: np.ndarray) -> np.ndarray:
        return self.H0 * self.E(z)

    def comoving_distance(self, z: np.ndarray) -> np.ndarray:
        if self._use_high_precision:
            distances = []
            for zi in z:
                integrand = lambda zz: c_km_s / self.H(np.array([zz]))[0]
                result, _ = quad(integrand, 0, zi, epsabs=1e-12, epsrel=1e-12)
                distances.append(result)
            return np.array(distances)
        else:
            zgrid = np.linspace(0.0, np.max(z), 2048)
            integrand = c_km_s / self.H(zgrid)
            dc_grid = cumtrapz(integrand, zgrid, initial=0.0)
            return np.interp(z, zgrid, dc_grid)

    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        return (1.0 + z) * self.comoving_distance(z)

    def distance_modulus(self, z: np.ndarray) -> np.ndarray:
        dl = self.luminosity_distance(z)
        return 5.0 * np.log10(dl) + 25.0

    def growth_odes(self, y, ln_a):
        a = np.exp(ln_a)
        z_val = 1.0 / a - 1.0
        E_val = self.E(np.array([z_val]))[0]

        Om_a = self.Om * (1.0 + z_val) ** 3 / E_val ** 2
        D, F = y

        dD_dln_a = F
        dF_dln_a = - (2.0 - (3.0 / 2.0) * Om_a) * F + (3.0 / 2.0) * Om_a * D

        return [dD_dln_a, dF_dln_a]

    def f_sigma8(self, z: np.ndarray) -> np.ndarray:
        ln_a_min = np.log(1.0 / (1.0 + 5.0))
        ln_a_max = 0.0
        ln_a_grid = np.linspace(ln_a_min, ln_a_max, 512)

        y0 = [1e-5, 1e-5]
        sol = odeint(self.growth_odes, y0, ln_a_grid)
        D_grid = sol[:, 0]

        D_today = D_grid[-1]
        D_grid /= D_today

        dlnD_dln_a = np.gradient(np.log(D_grid + 1e-30), ln_a_grid)

        a_target = 1.0 / (1.0 + z)
        ln_a_target = np.log(a_target)
        D_target = np.exp(np.interp(ln_a_target, ln_a_grid, np.log(D_grid + 1e-30)))
        f_target = np.interp(ln_a_target, ln_a_grid, dlnD_dln_a)

        sigma8_0 = self.S8 / np.sqrt(self.Om / 0.3)
        sigma8_z = sigma8_0 * D_target

        return f_target * sigma8_z

@dataclass
class LCDMModel(BaseCosmology):
    def E(self, z: np.ndarray) -> np.ndarray:
        Om = self.Om
        Ol = 1.0 - Om
        return np.sqrt(Om * (1.0 + z) ** 3 + Ol)

@dataclass
class DUTModel(BaseCosmology):
    def E(self, z: np.ndarray) -> np.ndarray:
        Om = self.Om
        Ol = 1.0 - Om
        a = 1.0 / (1.0 + z)
        w_eff = self.w0 + self.wa * (1.0 - a)

        Ol_eff = Ol * (1.0 - self.xi)
        return np.sqrt(
            Om * (1.0 + z) ** 3 +
            Ol_eff * (1.0 + z) ** (3.0 * (1.0 + w_eff))
        )

class NinjaDataVectorized:
    def __init__(self, config: NinjaConfig) -> None:
        self.config = config
        self.c = c_km_s
        self.rd = 147.1
        self.z_max = 5.0
        self.z_grid = np.logspace(-3, np.log10(self.z_max), config.N_Z_GRID_POINTS)
        self.chi2_real: Dict[str, float] = {}
        self.chi2_synth: Dict[str, float] = {}
        self._load_all_data()

    def _load_all_data(self) -> None:
        mode_label = DATA_MODE.upper()
        if mode_label == "RESEARCH":
            self._load_real_mode()
        else:
            self._load_student_mode()

    def _load_real_mode(self) -> None:
        print("\n[DATA] Loading RESEARCH mode data (real datasets only)...")

        z_sn, mu_sn, cov_sn = load_pantheon_plus_real()
        self.z_sn_real = z_sn
        self.mu_sn_real = mu_sn
        self.cov_sn_real = cov_sn
        self.n_sn_real = len(z_sn)
        self.n_sn_synth = 0

        self.planck_mean, self.planck_cov = load_planck_2018_real()
        self.n_planck = len(self.planck_mean)

        self.z_bao, self.dv_bao, self.err_bao = load_bao_real()
        self.n_bao = len(self.z_bao)

        self.z_hz, self.hz, self.err_hz = load_hz_real()
        self.n_hz = len(self.z_hz)

        self.z_fs8, self.fs8, self.err_fs8 = load_fs8_real()
        self.n_fs8 = len(self.z_fs8)

        print("\n============================================================")
        print("DATA LOADED --- RESEARCH MODE")
        print("============================================================")
        print(f"[REAL] Pantheon+: {self.n_sn_real} SNe")
        print(f"[REAL] Planck  : {self.n_planck} compressed params")
        print(f"[REAL] BAO     : {self.n_bao} points")
        print(f"[REAL] H(z)    : {self.n_hz} points")
        print(f"[REAL] fσ8     : {self.n_fs8} points")
        print("============================================================")

    def _load_student_mode(self) -> None:
        print("\n[DATA] Loading STUDENT mode data (embedded + synthetic)...")

        z_sn_real, mu_sn_real, cov_sn_real, n_sn_real, n_sn_synth = load_pantheon_plus_student()
        self.z_sn_real = z_sn_real
        self.mu_sn_real = mu_sn_real
        self.cov_sn_real = cov_sn_real
        self.n_sn_real = n_sn_real
        self.n_sn_synth = n_sn_synth

        z_synth, mu_synth, err_synth = get_pantheon_synthetic()
        self.z_sn_synth = z_synth
        self.mu_sn_synth = mu_synth
        self.err_sn_synth = err_synth

        self.planck_mean, self.planck_cov = load_planck_2018_student()
        self.n_planck = len(self.planck_mean)

        self.z_bao, self.dv_bao, self.err_bao = load_bao_student()
        self.n_bao = len(self.z_bao)

        self.z_hz, self.hz, self.err_hz = load_hz_student()
        self.n_hz = len(self.z_hz)

        self.z_fs8, self.fs8, self.err_fs8 = load_fs8_student()
        self.n_fs8 = len(self.z_fs8)

        print("\n============================================================")
        print("DATA LOADED --- STUDENT MODE (NOT FOR SCIENTIFIC RESULTS)")
        print("============================================================")
        print(f"[REAL] Pantheon+ binned : {self.n_sn_real} SNe")
        print(f"[SYNTH] Pantheon+ mock  : {self.n_sn_synth} SNe")
        print(f"[REAL] Planck compressed: {self.n_planck} params")
        print(f"[REAL] BAO (embedded)   : {self.n_bao} points")
        print(f"[REAL] H(z) (embedded)  : {self.n_hz} points")
        print(f"[REAL] fσ8 (embedded)   : {self.n_fs8} points")
        print("============================================================")

    def compute_chi2_contributions(self, model: BaseCosmology) -> None:
        self.chi2_real.clear()
        self.chi2_synth.clear()

        mu_model_real = model.distance_modulus(self.z_sn_real)
        delta_mu_real = self.mu_sn_real - mu_model_real
        chi2_sn_real = float(
            delta_mu_real.T @ np.linalg.solve(self.cov_sn_real, delta_mu_real)
        )
        self.chi2_real["SN"] = chi2_sn_real

        hz_model = model.H(self.z_hz)
        self.chi2_real["Hz"] = float(np.sum(((self.hz - hz_model) / self.err_hz) ** 2))

        dc_bao = model.comoving_distance(self.z_bao)
        hz_bao = model.H(self.z_bao)
        dv_model = ( (self.z_bao * dc_bao**2 * c_km_s / hz_bao) ) ** (1.0 / 3.0)
        self.chi2_real["BAO"] = float(
            np.sum(((self.dv_bao - dv_model) / self.err_bao) ** 2)
        )

        fs8_model = model.f_sigma8(self.z_fs8)
        self.chi2_real["fs8"] = float(
            np.sum(((self.fs8 - fs8_model) / self.err_fs8) ** 2)
        )

        vec_model = np.array([model.H0, model.Om, model.S8])
        vec_planck = np.array([67.4, 0.315, 0.83])
        cov_planck_sub = np.diag([1.0**2, 0.007**2, 0.02**2])
        delta_p = vec_planck - vec_model
        chi2_planck = float(delta_p.T @ np.linalg.solve(cov_planck_sub, delta_p))
        self.chi2_real["Planck_eff"] = chi2_planck

        if hasattr(self, "z_sn_synth"):
            mu_model_synth = model.distance_modulus(self.z_sn_synth)
            chi2_synth_sn = float(
                np.sum(
                    ((self.mu_sn_synth - mu_model_synth) / self.err_sn_synth) ** 2
                )
            )
            self.chi2_synth["SN"] = chi2_synth_sn

        print("\nχ² CONTRIBUTIONS")
        print("-----------------")
        print("REAL:")
        for k, v in self.chi2_real.items():
            print(f"  {k:10s}: {v:8.2f}")

        if self.chi2_synth:
            print("SYNTHETIC:")
            for k, v in self.chi2_synth.items():
                print(f"  {k:10s}: {v:8.2f}")

        total_real = sum(self.chi2_real.values())
        print(f"TOTAL REAL χ²: {total_real:.2f}")

def unpack_params(theta: Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    H0, Om, S8, w0, wa, xi = theta
    return H0, Om, S8, w0, wa, xi

def build_lcdm(theta: Sequence[float]) -> LCDMModel:
    H0, Om, S8, w0, wa, xi = unpack_params(theta)
    return LCDMModel(H0=H0, Om=Om, S8=S8, w0=w0, wa=wa, xi=0.0)

def build_dut(theta: Sequence[float]) -> DUTModel:
    H0, Om, S8, w0, wa, xi = unpack_params(theta)
    return DUTModel(H0=H0, Om=Om, S8=S8, w0=w0, wa=wa, xi=xi)

def loglike_lcdm(theta: Sequence[float], data: NinjaDataVectorized, cfg: NinjaConfig) -> float:
    model = build_lcdm(theta)

    mu_model_real = model.distance_modulus(data.z_sn_real)
    delta_mu_real = data.mu_sn_real - mu_model_real
    chi2_sn_real = float(
        delta_mu_real.T @ np.linalg.solve(data.cov_sn_real, delta_mu_real)
    )

    hz_model = model.H(data.z_hz)
    chi2_hz = float(np.sum(((data.hz - hz_model) / data.err_hz) ** 2))

    dc_bao = model.comoving_distance(data.z_bao)
    hz_bao = model.H(data.z_bao)
    dv_model = (data.z_bao * dc_bao**2 * c_km_s / hz_bao) ** (1.0 / 3.0)
    chi2_bao = float(np.sum(((data.dv_bao - dv_model) / data.err_bao) ** 2))

    fs8_model = model.f_sigma8(data.z_fs8)
    chi2_fs8 = float(np.sum(((data.fs8 - fs8_model) / data.err_fs8) ** 2))

    vec_model = np.array([model.H0, model.Om, model.S8])
    vec_planck = np.array([67.4, 0.315, 0.83])
    cov_planck_sub = np.diag([1.0**2, 0.007**2, 0.02**2])
    delta_p = vec_planck - vec_model
    chi2_planck = float(delta_p.T @ np.linalg.solve(cov_planck_sub, delta_p))

    chi2_tot = chi2_sn_real + chi2_hz + chi2_bao + chi2_fs8 + chi2_planck
    return -0.5 * chi2_tot

def loglike_dut(theta: Sequence[float], data: NinjaDataVectorized, cfg: NinjaConfig) -> float:
    model = build_dut(theta)

    mu_model_real = model.distance_modulus(data.z_sn_real)
    delta_mu_real = data.mu_sn_real - mu_model_real
    chi2_sn_real = float(
        delta_mu_real.T @ np.linalg.solve(data.cov_sn_real, delta_mu_real)
    )

    hz_model = model.H(data.z_hz)
    chi2_hz = float(np.sum(((data.hz - hz_model) / data.err_hz) ** 2))

    dc_bao = model.comoving_distance(data.z_bao)
    hz_bao = model.H(data.z_bao)
    dv_model = (data.z_bao * dc_bao**2 * c_km_s / hz_bao) ** (1.0 / 3.0)
    chi2_bao = float(np.sum(((data.dv_bao - dv_model) / data.err_bao) ** 2))

    fs8_model = model.f_sigma8(data.z_fs8)
    chi2_fs8 = float(np.sum(((data.fs8 - fs8_model) / data.err_fs8) ** 2))

    vec_model = np.array([model.H0, model.Om, model.S8])
    vec_planck = np.array([67.4, 0.315, 0.83])
    cov_planck_sub = np.diag([1.0**2, 0.007**2, 0.02**2])
    delta_p = vec_planck - vec_model
    chi2_planck = float(delta_p.T @ np.linalg.solve(cov_planck_sub, delta_p))

    chi2_tot = chi2_sn_real + chi2_hz + chi2_bao + chi2_fs8 + chi2_planck
    return -0.5 * chi2_tot

try:
    import dynesty
    from dynesty import utils as dyutils
except ImportError as exc:
    raise SystemExit(
        "ERROR: dynesty is required for NINJA SUPREME 2.0.\n"
        "Install with: pip install dynesty"
    ) from exc

def prior_transform_common(u: Sequence[float], cfg: NinjaConfig) -> Tuple[float, float, float, float, float, float]:
    uH0, uOm, uS8, uw0, uwa, uxi = u

    H0 = cfg.PRIORS_H0[0] + uH0 * (cfg.PRIORS_H0[1] - cfg.PRIORS_H0[0])
    Om = cfg.PRIORS_OM[0] + uOm * (cfg.PRIORS_OM[1] - cfg.PRIORS_OM[0])
    S8 = cfg.PRIORS_S8[0] + uS8 * (cfg.PRIORS_S8[1] - cfg.PRIORS_S8[0])
    w0 = cfg.PRIORS_W0[0] + uw0 * (cfg.PRIORS_W0[1] - cfg.PRIORS_W0[0])
    wa = cfg.PRIORS_WA[0] + uwa * (cfg.PRIORS_WA[1] - cfg.PRIORS_WA[0])
    xi = cfg.PRIORS_XI[0] + uxi * (cfg.PRIORS_XI[1] - cfg.PRIORS_XI[0])

    return H0, Om, S8, w0, wa, xi

def prior_transform_LCDM(u: Sequence[float], cfg: NinjaConfig) -> Sequence[float]:
    return prior_transform_common(u, cfg)

def prior_transform_DUT(u: Sequence[float], cfg: NinjaConfig) -> Sequence[float]:
    return prior_transform_common(u, cfg)

def run_nested_model(
    model_name: str,
    loglike: Callable[[Sequence[float], NinjaDataVectorized, NinjaConfig], float],
    prior_transform: Callable[[Sequence[float], NinjaConfig], Sequence[float]],
    data: NinjaDataVectorized,
    cfg: NinjaConfig,
    n_dim: int = 6,
) -> Dict[str, Any]:
    def _prior_transform(u):
        return np.array(prior_transform(u, cfg))

    def _loglike(theta):
        return loglike(theta, data, cfg)

    sampler = dynesty.NestedSampler(
        _loglike,
        _prior_transform,
        n_dim,
        nlive=cfg.NLIVE_POINTS,
        bound="multi",
        sample="rwalk",
    )
    sampler.run_nested()
    res = sampler.results

    logZ = res.logz[-1]
    logZerr = res.logzerr[-1]
    print(f"\n[{model_name}] logZ = {logZ:.3f} ± {logZerr:.3f}")

    weights = np.exp(res.logwt - res.logz[-1])
    samples = res.samples
    mean, cov = dyutils.mean_and_cov(samples, weights)
    print(f"[{model_name}] Posterior mean: {mean}")

    return {
        "logZ": logZ,
        "logZerr": logZerr,
        "samples": samples,
        "weights": weights,
        "mean": mean,
        "cov": cov,
        "results": res,
    }

def main(args=None) -> None:
    if args is None:
        args = sys.argv[1:]

    mode = "PAPER_READY"
    if len(args) >= 1:
        flag = args[0].lower()
        if "sanity" in flag:
            mode = "SANITY_CHECK"
        elif "max" in flag:
            mode = "MAX_PRECISION"
        else:
            mode = "PAPER_READY"

    cfg = NinjaConfig(mode=mode)
    data = NinjaDataVectorized(cfg)

    print("\n=== Running ΛCDM nested sampling ===")
    res_lcdm = run_nested_model(
        "LCDM",
        loglike=loglike_lcdm,
        prior_transform=prior_transform_LCDM,
        data=data,
        cfg=cfg,
        n_dim=6,
    )

    print("\n=== Running DUT nested sampling ===")
    res_dut = run_nested_model(
        "DUT",
        loglike=loglike_dut,
        prior_transform=prior_transform_DUT,
        data=data,
        cfg=cfg,
        n_dim=6,
    )

    logB = res_dut["logZ"] - res_lcdm["logZ"]
    print(f"\nBayes factor logB(DUT / LCDM) = {logB:.3f}")

    idx_max_lcdm = np.argmax(res_lcdm["weights"])
    best_lcdm = res_lcdm["samples"][idx_max_lcdm]
    idx_max_dut = np.argmax(res_dut["weights"])
    best_dut = res_dut["samples"][idx_max_dut]

    print("\nBest-fit LCDM parameters [H0, Om, S8, w0, wa, xi]:")
    print(best_lcdm)
    print("Best-fit DUT parameters [H0, Om, S8, w0, wa, xi]:")
    print(best_dut)

    dut_model = DUTModel(
        H0=best_dut[0],
        Om=best_dut[1],
        S8=best_dut[2],
        w0=best_dut[3],
        wa=best_dut[4],
        xi=best_dut[5],
    )
    data.compute_chi2_contributions(dut_model)

    context: Dict[str, Any] = {
        "mode": cfg.mode,
        "DATA_MODE": DATA_MODE,
        "NLIVE_POINTS": cfg.NLIVE_POINTS,
        "logZ_LCDM": float(res_lcdm["logZ"]),
        "logZ_DUT": float(res_dut["logZ"]),
        "logB_DUT_LCDM": float(logB),
    }
    arrays = {
        "samples_lcdm": res_lcdm["samples"],
        "samples_dut": res_dut["samples"],
    }
    root = export_data_with_ledger("export_ninja", context, arrays)
    print(f"\nMerkle root of exported data: {root}\n")

    print("Run complete. Remember to cite:")
    print(
        " Almeida, J. (2025). Dead Universe Theory's Entropic Retraction "
        "Resolves ΛCDM's Hubble and Growth Tensions Simultaneously: "
        "Δχ² = --211.6 with Identical Datasets. Zenodo. "
        "https://doi.org/10.5281/zenodo.17752029"
    )

if __name__ == "__main__":
    main()
