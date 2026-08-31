"""bbo.py — toolkit for the BBO capstone: eight black-box functions, maximise each.

Every function is opaque. I submit a vector of numbers in roughly [0.001, 1] and the
portal returns one scalar. One query per function per round, thirteen rounds, no
gradients and no formula.

The routine each round:

    1. load(fid)              read this function's query history
    2. ledger()               what is the BEST point known, not the latest one
    3. fit(fid)               GP surrogate: Matern-5/2 + WhiteKernel
    4. propose_ei(fid)        Expected Improvement over a 5,000-point grid
    5. propose_line(fid)      directional layer: centre-ward line + quadratic vertex
    6. submission(x)          format as dash-separated 6dp for the portal

The ledger is deliberately step 2 rather than an afterthought. The single largest
cost in this campaign was generating proposals relative to the *previous* round's
result rather than the best result on record; anchor_check() below exists so that
cannot happen silently again.

CLI:
    python bbo.py report        best-known table, final-round comparison, round ledger
    python bbo.py propose       GP+EI proposal for every function
    python bbo.py line 4        centre-ward line diagnostics for one function
    python bbo.py plot          best-so-far trajectories -> outputs/trajectories.png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# Length-scale and noise bounds are routinely hit when fitting ~12 points in up to
# eight dimensions. That is a real signal about the sample size, reported by loo_r2()
# below, and not something the per-fit warnings add anything to.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

FUNC_IDS = list(range(1, 9))
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

# The portal accepts values in roughly [0.001, 1]; F4's x3 hit the lower bound in W10.
LOW, HIGH = 0.001, 1.0

# F1's outputs span ~190 orders of magnitude with sign changes, so a stationary GP
# fitted to the raw values is meaningless. A signed, order-preserving compression
# makes it fittable; it does NOT make it trustworthy. See the model card, L5/§6.
TRANSFORM = {1: "signed_log", 2: None, 3: None, 4: None,
             5: None, 6: None, 7: None, 8: None}


# --------------------------------------------------------------------- history

# round -> {func_id: (inputs, output)}.  Rounds carrying inputs only are stored with
# an output of None and filtered out at load time.
HISTORY: dict[int, dict[int, tuple[list[float], float | None]]] = {
    1: {
        1: ([0.156843, 0.587493], -5.768712955574306e-69),
        2: ([0.156843, 0.587493], 0.008511217799492411),
        3: ([0.156843, 0.587493, 0.741585], -0.12884849611277463),
        4: ([0.156843, 0.587493, 0.741585, 0.985632], -24.426727116824043),
        5: ([0.156843, 0.587493, 0.741585, 0.985632], 566.341996544051),
        6: ([0.156843, 0.587493, 0.741585, 0.985632, 0.148524], -0.8632458340875497),
        7: ([0.156843, 0.587493, 0.741585, 0.985632, 0.148524, 0.179638], 0.08028539779449592),
        8: ([0.156843, 0.587493, 0.741585, 0.985632, 0.148524, 0.179638, 0.541596, 0.396585], 7.429392424567499),
    },
    2: {
        1: ([0.018957, 0.259878], -1.7422969321162947e-131),
        2: ([0.098559, 0.954719], 0.0032941097425224245),
        3: ([0.159998, 0.011915, 0.958587], -0.323441538925546),
        4: ([0.611147, 0.607958, 0.671974, 0.601141], -11.656144229892409),
        5: ([0.014852, 0.297741, 0.718557, 0.219953], 26.446551474663757),
        6: ([0.398747, 0.385554, 0.570014, 0.711777, 0.389141], -0.3982565809200795),
        7: ([0.151858, 0.148558, 0.071547, 0.258484, 0.285157, 0.741141], 2.4237613938590665),
        8: ([0.159174, 0.118198, 0.137956, 0.716535, 0.781515, 0.543548, 0.279585, 0.258543], 9.6444395995596),
    },
    3: {
        1: ([0.018917, 0.257418], None),
        2: ([0.098659, 0.954717], None),
        3: ([0.008917, 0.791931, 0.253392], None),
        4: ([0.313981, 0.427787, 0.451776, 0.356097], None),
        5: ([0.013742, 0.287641, 0.718987, 0.211753], None),
        6: ([0.398757, 0.385494, 0.571114, 0.711617, 0.389851], None),
        7: ([0.151868, 0.148658, 0.072557, 0.257384, 0.201157, 0.787141], None),
        8: ([0.159174, 0.118198, 0.137941, 0.716499, 0.782017, 0.543671, 0.279601, 0.258415], None),
    },
    4: {
        1: ([0.051481, 0.059635], 1.6545027412373418e-187),
        2: ([0.145789, 0.145667], 0.01484628428059747),
        3: ([0.175986, 0.176018, 0.176111], -0.1470980607233212),
        4: ([0.567001, 0.567177, 0.567222, 0.568100], -9.692519234626918),
        5: ([0.821489, 0.548961, 0.574899, 0.611175], 81.79728791398944),
        6: ([0.821489, 0.548961, 0.574899, 0.611175, 0.178585], -0.80217118935894),
        7: ([0.821489, 0.489635, 0.578511, 0.593470, 0.178585, 0.525279], 0.18430174146238856),
        8: ([0.785496, 0.489658, 0.571328, 0.593470, 0.178585, 0.525279, 0.478596, 0.019573], 7.781207717733601),
    },
    5: {
        1: ([0.048421, 0.137247], -3.8357016898973296e-154),
        2: ([0.048699, 0.137537], -0.10664658336026589),
        3: ([0.489658, 0.159666, 0.895963], -0.08523330692617447),
        4: ([0.569121, 0.443720, 0.402671, 0.283008], -4.082994951761037),
        5: ([0.275418, 0.703962, 0.058314, 0.629740], 2.379876233247608),
        6: ([0.448901, 0.916285, 0.331657, 0.205493, 0.612847], -2.000030696307407),
        7: ([0.659037, 0.842715, 0.276490, 0.501628, 0.787126, 0.394208], 0.16205712966379487),
        8: ([0.015899, 0.258569, 0.158963, 0.325259, 0.785858, 0.212159, 0.958585, 0.071899], 8.6790906222159),
    },
    6: {
        1: ([0.051203, 0.134018], -6.377968075830329e-154),
        2: ([0.046215, 0.140382], 0.054542698829435535),
        3: ([0.492341, 0.156982, 0.898517], -0.09678129158071659),
        4: ([0.566438, 0.446519, 0.399884, 0.285792], -4.15418124708534),
        5: ([0.278106, 0.701274, 0.061098, 0.632517], 2.435323698188698),
        6: ([0.451687, 0.913498, 0.334419, 0.208271, 0.610063], -2.0214828656562784),
        7: ([0.656251, 0.845498, 0.279274, 0.498842, 0.789910, 0.391421], 0.16998035132007513),
        8: ([0.018683, 0.255785, 0.161747, 0.328043, 0.783074, 0.214943, 0.955801, 0.074683], 8.6893948626831),
    },
    7: {
        1: ([0.669237, 0.752052], None),
        2: ([0.355232, 0.449399], None),
        3: ([0.110375, 0.775016, 0.516551], None),
        4: ([0.184472, 0.064553, 0.017918, 0.903826], None),
        5: ([0.308106, 0.731274, 0.091098, 0.662517], None),
        6: ([0.069721, 0.531532, 0.952453, 0.826305, 0.228097], None),
        7: ([0.965268, 0.154515, 0.588291, 0.807859, 0.098927, 0.700438], None),
        8: ([0.018683, 0.255785, 0.161747, 0.328043, 0.783074, 0.214943, 0.955801, 0.074683], None),
    },
    8: {
        1: ([0.669937, 0.751452], -7.058138619523124e-14),
        2: ([0.354432, 0.449899], 0.0836536493073266),
        3: ([0.111275, 0.774316, 0.516951], -0.01542931285755213),
        4: ([0.183872, 0.065353, 0.017618, 0.904326], -23.621861521751097),
        5: ([0.308806, 0.730874, 0.091998, 0.661917], 0.5222091523300119),
        6: ([0.070021, 0.530732, 0.952853, 0.825805, 0.228797], -0.9676493116712739),
        7: ([0.965568, 0.153915, 0.588691, 0.807159, 0.099427, 0.700138], 0.07982491545502426),
        8: ([0.038733, 0.275735, 0.181777, 0.348003, 0.803094, 0.234913, 0.935841, 0.094663], 8.7506992510951),
    },
    9: {
        1: ([0.330063, 0.371452], 1.04e-08),
        2: ([0.654432, 0.749899], 0.423255),
        3: ([0.888725, 0.225684, 0.483049], -0.035188),
        4: ([0.816128, 0.934647, 0.982382, 0.095674], -36.0907),
        5: ([0.322806, 0.716874, 0.106998, 0.647917], 0.309847),
        6: ([0.929979, 0.469268, 0.047147, 0.174195, 0.771203], -2.4701),
        7: ([0.686227, 0.361566, 0.535476, 0.622864, 0.339771, 0.580055], 0.416839),
        8: ([0.039233, 0.275235, 0.182077, 0.347603, 0.803294, 0.234613, 0.936241, 0.094463], 8.749654),
    },
    10: {
        1: ([0.150000, 0.850000], 2.8557e-181),
        2: ([0.754432, 0.849899], 0.402932),
        3: ([0.161275, 0.724316, 0.566951], -0.047062),
        4: ([0.152272, 0.021888, 0.001000, 0.944762], -27.222858),
        5: ([0.301806, 0.737874, 0.084498, 0.668917], 1.326997),
        6: ([0.052822, 0.531961, 0.970967, 0.838837, 0.217949], -1.014444),
        7: ([0.630359, 0.403096, 0.524833, 0.586005, 0.387840, 0.556038], 0.499330),
        8: ([0.038983, 0.275485, 0.181927, 0.347803, 0.803194, 0.234763, 0.936041, 0.094563], 8.750177),
    },
    11: {
        1: ([0.420000, 0.462000], -0.0036427),
        2: ([0.754032, 0.689499], 0.291066),
        3: ([0.061275, 0.824316, 0.466951], -0.038737),
        4: ([0.405162, 0.369606, 0.355285, 0.621298], -4.063130),
        5: ([0.280806, 0.758874, 0.061998, 0.689917], 7.134355),
        6: ([0.285011, 0.515366, 0.726427, 0.662903, 0.364399], -0.730692),
        7: ([0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000], 0.505315),
        8: ([0.177288, 0.342840, 0.277349, 0.393462, 0.712236, 0.314334, 0.805229, 0.216194], 9.040799),
    },
    12: {
        1: ([0.270105, 0.311087], -1.1805e-25),
        2: ([0.575432, 0.810899], 0.037313),
        3: ([0.167485, 0.838106, 0.513161], -0.047922),
        4: ([0.462508, 0.448451, 0.442790, 0.547953], -2.874597),
        5: ([0.252806, 0.786874, 0.031998, 0.717917], 24.903137),
        6: ([0.302252, 0.664134, 0.708267, 0.649836, 0.375274], -0.948038),
        7: ([0.562174, 0.453772, 0.511847, 0.541027, 0.446492, 0.526734], 0.534120),
        8: ([0.315593, 0.410194, 0.372771, 0.439121, 0.621278, 0.393905, 0.674416, 0.337825], 9.094693),
    },
    13: {
        1: ([0.487500, 0.528900], 3.140281895796733e-12),
        2: ([0.674000, 0.722000], 0.5799042634933687),
        3: ([0.383383, 0.582295, 0.505085], -0.015520611381581548),
        4: ([0.430293, 0.404078, 0.540611, 0.589237], -4.658694471813039),
        5: ([0.231806, 0.807874, 0.009498, 0.738917], 48.00775139689749),
        6: ([0.305686, 0.363889, 0.704651, 0.647233, 0.377423], -0.5789767818430653),
        7: ([0.643824, 0.372122, 0.593497, 0.459377, 0.528142, 0.445084], 0.3345754506634855),
        8: ([0.134496, 0.492931, 0.258165, 0.509239, 0.618801, 0.424815, 0.645029, 0.325630], 9.2759880860135),
    },
}

FINAL_ROUND = max(HISTORY)


# ------------------------------------------------------------------------ data

def load(fid: int, up_to: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, rounds) for one function, complete records only.

    up_to truncates the history to rounds <= up_to, so a weekly notebook can
    reconstruct exactly the data state that existed when that round was proposed.
    """
    rows = [(r, xy[0], xy[1]) for r, block in sorted(HISTORY.items())
            for f, xy in block.items()
            if f == fid and xy[1] is not None and (up_to is None or r <= up_to)]
    if not rows:
        raise ValueError(f"no records for F{fid}")
    rounds = np.array([r for r, _, _ in rows], dtype=int)
    X = np.array([x for _, x, _ in rows], dtype=float)
    y = np.array([v for _, _, v in rows], dtype=float)
    return X, y, rounds


def frame(fid: int, up_to: int | None = None) -> pd.DataFrame:
    X, y, rounds = load(fid, up_to=up_to)
    df = pd.DataFrame(X, columns=[f"x{j+1}" for j in range(X.shape[1])])
    df.insert(0, "round", rounds)
    df["y"] = y
    return df


def export_csvs(directory: str = DATA_DIR) -> list[str]:
    """Write one CSV per function so the history can be edited outside this file."""
    os.makedirs(directory, exist_ok=True)
    paths = []
    for fid in FUNC_IDS:
        p = os.path.join(directory, f"f{fid}.csv")
        frame(fid).to_csv(p, index=False)
        paths.append(p)
    return paths


def append_result(fid: int, x, y: float, rnd: int | None = None) -> None:
    """Record a new portal return in the in-memory history."""
    rnd = rnd if rnd is not None else max(HISTORY) + 1
    HISTORY.setdefault(rnd, {})[fid] = ([float(v) for v in np.ravel(x)], float(y))


# ---------------------------------------------------------------------- ledger

def ledger(up_to: int | None = None) -> pd.DataFrame:
    """Best-known point per function across the WHOLE history.

    This is the anchor every proposal should be generated from. Reading only the
    latest round instead is what cost this campaign most of its final score.
    """
    rows = []
    for fid in FUNC_IDS:
        try:
            X, y, rounds = load(fid, up_to=up_to)
        except ValueError:
            continue
        i = int(np.argmax(y))
        rows.append(dict(func=f"F{fid}", d=DIMS[fid], best=y[i],
                         round=int(rounds[i]), x=list(X[i])))
    return pd.DataFrame(rows)


def best_point(fid: int, up_to: int | None = None) -> tuple[np.ndarray, float, int]:
    X, y, rounds = load(fid, up_to=up_to)
    i = int(np.argmax(y))
    return X[i], float(y[i]), int(rounds[i])


def anchor_check(fid: int, anchor: np.ndarray, tol: float = 1e-9,
                 up_to: int | None = None) -> str | None:
    """Warn if a proposal is being generated from anything but the best-known point."""
    xb, yb, rb = best_point(fid, up_to=up_to)
    if np.allclose(np.ravel(anchor), xb, atol=tol):
        return None
    return (f"F{fid}: anchor is NOT the best-known point "
            f"(best {yb:.6g} from round {rb}). Confirm this is deliberate.")


def final_vs_best() -> pd.DataFrame:
    """Scoring is on the final submission alone, so this is the table that matters."""
    rows = []
    for fid in FUNC_IDS:
        xb, yb, rb = best_point(fid)
        yf = HISTORY[FINAL_ROUND][fid][1]
        rows.append(dict(func=f"F{fid}", final=yf, best=yb, best_round=rb,
                         matched=bool(np.isclose(yf, yb, rtol=1e-9))))
    return pd.DataFrame(rows)


def round_ledger() -> pd.DataFrame:
    """Per round: how many functions set a new best, and the running total."""
    running = {fid: -np.inf for fid in FUNC_IDS}
    rows = []
    for rnd in sorted(HISTORY):
        gains = []
        for fid in FUNC_IDS:
            rec = HISTORY[rnd].get(fid)
            if rec is None or rec[1] is None:
                continue
            if rec[1] > running[fid]:
                running[fid] = rec[1]
                gains.append(f"F{fid}")
        rows.append(dict(round=rnd, new_bests=len(gains), functions=" ".join(gains)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- transform

def _signed_log(y: np.ndarray) -> np.ndarray:
    """Order-preserving compression for F1's ~190 decades of dynamic range."""
    a = np.abs(y)
    scale = np.median(a[a > 0]) if np.any(a > 0) else 1.0
    return np.sign(y) * np.log1p(a / scale)


def apply_transform(fid: int, y: np.ndarray) -> np.ndarray:
    return _signed_log(y) if TRANSFORM.get(fid) == "signed_log" else y


# ------------------------------------------------------------------- surrogate

def build_gp(d: int, n_restarts: int = 10, seed: int = 0) -> GaussianProcessRegressor:
    """Matern-5/2 + WhiteKernel, hyperparameters by marginal likelihood.

    nu is fixed at 5/2 rather than searched: at ~12 observations there is no basis
    to select it from data.
    """
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.full(d, 0.2),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e0)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=n_restarts,
                                    random_state=seed)


class Model:
    def __init__(self, fid: int, gp, X: np.ndarray, y: np.ndarray, y_fit: np.ndarray):
        self.fid, self.gp, self.X, self.y, self.y_fit = fid, gp, X, y, y_fit
        self.d = X.shape[1]

    def predict(self, U: np.ndarray):
        mean, std = self.gp.predict(np.atleast_2d(U), return_std=True)
        return mean, std

    def loo_r2(self) -> float:
        """Leave-one-out R^2. Below zero means the surrogate is worse than the mean
        and any acquisition value computed from it is arbitrary."""
        n = len(self.y_fit)
        if n < 4:
            return float("nan")
        preds = np.empty(n)
        for i in range(n):
            m = np.ones(n, dtype=bool)
            m[i] = False
            g = build_gp(self.d)
            g.fit(self.X[m], self.y_fit[m])
            preds[i] = g.predict(self.X[i].reshape(1, -1))[0]
        ss_res = float(np.sum((self.y_fit - preds) ** 2))
        ss_tot = float(np.sum((self.y_fit - self.y_fit.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit(fid: int, seed: int = 0, up_to: int | None = None) -> Model:
    X, y, _ = load(fid, up_to=up_to)
    y_fit = apply_transform(fid, y)
    gp = build_gp(X.shape[1], seed=seed)
    gp.fit(X, y_fit)
    return Model(fid, gp, X, y, y_fit)


# ----------------------------------------------------------------- acquisition

def ei(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.05) -> np.ndarray:
    """Expected Improvement. xi = 0.05 throughout this project."""
    std = np.maximum(std, 1e-12)
    z = (mean - y_best - xi) / std
    return (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)


def propose_ei(fid: int, n_grid: int = 5000, xi: float = 0.05, seed: int = 0,
               up_to: int | None = None) -> dict:
    """GP + EI over a random grid of n_grid points. The grid size matches what was
    used across the campaign; a Sobol design would cover better at the same cost."""
    model = fit(fid, seed=seed, up_to=up_to)
    rng = np.random.default_rng(seed)
    U = rng.uniform(LOW, HIGH, size=(n_grid, model.d))
    mean, std = model.predict(U)
    vals = ei(mean, std, float(model.y_fit.max()), xi=xi)
    i = int(np.argmax(vals))
    x = U[i]
    return dict(func=fid, x=x, mean=float(mean[i]), std=float(std[i]),
                acq="ei", acq_value=float(vals[i]),
                loo_r2=model.loo_r2(), submission=submission(x),
                warning=anchor_check(fid, best_point(fid, up_to=up_to)[0], up_to=up_to))


# ------------------------------------------------------- directional heuristics

CENTRE = 0.5


def centre_line(anchor: np.ndarray, lam: float) -> np.ndarray:
    """x(lam) = anchor + lam * (c - anchor), c the domain centre.

    Replacing a d-dimensional point search with this 1-D search was the single
    most productive change in the campaign (round 11, five functions at once).
    """
    a = np.ravel(np.asarray(anchor, dtype=float))
    return np.clip(a + lam * (CENTRE - a), LOW, HIGH)


def project_to_line(anchor: np.ndarray, x: np.ndarray) -> float:
    """Recover lam for a point assumed to lie on the centre-ward line."""
    a = np.ravel(anchor).astype(float)
    dvec = CENTRE - a
    denom = float(dvec @ dvec)
    return float((np.ravel(x) - a) @ dvec / denom) if denom > 0 else np.nan


def fit_line_vertex(lams, ys) -> tuple[float, float, np.ndarray]:
    """Fit a parabola through >=3 collinear points and solve for its vertex.

    Returns (lam_star, predicted_y, coeffs). lam_star is NaN when the fit is not
    concave, i.e. the line has no interior maximum in range.
    """
    lams, ys = np.asarray(lams, float), np.asarray(ys, float)
    if len(lams) < 3:
        raise ValueError("need at least three points on the line")
    c = np.polyfit(lams, ys, 2)
    a, b = c[0], c[1]
    if a >= 0:
        return float("nan"), float("nan"), c
    lam_star = -b / (2 * a)
    return float(lam_star), float(np.polyval(c, lam_star)), c


def finite_difference(x1, y1, x2, y2) -> tuple[np.ndarray, float]:
    """Signed unit direction and directional derivative from two nearby queries.

    A direction costs the same single query as a point and tells you about an
    entire ray rather than one location.
    """
    x1, x2 = np.ravel(x1).astype(float), np.ravel(x2).astype(float)
    step = x2 - x1
    dist = float(np.linalg.norm(step))
    if dist == 0:
        raise ValueError("identical points carry no directional information")
    return step / dist, (y2 - y1) / dist


def propose_line(fid: int, anchor_round: int | None = None,
                 up_to: int | None = None) -> dict:
    """Centre-ward line diagnostics: project every observation onto the line from
    the anchor, fit the vertex, and propose the vertex point."""
    anchor = (np.array(HISTORY[anchor_round][fid][0], float)
              if anchor_round is not None else best_point(fid, up_to=up_to)[0])
    X, y, rounds = load(fid, up_to=up_to)
    lams = np.array([project_to_line(anchor, x) for x in X])
    on_line = np.array([np.allclose(centre_line(anchor, l), x, atol=1e-6)
                        for l, x in zip(lams, X)])
    out = dict(func=fid, anchor=anchor, n_on_line=int(on_line.sum()),
               warning=anchor_check(fid, anchor, up_to=up_to))
    if on_line.sum() >= 3:
        lam_star, pred, _ = fit_line_vertex(lams[on_line], y[on_line])
        out.update(lam_star=lam_star, predicted=pred)
        if np.isfinite(lam_star):
            x = centre_line(anchor, lam_star)
            out.update(x=x, submission=submission(x))
    else:
        # Not enough collinear points yet — probe the line to create them.
        probes = [0.3, 0.5, 0.7, 1.0]
        out["probes"] = {l: submission(centre_line(anchor, l)) for l in probes}
    return out


# ------------------------------------------------------------------ submission

def submission(x) -> str:
    """Portal format: six decimals, dash-separated, no labels."""
    return "-".join(f"{float(v):.6f}" for v in np.ravel(x))


def validate(s: str, d: int) -> bool:
    parts = s.split("-")
    return (len(parts) == d
            and all(LOW - 1e-9 <= float(p) <= HIGH + 1e-9 for p in parts))


# --------------------------------------------------------------------- reports

def report() -> None:
    pd.set_option("display.width", 140, "display.max_columns", 20)

    print("\n=== Best known per function ===")
    led = ledger().copy()
    led["best"] = led["best"].map(lambda v: f"{v:.6g}")
    led["x"] = led["x"].map(lambda v: submission(v))
    print(led.to_string(index=False))

    print(f"\n=== Final submission (round {FINAL_ROUND}) vs best known ===")
    fvb = final_vs_best().copy()
    for c in ("final", "best"):
        fvb[c] = fvb[c].map(lambda v: f"{v:.6g}")
    print(fvb.to_string(index=False))
    n = int(final_vs_best()["matched"].sum())
    print(f"\n{n} of {len(FUNC_IDS)} final submissions matched the best value found.")

    print("\n=== New bests per round ===")
    print(round_ledger().to_string(index=False))


def propose_all(n_grid: int = 5000, xi: float = 0.05, seed: int = 0) -> None:
    print(f"GP + EI proposals (grid={n_grid}, xi={xi})\n")
    warnings = []
    for fid in FUNC_IDS:
        r = propose_ei(fid, n_grid=n_grid, xi=xi, seed=seed)
        print(r["submission"])
        if not np.isnan(r["loo_r2"]) and r["loo_r2"] < 0:
            warnings.append(f"F{fid}: LOO R2 = {r['loo_r2']:.3f} — surrogate is worse "
                            f"than predicting the mean; treat this proposal as arbitrary.")
        if r["warning"]:
            warnings.append(r["warning"])
    for w in warnings:
        print(f"\n! {w}", file=sys.stderr)


def plot(path: str | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = path or os.path.join(OUTPUT_DIR, "trajectories.png")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, fid in zip(axes.ravel(), FUNC_IDS):
        _, y, rounds = load(fid)
        ax.plot(rounds, y, "o", ms=4, alpha=0.55, label="query")
        ax.plot(rounds, np.maximum.accumulate(y), "-", lw=2, label="best so far")
        xb, yb, rb = best_point(fid)
        ax.axvline(rb, color="grey", ls=":", lw=1)
        ax.set_title(f"F{fid}  (d={DIMS[fid]})  best {yb:.4g} @ W{rb}", fontsize=9)
        ax.set_xlabel("round", fontsize=8)
        ax.tick_params(labelsize=7)
        if fid in (5, 8):
            ax.set_yscale("symlog")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Best-so-far per function across the campaign", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


# ------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="best-known table and final-round comparison")
    pp = sub.add_parser("propose", help="GP + EI proposal for every function")
    pp.add_argument("--grid", type=int, default=5000)
    pp.add_argument("--xi", type=float, default=0.05)
    pp.add_argument("--seed", type=int, default=0)
    pl = sub.add_parser("line", help="centre-ward line diagnostics for one function")
    pl.add_argument("func", type=int, choices=FUNC_IDS)
    pl.add_argument("--anchor-round", type=int, default=None)
    sub.add_parser("plot", help="write best-so-far trajectory panels")
    sub.add_parser("export", help="write the history to CSV, one file per function")

    a = p.parse_args(argv)
    if a.cmd == "report":
        report()
    elif a.cmd == "propose":
        propose_all(n_grid=a.grid, xi=a.xi, seed=a.seed)
    elif a.cmd == "line":
        r = propose_line(a.func, anchor_round=a.anchor_round)
        for k, v in r.items():
            if k in ("anchor", "x") and v is not None:
                print(f"{k}: {submission(v)}")
            elif v is not None:
                print(f"{k}: {v}")
    elif a.cmd == "plot":
        print(plot())
    elif a.cmd == "export":
        for pth in export_csvs():
            print(pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
