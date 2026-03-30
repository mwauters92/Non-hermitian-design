@ -0,0 +1,477 @@
#!/usr/bin/env python
# coding: utf-8
"""
analyse_data.py
===============
Load saved .npz simulation data and produce publication-quality figures.

Figures
-------
  Figure 1 (4-panel):
    1.1  Observables vs time — post-selected & effective model
    1.2  |Δ observable| vs time (post-selected minus effective)
    1.3  Entanglement entropy vs time (both models)
    1.4  Success probability vs time + exponential fit

  Figure 2 (2-panel, trajectory view):
    2.1  Per-trajectory observables (faint) + mean (thick),
         coloured by final proj_traj value
    2.2  Per-trajectory entanglement (faint) + mean (thick),
         same colour scheme

Fit results are saved as a companion .json so they can be collected
across many runs and plotted vs J, Omega, N, etc.

Usage
-----
    # both figures
    python analyse_data.py results/data_N4_h1.0_J7.2_Om5.3_*.npz

    # only figure 1
    python analyse_data.py data.npz --no-fig2

    # only figure 2, do not show interactively, save to pdf
    python analyse_data.py data.npz --no-fig1 --fmt pdf --no-show

    # batch: process every npz in a directory
    python analyse_data.py results/*.npz --no-show
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp_decay(t, A, gamma, C):
    """A * exp(-gamma * t) + C"""
    return A * np.exp(-gamma * t) + C


def load_npz(path):
    """Load an .npz file and return a dict-like NpzFile object."""
    return np.load(path, allow_pickle=False)


def load_params(path):
    """
    Load the companion .json parameter file.
    Returns an empty dict if not found.
    """
    json_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


def success_probability(succ_PS):
    """
    Compute the fraction of time steps where the proj outcome was selected.
    succ_PS : bool array (n_times,)
    Returns float array (n_times,)
    """
    return succ_PS.astype(float)   # 1 = success, 0 = failure; plot as running mean below


def running_mean(x, window=1):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')

def success_until_t(ps):
    for ntr, traj in enumerate(ps):
        traj[0] = True
        for ist, pst in enumerate(traj):
            if not pst:
                traj[ist+1:]=False
                continue
        ps[ntr] = traj
    return ps

# ---------------------------------------------------------------------------
# Fit: exponential decay of success probability
# ---------------------------------------------------------------------------

def fit_success_prob(times, succ_PS, smooth_window=5):
    """
    Fit P_success(t) = A * exp(-gamma * t) + C.

    Parameters
    ----------
    times      : (n_times,) array
    succ_PS    : (n_times,) bool array from post-selected run
    smooth_window : int, light smoothing before fit

    Returns
    -------
    dict with keys: A, gamma, C, A_err, gamma_err, C_err, fit_ok
    """
    y = running_mean(succ_PS.astype(float), window=smooth_window)
    t = times

    # Initial guess: decaying from ~1 to ~0
    p0 = [1, 1.0 / (t[-1] - t[0] + 1e-9), 0]
    bounds = ([0, 0, 0], [2, 1/t[1], 1])

    try:
        popt, pcov = curve_fit(_exp_decay, t, y, p0=p0, bounds=bounds,
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        return dict(A=popt[0], gamma=popt[1], C=popt[2],
                    A_err=perr[0], gamma_err=perr[1], C_err=perr[2],
                    fit_ok=True)
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Exponential fit failed: {e}")
        return dict(A=np.nan, gamma=np.nan, C=np.nan,
                    A_err=np.nan, gamma_err=np.nan, C_err=np.nan,
                    fit_ok=False)


# ---------------------------------------------------------------------------
# Colour helpers for trajectory plots
# ---------------------------------------------------------------------------

# Two colours: True (proj succeeded) and False (proj failed)
_TRAJ_COLORS = {True: "#2196F3", False: "#F44336"}   # blue / red


def traj_color(proj_final):
    """proj_final : bool scalar — final value of proj_traj for that trajectory."""
    return _TRAJ_COLORS[bool(proj_final)]


# ---------------------------------------------------------------------------
# Figure 1: 4-panel summary
# ---------------------------------------------------------------------------

def make_figure1(data, params, fit_result, fmt="pdf", outdir=".", show=True,
                 smooth_window=5):
    """
    Four-panel summary figure.

    Parameters
    ----------
    data       : NpzFile
    params     : dict  (from companion JSON)
    fit_result : dict  (from fit_success_prob)
    """
    N     = int(params.get("N", data["ps_expect"].shape[0]))
    times = data["times"]

    ps_exp  = data["ps_expect"]      # (N, n_times)
    eff_exp = data["eff_expect"]     # (N, n_times)
    ps_ee   = data["ps_entanglement"]
    eff_ee  = data["eff_entanglement"]
    succ_PS = success_until_t(data["traj_proj_traj"]).mean(axis=0)     # (n_times,) bool

    # colour cycle for sites
    site_colors = plt.cm.tab10(np.linspace(0, 0.9, max(N,10)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(_make_title(params), fontsize=11, y=1.01)
    ax1, ax2, ax3, ax4 = axes.flat

    # ---- Panel 1: observables ------------------------------------------------
    label_sites=['even', 'odd']
    for j in range(2):
        ax1.plot(times, np.array(ps_exp[j::2]).mean(axis=0),  color=site_colors[j], lw=1.8,
                 label=f"{label_sites[j]} sites (PS)")
        ax1.plot(times, np.array(eff_exp[j::2]).mean(axis=0), color=site_colors[j+2], lw=1.8,
                 ls="--",label=f"{label_sites[j]} sites (eff.)")

    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\langle \sigma^z_{01} \rangle$")
    ax1.set_title("Observables")
    # custom legend: solid=PS, dashed=eff
    handles = [Line2D([0],[0], color="k", lw=1.8, label="Post-selected"),
               Line2D([0],[0], color="k", lw=1.8, ls="--", alpha=0.7,
                      label="Effective")]
    ax1.legend(handles=handles,fontsize=8, loc="best")

    # ---- Panel 2: absolute difference ----------------------------------------
    diff = np.abs(ps_exp - eff_exp).mean(axis=0)   # (N, n_times)
    #for j in range(N):
    ax2.plot(times, diff, color=site_colors[0], lw=1.5,
             label=f"site {j}")
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$|\langle O \rangle_{\rm PS} - \langle O \rangle_{\rm eff}|$")
    ax2.set_title("Observable difference  |PS − eff|")
    #ax2.legend(fontsize=8, loc="best")
    ax2.set_yscale("log")

    # ---- Panel 3: entanglement entropy ----------------------------------------
    ax3.plot(times, ps_ee/np.log(2),  color="#1a237e", lw=1.8, label="Post-selected")
    ax3.plot(times, eff_ee/np.log(2), color="#b71c1c", lw=1.8, ls="--", label="Effective")
    ax3.set_xlabel("t")
    ax3.set_ylabel("$S_{\\rm EE}$ (nats)")
    ax3.set_title("Half-chain entanglement entropy")
    ax3.legend(fontsize=8)

    # ---- Panel 4: success probability + fit -----------------------------------
    p_succ = running_mean(succ_PS.astype(float), window=smooth_window)
    #p_succ = succ_PS.astype(float)
    ax4.plot(times, p_succ, color="#37474f", lw=1.5, alpha=0.8,
             label="$P_{\\rm succ}$ (smoothed)")
    ax4.scatter(times[::max(1, len(times)//40)],
                p_succ[::max(1, len(times)//40)],
                color="#37474f", s=15, zorder=3)

    if fit_result["fit_ok"]:
        t_fit  = np.linspace(times[0], times[-1], 500)
        y_fit  = _exp_decay(t_fit, fit_result["A"],
                            fit_result["gamma"], fit_result["C"])
        gamma  = fit_result["gamma"]
        g_err  = fit_result["gamma_err"]
        label_fit = (f"Fit: $A e^{{-\\gamma t}}+C$\n"
                     f"$\\gamma = {gamma:.3f} \\pm {g_err:.3f}$")
        ax4.plot(t_fit, y_fit, color="#e53935", lw=2, ls="--", label=label_fit)

    ax4.set_xlabel("t")
    ax4.set_ylabel("$P_{\\rm succ}$")
    ax4.set_title("Success probability")
    ax4.set_ylim(-0.05, 1.1)
    ax4.legend(fontsize=8)

    fig.tight_layout()

    stem  = _stem(params, outdir)
    fpath = f"{stem}_fig1.{fmt}"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Figure 1 saved -> {fpath}")
    if show:
        plt.show()
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Figure 2: trajectory view
# ---------------------------------------------------------------------------

def make_figure2(data, params, fmt="png", outdir=".", show=True):
    """
    Two-panel trajectory figure.
    Trajectories are coloured by their *majority* proj_traj value
    (True = blue, False = red).
    """
    times      = data["times"]
    exp_traj   = data["traj_expect_traj"]   # (N, ntraj, n_times)
    exp_mean   = data["traj_expect_mean"]   # (N, n_times)
    ee_traj    = data["traj_ee_traj"]        # (ntraj, n_times)
    ee_mean    = data["traj_ee_mean"]        # (n_times,)
    proj_traj  = data["traj_proj_traj"]      # (ntraj, n_times) bool

    ntraj = exp_traj.shape[1]
    N     = exp_traj.shape[0]

    # Classify each trajectory by majority proj outcome
    majority_proj = proj_traj.mean(axis=1) >= 0.5   # (ntraj,) bool

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(_make_title(params) + "  [trajectories]", fontsize=11, y=1.01)

    # ---- Panel 1: observables per trajectory ---------------------------------
    # average across sites to get a single scalar per trajectory per time
    exp_traj_avg = exp_traj.mean(axis=0)   # (ntraj, n_times)
    exp_mean_avg = exp_mean.mean(axis=0)   # (n_times,)

    for t in range(ntraj):
        c = traj_color(majority_proj[t])
        ax1.plot(times, exp_traj_avg[t], color=c, lw=0.5, alpha=0.25)

    ax1.plot(times, exp_mean_avg, color="k", lw=2.2, zorder=5, label="Mean")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\langle \sigma^z_{01} \rangle$ (site avg)")
    ax1.set_title("Observables — trajectories")
    _traj_legend(ax1)

    # ---- Panel 2: entanglement per trajectory --------------------------------
    for t in range(ntraj):
        c = traj_color(majority_proj[t])
        ax2.plot(times, ee_traj[t], color=c, lw=0.5, alpha=0.25)

    ax2.plot(times, ee_mean, color="k", lw=2.2, zorder=5, label="Mean")
    ax2.set_xlabel("t")
    ax2.set_ylabel("$S_{\\rm EE}$ (nats)")
    ax2.set_title("Entanglement — trajectories")
    _traj_legend(ax2)

    fig.tight_layout()

    stem  = _stem(params, outdir)
    fpath = f"{stem}_fig2.{fmt}"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Figure 2 saved -> {fpath}")
    if show:
        plt.show()
    plt.close(fig)
    return fpath


def _traj_legend(ax):
    handles = [
        Line2D([0],[0], color=_TRAJ_COLORS[True],  lw=1.5, label="proj = True"),
        Line2D([0],[0], color=_TRAJ_COLORS[False], lw=1.5, label="proj = False"),
        Line2D([0],[0], color="k",                  lw=2.2, label="Mean"),
    ]
    ax.legend(handles=handles, fontsize=8)


# ---------------------------------------------------------------------------
# Fit result persistence
# ---------------------------------------------------------------------------

def save_fit_result(fit_result, params, outdir):
    """
    Append (or create) a JSON file that collects fit results from many runs,
    indexed by the physical parameters. Useful for plotting gamma vs J etc.

    File: <outdir>/fit_results.json
    Each entry is a dict with all params + fit values.
    """
    fpath = os.path.join(outdir, "fit_results.json")

    # Load existing entries
    if os.path.exists(fpath):
        with open(fpath) as f:
            all_results = json.load(f)
    else:
        all_results = []

    entry = {
        # physical parameters (from companion JSON, or defaults)
        "N":     params.get("N",     None),
        "h":     params.get("h",     None),
        "J":     params.get("J",     None),
        "Omega": params.get("Omega", None),
        "tf":    params.get("tf",    None),
        "nstep": params.get("nstep", None),
        "ntraj": params.get("ntraj", None),
        # fit values
        "gamma":     fit_result["gamma"],
        "gamma_err": fit_result["gamma_err"],
        "A":         fit_result["A"],
        "C":         fit_result["C"],
        "fit_ok":    fit_result["fit_ok"],
    }
    all_results.append(entry)

    with open(fpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Fit result appended -> {fpath}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _make_title(params):
    if not params:
        return "Qutrit dynamics"
    return (f"N={params.get('N','?')}  "
            f"h={params.get('h','?')}  "
            f"J={params.get('J','?')}  "
            f"Ω={params.get('Omega','?')}  "
            f"tf={params.get('tf','?')}  "
            f"nstep={params.get('nstep','?')}")


def _stem(params, outdir):
    if params:
        name = (f"data_N{params.get('N','X')}"
                f"_h{params.get('h','X')}"
                f"_J{params.get('J','X')}"
                f"_Om{params.get('Omega','X')}"
                f"_tf{params.get('tf','X')}"
                f"_nstep{params.get('nstep','X')}")
    else:
        name = "data_unknown"
    return os.path.join(outdir, name)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse qutrit dynamics simulation data (.npz)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("files", nargs="+", metavar="FILE.npz",
                   help="One or more .npz data files to process")
    p.add_argument("--no-fig1",  action="store_true",
                   help="Skip the 4-panel summary figure")
    p.add_argument("--no-fig2",  action="store_true",
                   help="Skip the trajectory figure")
    p.add_argument("--no-show",  action="store_true",
                   help="Do not call plt.show() (useful for batch/headless runs)")
    p.add_argument("--fmt",      default="png", choices=["png","pdf","svg"],
                   help="Output figure format")
    p.add_argument("--outdir",   default=None,
                   help="Output directory for figures (default: same as input file)")
    p.add_argument("--smooth",   type=int, default=5,
                   help="Smoothing window for success probability plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Use non-interactive backend when --no-show is set
    if args.no_show:
        matplotlib.use("Agg")

    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"[WARN] File not found: {fpath} — skipping.")
            continue

        print(f"\nProcessing: {fpath}")
        data   = load_npz(fpath)
        params = load_params(fpath)
        outdir = args.outdir or os.path.dirname(fpath) or "."
        os.makedirs(outdir, exist_ok=True)

        # --- Exponential fit (always computed, even if fig1 is skipped)
        fit = fit_success_prob(data["times"], success_until_t(data["traj_proj_traj"]).mean(axis=0),
                               smooth_window=args.smooth)
        if fit["fit_ok"]:
            print(f"  Fit: gamma = {fit['gamma']:.4f} ± {fit['gamma_err']:.4f}")
        else:
            print("  Fit: FAILED (check data or try different --smooth)")

        # Always save fit result for later collection
        save_fit_result(fit, params, outdir)

        # --- Figures
        if not args.no_fig1:
            make_figure1(data, params, fit,
                         fmt=args.fmt, outdir=outdir,
                         show=not args.no_show,
                         smooth_window=args.smooth)

        if not args.no_fig2:
            make_figure2(data, params,
                         fmt=args.fmt, outdir=outdir,
                         show=not args.no_show)

    print("\nDone.")
