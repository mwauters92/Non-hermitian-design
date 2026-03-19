#!/usr/bin/env python
# coding: utf-8
"""
run_cluster.py
==============
Cluster script for qutrit measurement-induced dynamics.

Runs two simulations:
  1. Post-selected dynamics vs effective NH Hamiltonian (force_proj=True)
  2. Stochastic trajectories with projective measurements

Results are saved as .npz + companion .json (parameters).

Usage
-----
    python run_cluster.py --N 4 --h 1.0 --J 7.2 --Omega 5.3 \\
                          --tf 15 --nstep 150 --ntraj 100 --n_jobs 10

    python run_cluster.py --N 4 --h 1.0 --J 4.0 --Omega 0.5 \\
                          --tf 15 --nstep 150 --ntraj 200 --n_jobs -1 \\
                          --force_proj --outdir ./results
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import qutip as qt

# --- import simulation library (must be in the same directory or on PYTHONPATH)
from qudit_nh_design import (
    build_hamiltonian,
    embedded_pauli,
    embed_on_site,
    time_evo,
    run_trajectories,
)


# ---------------------------------------------------------------------------
# Projectors (global, as in the notebook)
# ---------------------------------------------------------------------------
Q_mat      = np.zeros([3, 3])
Q_mat[2,2] = 1
Q = qt.Qobj(Q_mat)   # projector onto level |2>  (the "ancilla" level)
P = 1 - Q            # projector onto the {|0>,|1>} subspace


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qutrit measurement-induced dynamics — cluster runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Hamiltonian parameters
    phys = parser.add_argument_group("Physics")
    phys.add_argument("--N",     type=int,   default=4,   help="Number of qutrits")
    phys.add_argument("--h",     type=float, default=1.0, help="Transverse field (01 subspace)")
    phys.add_argument("--J",     type=float, default=7.2, help="Two-body coupling")
    phys.add_argument("--Omega", type=float, default=5.3, help="Drive on 12 subspace")

    # --- Time evolution parameters
    evo = parser.add_argument_group("Evolution")
    evo.add_argument("--tf",         type=float, default=15.0, help="Final time")
    evo.add_argument("--nstep",      type=int,   default=150,  help="Number of time steps")
    evo.add_argument("--force_proj", action="store_true",      help="Force proj outcome (post-selection)")
    evo.add_argument("--ntraj",      type=int,   default=100,  help="Number of stochastic trajectories")
    evo.add_argument("--n_jobs",     type=int,   default=1,    help="Parallel workers (-1 = all cores)")
    evo.add_argument("--seed",       type=int,   default=None, help="RNG base seed (for reproducibility)")
    evo.add_argument("--cut",        type=int,   default=None, help="EE bipartition cut (default N//2)")

    # --- Output
    out = parser.add_argument_group("Output")
    out.add_argument("--outdir", type=str, default=".", help="Output directory")
    out.add_argument("--tag",    type=str, default="",  help="Optional extra tag appended to filename")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Filename builder
# ---------------------------------------------------------------------------

def make_filename(args, suffix=""):
    """Build a descriptive filename stem from the run parameters."""
    fp = (f"data_N{args.N}_h{args.h}_J{args.J}_Om{args.Omega}"
          f"_tf{args.tf}_nstep{args.nstep}"
          f"_ntraj{args.ntraj}_forceProj{args.force_proj}")
    if args.tag:
        fp += f"_{args.tag}"
    if suffix:
        fp += f"_{suffix}"
    return fp


# ---------------------------------------------------------------------------
# Hamiltonian builder (shared setup)
# ---------------------------------------------------------------------------

def build_models(args):
    N, h, J, Omega = args.N, args.h, args.J, args.Omega
    rate = args.nstep / args.tf

    # --- Full qutrit Hamiltonian (Hermitian)
    single_body_terms = [
        [h,     ('x', 0, 1), None],
        [Omega, ('x', 1, 2), None],
    ]
    two_body_terms = [
        [J, ('x', 1, 2), ('x', 1, 2)],
    ]
    H_qutrit = build_hamiltonian(N, single_body_terms, two_body_terms)

    # --- Effective NH Hamiltonian (qubit, d=2)
    lambda1 = -1j * Omega**2 / rate / 4
    lambda2 = -1j * J**2     / rate / 4
    single_body_eff = [
        [h,          ('x', 0, 1), None],
        [2*lambda1,  ('p-', 0, 1), None],
    ]
    two_body_eff = [
        [2*lambda2, ('p-', 0, 1), ('p-', 0, 1)],
    ]
    H_eff = build_hamiltonian(N, single_body_eff, two_body_eff, d=2)

    print(f"  H_qutrit Hermitian : {H_qutrit.isherm}")
    print(f"  H_eff    Hermitian : {H_eff.isherm}")
    print(f"  lambda_1 = {lambda1:.4f},  lambda_2 = {lambda2:.4f}")

    return H_qutrit, H_eff, rate


# ---------------------------------------------------------------------------
# Simulation 1: post-selected vs effective
# ---------------------------------------------------------------------------

def run_postselected(args, H_qutrit, H_eff):
    N  = args.N
    tvec = np.linspace(0, args.tf, args.nstep + 1)
    cut  = args.cut if args.cut is not None else N // 2

    # --- Initial states
    psi0     = qt.basis([3] * N, n=[0, 1] * (N // 2))
    psi0_eff = qt.basis([2] * N, n=[0, 1] * (N // 2))

    # --- Observables: sigma^z_{01} on each site
    Z01      = embedded_pauli('z', 0, 1)
    e_evo     = [embed_on_site(Z01,          j, N, d=3) for j in range(N)]
    e_evo_eff = [embed_on_site(qt.sigmaz(),  j, N, d=2) for j in range(N)]

    options = {'progress': False}

    print("  Running post-selected qutrit dynamics...")
    t0 = time.time()
    res_qutrit = time_evo(
        H_qutrit, psi0, tvec, P, N,
        measured_sites = list(range(N)),
        e_ops          = e_evo,
        force_proj     = True,
        cut            = cut,
        options        = options,
    )
    print(f"    done in {time.time()-t0:.1f}s")

    print("  Running effective NH dynamics...")
    t0 = time.time()
    res_eff = time_evo(
        H_eff, psi0_eff, tvec, P, N,
        measured_sites = [],
        e_ops          = e_evo_eff,
        force_proj     = False,
        cut            = cut,
        options        = options,
    )
    print(f"    done in {time.time()-t0:.1f}s")

    return res_qutrit, res_eff


# ---------------------------------------------------------------------------
# Simulation 2: stochastic trajectories
# ---------------------------------------------------------------------------

def run_stochastic(args, H_qutrit):
    N    = args.N
    tvec = np.linspace(0, args.tf, args.nstep + 1)
    cut  = args.cut if args.cut is not None else N // 2

    psi0 = qt.basis([3] * N, n=[0, 1] * (N // 2))
    Z01  = embedded_pauli('z', 0, 1)
    e_evo = [embed_on_site(Z01, j, N, d=3) for j in range(N)]

    print(f"  Running {args.ntraj} stochastic trajectories "
          f"on {args.n_jobs} worker(s)...")
    t0 = time.time()
    res = run_trajectories(
        H_qutrit, psi0, tvec, P, N,
        ntraj          = args.ntraj,
        measured_sites = list(range(N)),
        e_ops          = e_evo,
        n_jobs         = args.n_jobs,
        cut            = cut,
        seed           = args.seed,
        options        = {'progress': False},
    )
    print(f"    done in {time.time()-t0:.1f}s")
    return res


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(args, res_ps, res_eff, res_traj):
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, make_filename(args))

    # --- Build save dict
    # Post-selected: expect shape (N, n_times), eff idem
    data = {
        # shared time axis
        "times"           : res_ps.times,

        # post-selected qutrit
        "ps_expect"       : np.array(res_ps.expect),    # (N, n_times)
        "ps_succ_PS"      : np.array(res_ps.succ_PS, dtype=bool),
        "ps_entanglement" : res_ps.entanglement,        # (n_times,)

        # effective NH model
        "eff_expect"      : np.array(res_eff.expect),
        "eff_entanglement": res_eff.entanglement,

        # stochastic trajectories
        "traj_expect_mean": np.array(res_traj.expect_mean),   # (N, n_times)
        "traj_expect_std" : np.array(res_traj.expect_std),
        "traj_expect_traj": np.array(res_traj.expect_traj),   # (N, ntraj, n_times)
        "traj_proj_traj"  : res_traj.proj_traj,               # (ntraj, n_times)
        "traj_ee_mean"    : res_traj.ee_mean,                  # (n_times,)
        "traj_ee_std"     : res_traj.ee_std,
        "traj_ee_traj"    : res_traj.ee_traj,                  # (ntraj, n_times)
    }

    npz_path  = stem + ".npz"
    json_path = stem + ".json"

    np.savez(npz_path, **data)
    print(f"  Data saved  -> {npz_path}")

    # --- Save parameters as companion JSON (readable without loading data)
    params = vars(args)
    params["cut_used"] = args.cut if args.cut is not None else args.N // 2
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Params saved-> {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  Qutrit measurement-induced dynamics — cluster run")
    print("=" * 60)
    print(f"  N={args.N}  h={args.h}  J={args.J}  Omega={args.Omega}")
    print(f"  tf={args.tf}  nstep={args.nstep}  ntraj={args.ntraj}")
    print(f"  force_proj={args.force_proj}  n_jobs={args.n_jobs}")
    print(f"  outdir={args.outdir}")
    print()

    t_total = time.time()

    # --- Build Hamiltonians
    print("[1/3] Building Hamiltonians...")
    H_qutrit, H_eff, rate = build_models(args)

    # --- Simulation 1
    print("\n[2/3] Post-selected + effective dynamics...")
    res_ps, res_eff = run_postselected(args, H_qutrit, H_eff)

    # --- Simulation 2
    print("\n[3/3] Stochastic trajectories...")
    res_traj = run_stochastic(args, H_qutrit)

    # --- Save
    print("\nSaving results...")
    save_results(args, res_ps, res_eff, res_traj)

    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")
    print("Done.")
