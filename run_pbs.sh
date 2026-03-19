#!/bin/bash
#PBS -N qutrit_meas
#PBS -o logs/job_${PBS_JOBID}.out
#PBS -e logs/job_${PBS_JOBID}.err
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=10:mem=16gb
#PBS -J 1-5

# ---------------------------------------------------------------------------
# Job array over J values.
# Submit with:   qsub run_pbs.sh
# Monitor with:  qstat -J <jobid>
# ---------------------------------------------------------------------------

# --- parameter sweep (one value per array index)
J_VALUES=(1.0 3.0 5.0 7.2 10.0)
J=${J_VALUES[$PBS_ARRAY_INDEX-1]}

# --- fixed parameters
N=4
H=1.0
OMEGA=5.3
TF=15
NSTEP=150
NTRAJ=200
N_JOBS=10       # should match ncpus above
SEED=42
OUTDIR="${PBS_O_WORKDIR}/results"

# --- environment setup
cd "${PBS_O_WORKDIR}"
mkdir -p logs "$OUTDIR"

module load python/3.10   # adjust to your cluster's module name

# activate virtualenv or conda env if needed:
# source ~/venvs/qutip/bin/activate
# conda activate qutip_env

# --- run
echo "Starting job array index ${PBS_ARRAY_INDEX}, J=${J}"
echo "Running on host: $(hostname),  $(date)"

python run_cluster.py \
    --N      "$N"      \
    --h      "$H"      \
    --J      "$J"      \
    --Omega  "$OMEGA"  \
    --tf     "$TF"     \
    --nstep  "$NSTEP"  \
    --ntraj  "$NTRAJ"  \
    --n_jobs "$N_JOBS" \
    --seed   "$SEED"   \
    --outdir "$OUTDIR"

echo "Job finished: $(date)"
