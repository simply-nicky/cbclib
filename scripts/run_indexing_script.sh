#!/bin/bash
#SBATCH --partition=upex
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --chdir=/gpfs/cfel/user/nivanov/cbclib
#SBATCH --job-name=cbc-indexing
#SBATCH --output=results/slog/cbc-indexing-%N-%j.out
#SBATCH --error=results/slog/cbc-indexing-%N-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

source /software/anaconda3/5.2/etc/profile.d/conda.sh
conda activate /gpfs/cfel/user/nivanov/.conda/envs/cbc
python scripts/cbc_indexing.py pygmo results/scan_232_samples_trial_3.h5 \
results/scan_232_indexing.h5 results/exp_geom_232_new.ini --smp_tol 0.0 \
--num_threads 32 -v
