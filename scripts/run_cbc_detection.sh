#!/bin/bash
#SBATCH --partition=upex
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --chdir=/gpfs/cfel/user/nivanov/cbclib
#SBATCH --job-name=cbc-detection
#SBATCH --output=results/slog/cbc-detection-%N-%j.out
#SBATCH --error=results/slog/cbc-detection-%N-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

source /software/anaconda3/5.2/etc/profile.d/conda.sh
conda activate /gpfs/cfel/user/nivanov/.conda/envs/cbc
python scripts/cbc_detection.py 206 /asap3/petra3/gpfs/p11/2022/data/11012881/raw \
results/exp_geom_206.ini results/scan_206_data.h5 results/scan_206_indexing.h5 \
--roi 800 3900 800 3600 --frames 0 7539 --mask_max 40 --cor_range 0.3 2.0 --quant 0.018 \
--cutoff 70.0 --filter_threshold 6.0 --group_threshold 0.4 -v
