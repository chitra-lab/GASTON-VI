#!/bin/bash
#SBATCH --job-name=gaston_vi
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/logs/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/logs/%A_%a.err
#SBATCH --partition=l40s
#SBATCH --exclude=l07,l05,c011,c007
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0

# ============================================================
# Usage:
#   sbatch gaston_vi_model.slurm
#
# Override any variable at submission time, e.g.:
#   sbatch --export=ALL,Z_DIM=10,EPOCHS=175 gaston_vi_model.slurm
# ============================================================

# ---- environment ----
source /home/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/zning4/miniconda3/envs/scvi-env
which python
python --version

# ---- data files (edit these to point at preprocessed outputs) ----
COORDS_FILE="${COORDS_FILE:-/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium/coords_mat.npy}"
COUNTS_FILE="${COUNTS_FILE:-/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium/counts_mat_hvg5000.npz}"

# ---- hyperparameters ----
Z_DIM="${Z_DIM:-0}"
N_ISODEPTH="${N_ISODEPTH:-1}"
SIGMA="${SIGMA:-0.183959}"
NUM_FREQS="${NUM_FREQS:-16}"
BETA="${BETA:-0.01}"
EPOCHS="${EPOCHS:-175}"
REPEATS="${REPEATS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2586}"
VAL_FRAC="${VAL_FRAC:-0.3}"

RESULTS_DIR="${RESULTS_DIR:-/scratch/uchitra1/projects/GASTON-VAE/results/zdim${Z_DIM}}"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "GASTON-VI model run"
echo "coords_file: $COORDS_FILE"
echo "counts_file: $COUNTS_FILE"
echo "z_dim:       $Z_DIM"
echo "n_isodepth:  $N_ISODEPTH"
echo "sigma:       $SIGMA"
echo "beta:        $BETA"
echo "epochs:      $EPOCHS"
echo "repeats:     $REPEATS"
echo "results_dir: $RESULTS_DIR"
echo "Job ID:      $SLURM_JOB_ID"
echo "Start:       $(date)"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -u "$SCRIPT_DIR/gaston_vi_model.py" \
    --coords_file  "$COORDS_FILE" \
    --counts_file  "$COUNTS_FILE" \
    --z_dim        "$Z_DIM" \
    --n_isodepth   "$N_ISODEPTH" \
    --sigma        "$SIGMA" \
    --num_freqs    "$NUM_FREQS" \
    --beta         "$BETA" \
    --epochs       "$EPOCHS" \
    --repeats      "$REPEATS" \
    --batch_size   "$BATCH_SIZE" \
    --val_frac     "$VAL_FRAC" \
    --results_dir  "$RESULTS_DIR"

EXIT_CODE=$?
echo "=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
