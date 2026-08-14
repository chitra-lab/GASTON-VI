#!/bin/bash
#SBATCH --job-name=zdim_sweep
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/logs/zdim_sweep/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/logs/zdim_sweep/%A_%a.err
#SBATCH --partition=l40s
#SBATCH --exclude=l07,l05,c011,c007
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9   # 10 z_dim values (see Z_DIMS array below)

# ============================================================
# Sweeps over z_dim values by running one Slurm array task per value.
# Each task calls gaston_vi_model.py with the corresponding z_dim.
#
# Usage:
#   # kidney (default)
#   sbatch gaston_vi_zdim_sweep.slurm
#
#   # other datasets (pass preprocessed file paths)
#   sbatch --export=ALL,COORDS_FILE=/path/to/coords_mat.npy,COUNTS_FILE=/path/to/counts_mat_hvg.npz,DATASET_TAG=liver \
#       gaston_vi_zdim_sweep.slurm
# ============================================================

# ---- environment ----
source /home/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/zning4/miniconda3/envs/scvi-env
which python
python --version

# ---- z_dim sweep values (one per array task) ----
Z_DIMS=(0 1 2 4 6 8 10 12 16 20)

IDX=${SLURM_ARRAY_TASK_ID}
Z_DIM=${Z_DIMS[$IDX]}

# ---- data files ----
# Default to kidney preprocessed data; override via --export at submission.
COORDS_FILE="${COORDS_FILE:-/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium/coords_mat.npy}"
COUNTS_FILE="${COUNTS_FILE:-/scratch/uchitra1/datasets/spatial-RNA/visiumhd/mouse_kidney_visium/counts_mat_hvg5000.npz}"
DATASET_TAG="${DATASET_TAG:-kidney}"

# ---- hyperparameters ----
N_ISODEPTH="${N_ISODEPTH:-1}"
SIGMA="${SIGMA:-0.183959}"
NUM_FREQS="${NUM_FREQS:-16}"
BETA="${BETA:-0.01}"
EPOCHS="${EPOCHS:-175}"
REPEATS="${REPEATS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2586}"
VAL_FRAC="${VAL_FRAC:-0.3}"

BASE="/scratch/uchitra1/projects/GASTON-VAE/zdim_sweep/${DATASET_TAG}"
RESULTS_DIR="${RESULTS_DIR:-${BASE}/zdim${Z_DIM}}"
LOG_DIR="${BASE}/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "=========================================="
echo "GASTON-VI z_dim sweep"
echo "Dataset tag: $DATASET_TAG"
echo "z_dim:       $Z_DIM  (array index $IDX)"
echo "sigma:       $SIGMA"
echo "coords:      $COORDS_FILE"
echo "counts:      $COUNTS_FILE"
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
