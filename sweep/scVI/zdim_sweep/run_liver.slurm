#!/bin/bash
#SBATCH --job-name=gaston_liver
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/logs/liver/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/logs/liver/%A_%a.err
#SBATCH --partition=l40s
#SBATCH --exclude=l07,l05,c011,c007
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0

# ============================================================
# Full pipeline for the mouse liver MERFISH dataset:
#   1. Preprocess raw data → standardised coords/counts files
#   2. Run GASTON-VI model
#
# Usage:
#   sbatch run_liver.slurm
#
# Override hyperparameters at submission, e.g.:
#   sbatch --export=ALL,Z_DIM=10,EPOCHS=175 run_liver.slurm
# ============================================================

# ---- environment ----
source /home/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/zning4/miniconda3/envs/scvi-env
which python
python --version

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- paths ----
RAW_INPUT_DIR="/scratch/uchitra1/datasets/spatial-RNA/merfish/mouse_liver_merfish"
PREPROCESSED_DIR="/scratch/uchitra1/projects/GASTON-VAE/preprocessed/liver"
COORDS_FILE="$PREPROCESSED_DIR/coords_mat.npy"
COUNTS_FILE="$PREPROCESSED_DIR/counts_mat_hvg.npz"

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
N_HVGS="${N_HVGS:-5000}"

RESULTS_DIR="${RESULTS_DIR:-/scratch/uchitra1/projects/GASTON-VAE/results/liver/zdim${Z_DIM}}"

mkdir -p "$PREPROCESSED_DIR" "$RESULTS_DIR" \
    "/scratch/uchitra1/projects/GASTON-VAE/logs/liver"

echo "=========================================="
echo "GASTON-VI: liver dataset"
echo "z_dim:   $Z_DIM"
echo "sigma:   $SIGMA"
echo "epochs:  $EPOCHS"
echo "results: $RESULTS_DIR"
echo "Job ID:  $SLURM_JOB_ID"
echo "Start:   $(date)"
echo "=========================================="

# ---- Step 1: preprocess ----
echo "--- Preprocessing ---"
python -u "$SCRIPT_DIR/preprocess_liver.py" \
    --input_dir  "$RAW_INPUT_DIR" \
    --output_dir "$PREPROCESSED_DIR" \
    --n_hvgs     "$N_HVGS"

if [ $? -ne 0 ]; then
    echo "ERROR: preprocessing failed"; exit 1
fi

# ---- Step 2: run model ----
echo "--- Running model ---"
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
