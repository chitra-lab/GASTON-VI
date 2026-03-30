#!/bin/bash
#SBATCH --job-name=zdim_sweep
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/sweep_sai/debugging/ft/logs_f/zdim_sweep/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/sweep_sai/debugging/ft/logs_f/zdim_sweep/%A_%a.err
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
#SBATCH --time=24:00:00

# ============================================================
# Usage:
#   sbatch --export=DATASET=kidney GASTON_VI_zdim_sweep_all.sh
#   sbatch --export=DATASET=liver  GASTON_VI_zdim_sweep_all.sh
#   sbatch --export=DATASET=tonsil GASTON_VI_zdim_sweep_all.sh
#
# DATASET defaults to "kidney" if not provided.
# ============================================================

DATASET=${DATASET:-kidney}

# ---- Dataset-specific sigma (best from smoothness sweep) ----
if [ "$DATASET" == "kidney" ]; then
    SIGMA=0.183959
elif [ "$DATASET" == "liver" ]; then
    SIGMA=0.183959
  #SIGMA=0.647718
elif [ "$DATASET" == "tonsil" ]; then
    SIGMA=0.454323
else
    echo "ERROR: Unknown DATASET='$DATASET'. Choose kidney, liver, or tonsil."
    exit 1
fi

# ---- Fixed hyperparameters (same as original) ----
BATCH_SIZE=2586
NUM_FREQS=16
BETA=0.01
EPOCHS=175
REPEATS=5

# ---- z_dim sweep values ----
Z_DIMS=(0 1 2 4 6 8 10 12 16 20)

IDX=${SLURM_ARRAY_TASK_ID}
Z_DIM=${Z_DIMS[$IDX]}

echo "=========================================="
echo "GASTON VI z_dim sweep"
echo "Dataset:    $DATASET"
echo "z_dim:      $Z_DIM  (array index $IDX)"
echo "sigma:      $SIGMA"
echo "batch_size: $BATCH_SIZE"
echo "Job ID:     $SLURM_JOB_ID"
echo "Start:      $(date)"
echo "=========================================="

# ---- Directories ----
BASE=/scratch/uchitra1/projects/GASTON-VAE/sweep_sai/debugging/ft
LOG_DIR=$BASE/logs_f/zdim_sweep
RESULTS_DIR=$BASE/results/${DATASET}/zdim_sweep

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# ---- Environment ----
source /scratch/uchitra1/users/sdharma5_code/class/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/uchitra1/users/sdharma5_code/class/miniconda3/envs/gaston_pytorch
export PYTHONPATH=/scratch/uchitra1/projects/GASTON/src:$PYTHONPATH
which python
python --version

# ---- Run ----
cd $BASE/final
python -u GASTON_VI_zdim_sweep_all.py \
    --dataset    "$DATASET" \
    --z_dim      $Z_DIM \
    --sigma      $SIGMA \
    --num_freqs  $NUM_FREQS \
    --beta       $BETA \
    --epochs     $EPOCHS \
    --repeats    $REPEATS \
    --batch_size $BATCH_SIZE \
    --results_dir "$RESULTS_DIR"

EXIT_CODE=$?
echo "=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
