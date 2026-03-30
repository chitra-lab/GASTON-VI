#!/bin/bash
#SBATCH --job-name=zdim_sweep
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/zdim_sweep/logs/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/zdim_sweep/logs/%A_%a.err
#SBATCH --partition=a100
#SBATCH --exclude=l06,l07,l05,c012,c013,c009
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-4
#SBATCH --time=24:00:00
# ============================================================
# Usage:
#   sbatch --export=DATASET=kidney GASTON_VI_zdim_sweep.sh
#   sbatch --export=DATASET=liver  GASTON_VI_zdim_sweep.sh
#
# DATASET defaults to "kidney" if not provided.
#
# Sweep arrays (update values and --array accordingly):
#   Z_DIMS, D_DIMS, CORR_WEIGHTS, NUM_NODES_SPATIAL
#   Total jobs = product of all array lengths.
#   Current: 1 x 1 x 1 x 1 = 1 job (--array=0).
# ============================================================
DATASET=${DATASET:-kidney}

# ---- Dataset-specific sigma (best from smoothness sweep) ----
if [ "$DATASET" == "kidney" ]; then
    SIGMA=0.183959
elif [ "$DATASET" == "liver" ]; then
    SIGMA=0.647718
else
    echo "ERROR: Unknown DATASET='$DATASET'. Choose kidney or liver."
    exit 1
fi

# ---- Fixed hyperparameters ----
BATCH_SIZE=2586
NUM_FREQS=16
BETA=0.01
EPOCHS=150
REPEATS=1

# ---- Sweep parameters ----
Z_DIMS=(4)
D_DIMS=(1)
CORR_WEIGHTS=(0 0.00001 0.01 0.1 1)
NUM_NODES_SPATIALS=(20)

N_Z=${#Z_DIMS[@]}
N_D=${#D_DIMS[@]}
N_C=${#CORR_WEIGHTS[@]}
N_N=${#NUM_NODES_SPATIALS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
Z_DIM=${Z_DIMS[$((IDX / (N_D * N_C * N_N)))]}
D_DIM=${D_DIMS[$(( (IDX / (N_C * N_N)) % N_D ))]}
CORR_WEIGHT=${CORR_WEIGHTS[$(( (IDX / N_N) % N_C ))]}
NUM_NODES_SPATIAL=${NUM_NODES_SPATIALS[$((IDX % N_N))]}

echo "=========================================="
echo "GASTON VI spatial dim sweep"
echo "Dataset:           $DATASET"
echo "z_dim:             $Z_DIM"
echo "d_dim:             $D_DIM"
echo "corr_weight:       $CORR_WEIGHT"
echo "num_nodes_spatial: $NUM_NODES_SPATIAL"
echo "sigma:             $SIGMA"
echo "batch_size:        $BATCH_SIZE"
echo "Job ID:            $SLURM_JOB_ID"
echo "Array index:       $IDX"
echo "Start:             $(date)"
echo "=========================================="

# ---- Directories ----
BASE=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/zdim_sweep
LOG_DIR=$BASE/logs
RESULTS_DIR=$BASE/results/spatial_dim/v2/${DATASET}
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# ---- Environment ----
source /home/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/zning4/miniconda3/envs/scvi-env

which python
python --version

# ---- Run ----
cd $BASE
python -u spatial_dim.py \
    --dataset            "$DATASET" \
    --z_dim              $Z_DIM \
    --d_dim              $D_DIM \
    --sigma              $SIGMA \
    --num_freqs          $NUM_FREQS \
    --beta               $BETA \
    --epochs             $EPOCHS \
    --repeats            $REPEATS \
    --batch_size         $BATCH_SIZE \
    --corr_weight        $CORR_WEIGHT \
    --num_nodes_spatial  $NUM_NODES_SPATIAL \
    --results_dir        "$RESULTS_DIR"

EXIT_CODE=$?
echo "=========================================="
echo "End:       $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE