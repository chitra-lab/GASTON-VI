#!/bin/bash
#SBATCH --job-name=gaston_seq_add_batch_d_condf
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/logs/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/logs/%A_%a.err
#SBATCH --partition=l40s,a100
#SBATCH --exclude=l07,l05,l06,c011,c012,c013,c008,c002,c004,c014
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
# array size = N_DECODER * N_WARMUP * N_DI * N_HSPAT * N_HDEC * N_BATCH_ALIGN
# with the defaults below (1*1*1*1*1*4) that is 4 tasks -> indices 0-3.
#SBATCH --array=0

# ---- environment ----
source /home/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/zning4/miniconda3/envs/scvi-env
which python
python --version

# ---- hardcoded paths ----
SWEEP_DIR="/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation"
SCRIPT_PATH="${SWEEP_DIR}/healthy_diseased/seq_train_additive_batch_d_condf_at_model.py"      # SCRIPT PATH 
#DATA_DIR="/scratch/uchitra1/datasets/spatial-RNA/merfish/mouse_liver_merfish"
DATA_DIR="/scratch/uchitra1/datasets/spatial-RNA/merfish/healthy_diseased_liver"
COORDS_FILE="${DATA_DIR}/coords_mat_combined.npy"
COUNTS_FILE="${DATA_DIR}/counts_mat_combined.npy"
GENE_NAMES_FILE="${DATA_DIR}/gene_names_combined.npy"

# ---- BATCH VECTOR (required for cross-batch isodepth alignment) ----
# Path to a .npy of integer batch indices, shape (N,), values in {0..K-1},
# row order matching COUNTS_FILE. The MMD alignment only activates when this
# is provided AND there are >=2 batches. Leave empty "" to run without batch
# conditioning (alignment becomes a no-op).
BATCH_FILE="${DATA_DIR}/batch_combined.npy"

# ---- CONDITION VECTOR (optional; restricts the cross-batch MMD to within-
# condition batch pairs, e.g. healthy vs diseased) ----
# Path to a .npy of integer condition labels, shape (N,), row order matching
# COUNTS_FILE, with each batch nested inside a single condition. When provided,
# the isodepth RBF-MMD aligns d only across batches that share a condition and
# never across conditions. Leave empty "" to align across all batches.
CONDITION_FILE="${DATA_DIR}/condition_combined.npy"

# ---- fixed hyperparameters ----
Z_DIM="${Z_DIM:-4}"    # NUMBER OF Z DIMENSIONS
N_ISODEPTH="${N_ISODEPTH:-1}"
SIGMA="${SIGMA:-0.18}"
NUM_FREQS="${NUM_FREQS:-16}"
BETA="${BETA:-0.01}"
EPOCHS="${EPOCHS:-175}"
REPEATS="${REPEATS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2586}"
VAL_FRAC="${VAL_FRAC:-0.3}"
DI_WEIGHT="${DI_WEIGHT:-0.01}"

# ---- sweep axes ----
# BATCH_ALIGN_WEIGHTS is the cross-batch isodepth alignment strength: weight on
# the RBF-MMD penalty that pulls each batch's marginal distribution of the
# learned isodepth d onto a common scale. 0 disables it (gradients from this
# term flow to the spatial encoders only). Sweeping it lets you compare scales.
DECODER_TYPES=(linear_hz)
WARMUP_EPOCHS=(100)
DI_METHODS=(simvi_mmd)
HIDDEN_DIM_SPATIALS=(64)
HIDDEN_DIM_DECODERS=(20)
BATCH_ALIGN_WEIGHTS=(0.1)

N_D=${#DECODER_TYPES[@]}
N_W=${#WARMUP_EPOCHS[@]}
N_M=${#DI_METHODS[@]}
N_H=${#HIDDEN_DIM_SPATIALS[@]}
N_HD=${#HIDDEN_DIM_DECODERS[@]}
N_BA=${#BATCH_ALIGN_WEIGHTS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
DECODER_TYPE=${DECODER_TYPES[$((IDX / (N_W * N_M * N_H * N_HD * N_BA)))]}
WARMUP=${WARMUP_EPOCHS[$(( (IDX / (N_M * N_H * N_HD * N_BA)) % N_W ))]}
DI_METHOD=${DI_METHODS[$(( (IDX / (N_H * N_HD * N_BA)) % N_M ))]}
HIDDEN_DIM_SPATIAL=${HIDDEN_DIM_SPATIALS[$(( (IDX / (N_HD * N_BA)) % N_H ))]}
HIDDEN_DIM_DECODER=${HIDDEN_DIM_DECODERS[$(( (IDX / N_BA) % N_HD ))]}
BATCH_ALIGN_WEIGHT=${BATCH_ALIGN_WEIGHTS[$(( IDX % N_BA ))]}

# NOTE: include BATCH_ALIGN_WEIGHT in the results path — the python script's
# output tags don't encode it, so without this, array tasks that differ only in
# alignment weight would overwrite each other.
RESULTS_DIR="${RESULTS_DIR:-${SWEEP_DIR}/healthy_diseased/results/seq_additive_batch_d_condf/linear_decoder/combined/zdim${Z_DIM}/balign${BATCH_ALIGN_WEIGHT}}"

mkdir -p "$RESULTS_DIR" \
    "${SWEEP_DIR}/logs"

# ---- build optional --batch_file arg ----
# If BATCH_FILE is empty or still the placeholder, skip the flag.
BATCH_ARG=()
if [[ -n "$BATCH_FILE" && "$BATCH_FILE" != "<PATH/TO/YOUR/batch.npy>" ]]; then
    if [[ ! -f "$BATCH_FILE" ]]; then
        echo "ERROR: BATCH_FILE='$BATCH_FILE' does not exist." >&2
        exit 1
    fi
    BATCH_ARG=(--batch_file "$BATCH_FILE")
fi

# ---- build optional --condition_file arg ----
# If CONDITION_FILE is empty or the placeholder, skip the flag.
CONDITION_ARG=()
if [[ -n "$CONDITION_FILE" && "$CONDITION_FILE" != "<PATH/TO/YOUR/condition.npy>" ]]; then
    if [[ ! -f "$CONDITION_FILE" ]]; then
        echo "ERROR: CONDITION_FILE='$CONDITION_FILE' does not exist." >&2s
        exit 1
    fi
    CONDITION_ARG=(--condition_file "$CONDITION_FILE")
fi

echo "=========================================="
echo "GASTON-VI: liver — sequential + additive decoder (batch-aware, d-aligned)"
echo "decoder_type:  $DECODER_TYPE"
echo "warmup_epochs: $WARMUP"
echo "di_method:     $DI_METHOD"
echo "di_weight:     $DI_WEIGHT"
echo "batch_align_weight: $BATCH_ALIGN_WEIGHT"
echo "z_dim:         $Z_DIM"
echo "sigma:         $SIGMA"
echo "hidden_dim_spatial: $HIDDEN_DIM_SPATIAL"
echo "hidden_dim_decoder: $HIDDEN_DIM_DECODER"
echo "epochs:        $EPOCHS"
echo "results:       $RESULTS_DIR"
echo "script:        $SCRIPT_PATH"
echo "coords:        $COORDS_FILE"
echo "counts:        $COUNTS_FILE"
echo "batch_file:    ${BATCH_ARG[1]:-<not provided>}"
echo "condition_file: ${CONDITION_ARG[1]:-<not provided>}"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array idx:     $IDX"
echo "Start:         $(date)"
echo "=========================================="

python -u "$SCRIPT_PATH" \
    --coords_file      "$COORDS_FILE" \
    --counts_file      "$COUNTS_FILE" \
    --z_dim            "$Z_DIM" \
    --n_isodepth       "$N_ISODEPTH" \
    --sigma            "$SIGMA" \
    --num_freqs        "$NUM_FREQS" \
    --beta             "$BETA" \
    --hidden_dim_spatial "$HIDDEN_DIM_SPATIAL" \
    --hidden_dim_decoder "$HIDDEN_DIM_DECODER" \
    --warmup_epochs    "$WARMUP" \
    --epochs           "$EPOCHS" \
    --repeats          "$REPEATS" \
    --batch_size       "$BATCH_SIZE" \
    --val_frac         "$VAL_FRAC" \
    --di_method        "$DI_METHOD" \
    --di_weight        "$DI_WEIGHT" \
    --batch_align_weight "$BATCH_ALIGN_WEIGHT" \
    --decoder_type     "$DECODER_TYPE" \
    --gene_names_file  "$GENE_NAMES_FILE" \
    --results_dir      "$RESULTS_DIR" \
    "${BATCH_ARG[@]}" \
    "${CONDITION_ARG[@]}"

EXIT_CODE=$?
echo "=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
