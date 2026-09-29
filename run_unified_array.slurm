#!/bin/bash
#SBATCH --job-name=pe_unified
#SBATCH --output=/scratch/jhu/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/pe_heuristic/logs/unified_%A_%a.out
#SBATCH --error=/scratch/jhu/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/pe_heuristic/logs/unified_%A_%a.err
#SBATCH --partition=l40s
#SBATCH --mail-user=zning4@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=2
#
# One runner for train.py -- single-section AND joint, manual bands
# or --pe_auto.  Submit with:
#     sbatch --array=0-N --export=ALL,CONFIGS=/abs/path/configs.tsv run_unified.slurm
#
# CONFIGS is a 13-column TSV, one run per line, "-" for any field that does not
# apply:
#   1 dataset     free-text tag (logging only)
#   2 tag         config name; becomes the results subdirectory
#   3 coords      coords .npy
#   4 counts      counts .npy
#   5 genes       gene-names .npy
#   6 batch       batch .npy, or "-" for single-section
#   7 condition   condition .npy, or "-"
#   8 results     output directory
#   9 sigma       "-" when using --pe_auto
#  10 num_freqs   "-" when using --pe_auto
#  11 extras      "" | "0.1,0.2,..." | "geom:lo,hi,n" | "pbpe:/path.json" | "-"
#  12 pe_auto     none | single | both
#  13 pe_cache    cache JSON path, or "-"
#
# pe_auto semantics (from train.py):
#   none   : use columns 9-11 exactly as given (previous behaviour)
#   single : derive the bands from the data for single-section input; MULTI-BATCH
#            input falls back to the plain sigma=0.18 / num_freqs=16 ladder
#   both   : derive everywhere, one PE PER BATCH when a batch file is given
# --pe_auto is mutually exclusive with sigma/num_freqs/extras: the script errors
# rather than silently picking one, so leave 9-11 as "-" when using it.

# NOTE: --hidden_dim_spatial is deliberately NOT passed. train.py derives it
# from the data (64 multi-batch / 20 single-section) so the rule lives in one
# place; duplicating it here is how the two silently drift apart.
source /home/jhu/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/jhu/zning4/miniconda3/envs/scvi-env

# CODE_DIR is where train.py and the gaston_vi/ package live (this repo).
# Override it to run a checkout elsewhere: CODE_DIR=/path/to/GASTON-VI sbatch ...
CODE_DIR="${CODE_DIR:-/weka/scratch/jhu/uchitra1/projects/GASTON-VAE/FINAL_CODE}"
SCRIPT="${CODE_DIR}/train.py"
CONFIGS="${CONFIGS:?set CONFIGS=/abs/path/configs.tsv}"

IDX=${SLURM_ARRAY_TASK_ID:-0}
# Parse the config line FIELD-EXACTLY. `IFS=$'\t' read` collapses consecutive
# tabs (tab is IFS whitespace), so a single empty field silently shifts every
# later column -- which once made a manual run read extras="none" and die with
# "could not convert string to float". awk -F'\t' preserves empty fields, and
# the count is validated so a malformed line fails loudly instead of misparsing.
mapfile -t _F < <(awk -F'\t' -v r="$((IDX + 1))" 'NR==r{for(i=1;i<=NF;i++) print $i}' "$CONFIGS")
if (( ${#_F[@]} != 13 )); then
    echo "ERROR: config line $((IDX + 1)) of $CONFIGS has ${#_F[@]} fields, expected 13" >&2
    exit 1
fi
DATASET=${_F[0]};  TAG=${_F[1]};      COORDS=${_F[2]};  COUNTS=${_F[3]}
GENES=${_F[4]};    BATCH=${_F[5]};    COND=${_F[6]};    RESULTS=${_F[7]}
SIGMA=${_F[8]};    NUM_FREQS=${_F[9]}; EXTRAS=${_F[10]}; PE_AUTO=${_F[11]}
PE_CACHE=${_F[12]}

# ---- fixed hyper-parameters (identical to run_validation / run_joint_v2) ----
Z_DIM=4; N_ISODEPTH=1; BETA=0.01; EPOCHS=175; REPEATS=2; BATCH_SIZE=2586; VAL_FRAC=0.3
DI_METHOD=simvi_mmd; DI_WEIGHT=0.01; WARMUP=100
DECODER_TYPE=linear_hz; HIDDEN_DIM_DECODER=20; BATCH_ALIGN_WEIGHT=0.1
# joint runs used a wider spatial encoder than single-section ones

ARGS=()
[[ -n "$BATCH" && "$BATCH" != "-" ]] && ARGS+=(--batch_file "$BATCH")
[[ -n "$COND"  && "$COND"  != "-" ]] && ARGS+=(--condition_file "$COND")

PE_AUTO="${PE_AUTO:-none}"; [[ "$PE_AUTO" == "-" ]] && PE_AUTO=none
if [[ "$PE_AUTO" != "none" ]]; then
    ARGS+=(--pe_auto "$PE_AUTO")
    [[ -n "$PE_CACHE" && "$PE_CACHE" != "-" ]] && ARGS+=(--pe_cache "$PE_CACHE")
else
    [[ -n "$SIGMA"     && "$SIGMA"     != "-" ]] && ARGS+=(--sigma "$SIGMA")
    [[ -n "$NUM_FREQS" && "$NUM_FREQS" != "-" ]] && ARGS+=(--num_freqs "$NUM_FREQS")
    if   [[ "$EXTRAS" == pbpe:* ]]; then ARGS+=(--pe_per_batch_json "${EXTRAS#pbpe:}")
    elif [[ "$EXTRAS" == geom:* ]]; then ARGS+=(--extra_wavelengths_geomspace "${EXTRAS#geom:}")
    elif [[ -n "$EXTRAS" && "$EXTRAS" != "-" ]]; then ARGS+=(--extra_wavelengths "$EXTRAS"); fi
fi

mkdir -p "$RESULTS"
echo "=========================================="
echo "dataset=$DATASET tag=$TAG  pe_auto=$PE_AUTO"
echo "coords=$COORDS"; echo "counts=$COUNTS"; echo "batch=${BATCH:-<none>}"
echo "results=$RESULTS"; echo "extra args: ${ARGS[*]}"
echo "Job ${SLURM_JOB_ID:-local} idx $IDX  start $(date)"
echo "=========================================="

python -u "$SCRIPT" \
    --coords_file "$COORDS" --counts_file "$COUNTS" --gene_names_file "$GENES" \
    --z_dim "$Z_DIM" --n_isodepth "$N_ISODEPTH" \
    --beta "$BETA" \
    --hidden_dim_decoder "$HIDDEN_DIM_DECODER" \
    --warmup_epochs "$WARMUP" --epochs "$EPOCHS" --repeats "$REPEATS" \
    --batch_size "$BATCH_SIZE" --val_frac "$VAL_FRAC" \
    --di_method "$DI_METHOD" --di_weight "$DI_WEIGHT" \
    --batch_align_weight "$BATCH_ALIGN_WEIGHT" \
    --decoder_type "$DECODER_TYPE" --results_dir "$RESULTS" "${ARGS[@]}"
EXIT=$?
echo "end $(date)  exit $EXIT"
exit $EXIT
