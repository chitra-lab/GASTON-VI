#!/bin/bash
#SBATCH --job-name=pe_unified
#SBATCH --output=/scratch/jhu/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/pe_heuristic/logs/unified_%j.out
#SBATCH --error=/scratch/jhu/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/pe_heuristic/logs/unified_%j.err
#SBATCH --partition=l40s
#SBATCH --mail-user=zning4@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=2
#
# One run of train.py.  Edit the block below and `sbatch run_unified.slurm`.
# (run_unified_array.slurm is the TSV-driven version for sweeping many configs.)

# ======================= EDIT THIS BLOCK =======================
DATA=/scratch/jhu/uchitra1/datasets/spatial-RNA/merfish/mouse_liver_merfish

COORDS=$DATA/coords_mat_rep2_qc20.npy
COUNTS=$DATA/counts_mat_rep2_qc20.npy
GENES=$DATA/gene_names.npy
BATCH=                      # batch .npy for JOINT; leave EMPTY for single-section
COND=                       # condition .npy, or leave empty

# Results layout:  runs_unified/<dataset>/<single_b{N}|joint>/<pe_tag>/
# Three levels, each answering one question: which data, single or joint, which PE.
# Everything else (seeds, pe_source, band count, lambda_min, why the run was made)
# is already inside summary_*.csv, so it does not belong in the path.  The old tree
# had 15 top-level runs_* dirs keyed on the REASON for a run -- mouse joint ended up
# split across runs_joint/ and runs_joint_v2/ with no overlap, and every scorer
# needed a hand-written map of which tree held what.  One tree globs as
# runs_unified/*/*/*/.
RUNS=/scratch/jhu/uchitra1/projects/GASTON-VAE/sweep/scVI/interpretation/pe_heuristic/runs_unified
DATASET_TAG=mouse
SPLIT_TAG=single_b1         # single_b{N} for one section, or "joint"
PE_TAG=pe_auto_single       # original | pe_auto_single | pe_auto_both | altB | ...
RESULTS=$RUNS/$DATASET_TAG/$SPLIT_TAG/$PE_TAG

# ---- how to choose the PE bands: pick ONE of the two blocks ----
#
# (A) AUTOMATIC -- derive from the data with pe_heuristic
#       none   : don't; use the manual block (B) below
#       single : derive for single-section data; MULTI-BATCH input falls back to
#                the plain sigma=0.18 / num_freqs=16 ladder
#       both   : derive everywhere, one PE PER BATCH when BATCH is set
PE_AUTO=single
PE_CACHE=                   # optional cache .json; reused only if the inputs are
                            # unchanged (path+size+mtime of counts/coords/batch
                            # and every knob), so a regenerated counts file
                            # recomputes instead of silently reusing old bands

# (B) MANUAL -- used only when PE_AUTO=none.  Setting these together with
#     PE_AUTO != none makes the script ERROR rather than silently pick one.
SIGMA=0.18
NUM_FREQS=16
EXTRAS=                     # "" = plain ladder (this is `original`)
                            # "0.0664,0.0743,..."   explicit wavelengths
                            # "geom:0.0664,1.1131,27"  lo,hi,n geomspace
                            # "pbpe:/path/to/per_batch.json"  per-batch spec

# ---- model / training ----
Z_DIM=4
N_ISODEPTH=1
EPOCHS=175
WARMUP=100
REPEATS=2                   # seeds
BATCH_SIZE=2586
VAL_FRAC=0.3
BETA=0.01
HIDDEN_DIM_SPATIAL=         # leave EMPTY to let train.py pick (64 joint / 20 single)
HIDDEN_DIM_DECODER=20
DECODER_TYPE=linear_hz      # linear_hz | additive | combined
DI_METHOD=simvi_mmd
DI_WEIGHT=0.01
BATCH_ALIGN_WEIGHT=0.1
# ===================== END OF EDIT BLOCK =======================

source /home/jhu/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/jhu/zning4/miniconda3/envs/scvi-env
# CODE_DIR is where train.py and the gaston_vi/ package live (this repo).
# Override it to run a checkout elsewhere: CODE_DIR=/path/to/GASTON-VI sbatch ...
CODE_DIR="${CODE_DIR:-/weka/scratch/jhu/uchitra1/projects/GASTON-VAE/FINAL_CODE}"
SCRIPT="${CODE_DIR}/train.py"

ARGS=()
# Only forward --hidden_dim_spatial when set; otherwise train.py applies the
# data-dependent default (the rule lives there, not duplicated here).
[[ -n "$HIDDEN_DIM_SPATIAL" ]] && ARGS+=(--hidden_dim_spatial "$HIDDEN_DIM_SPATIAL")
[[ -n "$BATCH" ]] && ARGS+=(--batch_file "$BATCH")
[[ -n "$COND"  ]] && ARGS+=(--condition_file "$COND")

PE_AUTO="${PE_AUTO:-none}"
if [[ "$PE_AUTO" != "none" ]]; then
    ARGS+=(--pe_auto "$PE_AUTO")
    [[ -n "$PE_CACHE" ]] && ARGS+=(--pe_cache "$PE_CACHE")
else
    ARGS+=(--sigma "$SIGMA" --num_freqs "$NUM_FREQS")
    if   [[ "$EXTRAS" == pbpe:* ]]; then ARGS+=(--pe_per_batch_json "${EXTRAS#pbpe:}")
    elif [[ "$EXTRAS" == geom:* ]]; then ARGS+=(--extra_wavelengths_geomspace "${EXTRAS#geom:}")
    elif [[ -n "$EXTRAS" ]];       then ARGS+=(--extra_wavelengths "$EXTRAS"); fi
fi

mkdir -p "$RESULTS"
echo "=========================================="
echo "coords=$COORDS"; echo "counts=$COUNTS"
echo "batch=${BATCH:-<single-section>}  pe_auto=$PE_AUTO  hspat=${HIDDEN_DIM_SPATIAL:-<train.py default>}"
echo "results=$RESULTS"; echo "extra args: ${ARGS[*]}"
echo "start $(date)"
echo "=========================================="

python -u "$SCRIPT" \
    --coords_file "$COORDS" --counts_file "$COUNTS" --gene_names_file "$GENES" \
    --z_dim "$Z_DIM" --n_isodepth "$N_ISODEPTH" --beta "$BETA" \
    --hidden_dim_decoder "$HIDDEN_DIM_DECODER" \
    --warmup_epochs "$WARMUP" --epochs "$EPOCHS" --repeats "$REPEATS" \
    --batch_size "$BATCH_SIZE" --val_frac "$VAL_FRAC" \
    --di_method "$DI_METHOD" --di_weight "$DI_WEIGHT" \
    --batch_align_weight "$BATCH_ALIGN_WEIGHT" \
    --decoder_type "$DECODER_TYPE" --results_dir "$RESULTS" "${ARGS[@]}"
EXIT=$?
echo "end $(date)  exit $EXIT"
exit $EXIT
