#!/bin/bash
#SBATCH --job-name=kidney_C
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/sweep/kidney/logs/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/sweep/kidney/logs/%A_%a.err
#SBATCH --partition=l40s
#SBATCH --exclude=l07,l05,c011,c007
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0   # two jobs

source /scratch/uchitra1/users/zning/gaston-env/bin/activate

# ---- define hyperparameter lists ----
SIGMAS=(0.183959)
BETAS=(0.01)

# ---- select params for this job ----
IDX=${SLURM_ARRAY_TASK_ID}
SIGMA=${SIGMAS[$IDX]}
BETA=${BETAS[$IDX]}

echo "Running job $IDX with sigma=$SIGMA, beta=$BETA"

python -u GASTON_VI.py \
  --variant "C" \
  --sigma $SIGMA \
  --num_freqs 16 \
  --noise_level 1.0 \
  --epochs 100 \
  --repeats 1 \
  --norm "z_score" \
  --beta $BETA \
  --likelihood "nb" \
  --results_dir "/scratch/uchitra1/projects/GASTON-VAE/sweep/discrete" \
  --batch_size 2586