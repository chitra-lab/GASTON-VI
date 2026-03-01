#!/bin/bash
#SBATCH --job-name=scVI
#SBATCH --output=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/logs/%A_%a.out
#SBATCH --error=/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI/logs/%A_%a.err
#SBATCH --partition=a100
#SBATCH --exclude=l07,l05,c011,c007
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0   # two jobs

source /home/zning4/miniconda3/etc/profile.d/conda.sh
conda activate /home/zning4/miniconda3/envs/scvi-env
which python
python --version

# ---- define hyperparameter lists ----
SIGMAS=(0.183959)
BETAS=(0.01)

# ---- select params for this job ----
IDX=${SLURM_ARRAY_TASK_ID}
SIGMA=${SIGMAS[$IDX]}
BETA=${BETAS[$IDX]}

echo "Running job $IDX with sigma=$SIGMA, beta=$BETA"

python -u GASTON_VI.py \
  --sigma $SIGMA \
  --num_freqs 16 \
  --noise_level 1.0 \
  --epochs 175 \
  --repeats 1 \
  --norm "z_score" \
  --beta $BETA \
  --results_dir "/scratch/uchitra1/projects/GASTON-VAE/sweep/scVI" \
  --batch_size 2586 \
  --z_dim 10
