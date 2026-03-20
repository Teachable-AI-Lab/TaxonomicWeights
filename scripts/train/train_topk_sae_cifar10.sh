#!/bin/bash
#SBATCH --job-name=train_topk_sae_cifar10
#SBATCH --output=slurm/slurm_outputs/train_topk_sae_cifar10_%j.out
#SBATCH --error=slurm/slurm_errors/train_topk_sae_cifar10_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Run training ───────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_topk_sae_cifar10 (job $SLURM_JOB_ID)"

python src/train/train_topk_sae_cifar10.py \
    --config configs/topk_sae_cifar10.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_topk_sae_cifar10 (job $SLURM_JOB_ID)"
