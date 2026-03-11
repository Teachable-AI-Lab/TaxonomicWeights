#!/bin/bash
#SBATCH --job-name=train_taxon_ae
#SBATCH --output=slurm/slurm_outputs/train_taxon_ae_%j.out
#SBATCH --error=slurm/slurm_errors/train_taxon_ae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=tail-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Run training ───────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_celeba_hq_ae (job $SLURM_JOB_ID)"

python tests/train_celeba_hq_ae.py \
    --config configs/celeba_hq.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_celeba_hq_ae (job $SLURM_JOB_ID)"
