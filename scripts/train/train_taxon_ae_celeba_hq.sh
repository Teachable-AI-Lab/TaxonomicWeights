#!/bin/bash
#SBATCH --job-name=train_taxon_ae_celeba_hq
#SBATCH --output=slurm/slurm_outputs/train_taxon_ae_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/train_taxon_ae_celeba_hq_%j.err
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_taxon_ae_celeba_hq (job $SLURM_JOB_ID)"

python src/train/train_taxon_ae_celeba_hq.py \
    --config configs/taxon_ae_celeba_hq.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_taxon_ae_celeba_hq (job $SLURM_JOB_ID)"
