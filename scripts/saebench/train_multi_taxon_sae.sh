#!/bin/bash
#SBATCH --job-name=train_multi_taxon_sae
#SBATCH --output=slurm/slurm_outputs/train_multi_taxon_sae_%j.out
#SBATCH --error=slurm/slurm_errors/train_multi_taxon_sae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Train MultiTaxonSAE (Pythia-160M, layer 8) ──────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MultiTaxonSAE training (job $SLURM_JOB_ID)"

python src/train/saebench/train_multi_taxon_sae.py \
    --config configs/saebench/multi_taxon_sae_pythia160m.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished MultiTaxonSAE training (job $SLURM_JOB_ID)"
