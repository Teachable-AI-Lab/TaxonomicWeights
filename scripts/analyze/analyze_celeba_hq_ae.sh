#!/bin/bash
#SBATCH --job-name=analyze_taxon_ae
#SBATCH --output=slurm/slurm_outputs/analyze_taxon_ae_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_taxon_ae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Run analysis ───────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_celeba_hq_ae (job $SLURM_JOB_ID)"

python src/analyze/analyze_celeba_hq_ae.py \
    --config configs/taxon_ae_celeba_hq.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_celeba_hq_ae (job $SLURM_JOB_ID)"
