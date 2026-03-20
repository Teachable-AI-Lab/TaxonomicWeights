#!/bin/bash
#SBATCH --job-name=analyze_multi_taxon_ae_celeba_hq
#SBATCH --output=slurm/slurm_outputs/analyze_multi_taxon_ae_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_multi_taxon_ae_celeba_hq_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00

# ── Environment ───────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ─────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${1:-configs/multi_taxon_ae_celeba_hq.json}"

# ── Run analysis ──────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_multi_taxon_ae_celeba_hq (job $SLURM_JOB_ID) config=$CONFIG"

python src/analyze/analyze_multi_taxon_ae_celeba_hq.py \
    --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_multi_taxon_ae_celeba_hq (job $SLURM_JOB_ID)"
