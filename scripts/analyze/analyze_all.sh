#!/bin/bash
#SBATCH --job-name=analyze_all
#SBATCH --output=slurm/slurm_outputs/analyze_all_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_all_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# Run analysis for every output directory under outputs/, dispatching to the
# appropriate analyze script based on the directory name prefix:
#
#   taxon_ae_*       → analyze_celeba_hq_ae.py
#   multi_taxon_ae_* → analyze_multi_taxon_ae_celeba_hq.py
#   baseline_ae_*    → analyze_baseline_ae.py
#   sae_*            → analyze_sae.py
#
# Pass --skip-existing to skip runs that already have an analysis/ directory.
# Pass --dry-run to preview the commands without executing them.
#
# Usage:
#   sbatch scripts/analyze/analyze_all.sh
#   sbatch scripts/analyze/analyze_all.sh --skip-existing
#   sbatch scripts/analyze/analyze_all.sh --dry-run

EXTRA_ARGS="$@"

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_all (job ${SLURM_JOB_ID:-local})"

python src/analyze/analyze_all.py $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_all (job ${SLURM_JOB_ID:-local})"
