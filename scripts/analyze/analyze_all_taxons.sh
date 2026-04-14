#!/bin/bash
#SBATCH --job-name=analyze_all_taxons
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/analyze_all_taxons_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_all_taxons_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# Run analysis for all taxon variant output directories, dispatching to:
#
#   taxon_ae_*              → analyze_celeba_hq_ae.py
#   multi_taxon_ae_*        → analyze_multi_taxon_ae_celeba_hq.py
#   topk_taxon_ae_*         → analyze_topk_taxon_ae_celeba_hq.py
#   topk_multi_taxon_ae_*   → analyze_topk_multi_taxon_ae_celeba_hq.py
#   bias_taxon_ae_*         → analyze_bias_taxon_ae_celeba_hq.py
#   bias_multi_taxon_ae_*   → analyze_bias_multi_taxon_ae_celeba_hq.py
#
# Pass --skip-existing to skip runs that already have an analysis/ directory.
# Pass --dry-run to preview the commands without executing them.
#
# Usage:
#   sbatch scripts/analyze/analyze_all_taxons.sh
#   sbatch scripts/analyze/analyze_all_taxons.sh --skip-existing
#   sbatch scripts/analyze/analyze_all_taxons.sh --dry-run

EXTRA_ARGS="$@"

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_all_taxons (job ${SLURM_JOB_ID:-local})"

python src/analyze/analyze_all.py --taxon-only --skip-existing --skip-partonomy $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_all_taxons (job ${SLURM_JOB_ID:-local})"
