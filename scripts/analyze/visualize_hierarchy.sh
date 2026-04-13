#!/bin/bash
#SBATCH --job-name=visualize_hierarchy
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/visualize_hierarchy_%j.out
#SBATCH --error=slurm/slurm_errors/visualize_hierarchy_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# GradCAM + Gradient Maximization hierarchy tree visualizations for all taxon
# model run directories found under outputs/.
#
# For each taxon AE variant (taxon, multi_taxon, topk_taxon, topk_multi_taxon,
# bias_taxon, bias_multi_taxon), for every stage in the model, produces:
#
#   <run_dir>/hierarchy_visualizations/
#     *_gradcam_full.png   — top-K activating images with GradCAM overlay,
#                            arranged as a top-down binary tree.
#     *_gradmax_full.png   — gradient-maximization synthetic images per node,
#                            arranged as a top-down binary tree.
#     *_top4.png           — (only for hierarchies > 4 levels) top-4-level panel.
#     *_leaf*.png          — (only for hierarchies > 4 levels) leaf-4-level panels.
#
# Usage:
#   sbatch scripts/analyze/visualize_hierarchy.sh
#   sbatch scripts/analyze/visualize_hierarchy.sh --skip-existing
#   sbatch scripts/analyze/visualize_hierarchy.sh --dry-run
#   sbatch scripts/analyze/visualize_hierarchy.sh --outputs-dir outputs/imagenet
#
# Gradient-max options (passed via EXTRA_ARGS / positional args):
#   --gradmax-steps N   (default 512)
#   --gradmax-lr F      (default 0.02)
#   --gradmax-tv F      (default 1e-4)
#   --gradmax-l2 F      (default 1e-5)
#   --eval-batches N    (default 30)
#   --top-k N           (default 4)

EXTRA_ARGS="$@"

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting visualize_hierarchy (job ${SLURM_JOB_ID:-local})"

python src/analyze/visualize_hierarchy.py --skip-existing $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished visualize_hierarchy (job ${SLURM_JOB_ID:-local})"
