#!/bin/bash
#SBATCH --job-name=viz_hier_celeba_hq
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/viz_hier_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/viz_hier_celeba_hq_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────────────────────
# CelebA-HQ taxon-hierarchy visualisations.
#
# For every taxon model run found under outputs/celeba_hq/{,main/}, this writes
#   outputs/analysis_celeba_hq/taxon_hierarchy/<short>/
#       single_img{i}_<stage>_full.png   (one fixed image at every node)
#       topk_<stage>_full.png            (2x2 of top-K activating per node)
#       cache/*.npy                      (raw acts, top-K imgs, gradcams)
#
# Usage
# -----
#   sbatch scripts/analyze/visualize_hierarchy_celeba_hq.sh
#   sbatch scripts/analyze/visualize_hierarchy_celeba_hq.sh --model-filter bottleneck_topk
#   sbatch scripts/analyze/visualize_hierarchy_celeba_hq.sh --force-recompute
#   sbatch scripts/analyze/visualize_hierarchy_celeba_hq.sh --skip-existing
#   sbatch scripts/analyze/visualize_hierarchy_celeba_hq.sh --ablations ./outputs/celeba_hq/ablations
#
# Forwarded args (see src/analyze/visualize_hierarchy_celeba_hq.py):
#   --max-samples N      : val images scanned for top-K + fixed selection (default 1000)
#   --n-fixed N          : fixed images shown across the tree (default 4)
#   --top-k K            : top-K activating images per node (default 4)
#   --batch-size B       : eval batch size (default 16)
#   --num-workers W      : DataLoader workers (default 4)
#   --model-filter STR   : only process runs whose name contains STR
#   --force-recompute    : ignore cached .npy files and re-extract everything
#   --skip-existing      : skip runs that already have figures
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_ARGS="${@}"   # forward all CLI arguments to the python script

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting viz_hier_celeba_hq (job $SLURM_JOB_ID)"
echo "  extra_args: $EXTRA_ARGS"

python src/analyze/visualize_hierarchy_celeba_hq.py \
    --outputs-dir ./outputs/celeba_hq \
    --data-root   ./data/celeba_hq \
    --save-dir    ./outputs/analysis_celeba_hq \
    --image-size  256 \
    --max-samples 1000 \
    --n-fixed     4 \
    --top-k       4 \
    --batch-size  16 \
    --num-workers 4 \
    $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished viz_hier_celeba_hq (job $SLURM_JOB_ID)"
