#!/bin/bash
#SBATCH --job-name=viz_hier_imagenet
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/viz_hier_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/viz_hier_imagenet_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Tiny-ImageNet taxon-hierarchy visualisations.
#
# For every taxon model run found under outputs/imagenet/{,main/}, this writes
#   outputs/analysis_imagenet/taxon_hierarchy/<short>/
#       single_img{i}_<stage>_full.png   (one fixed image at every node)
#       topk_<stage>_full.png            (2x2 of top-K activating per node)
#       cache/*.npy                      (raw acts, top-K imgs, gradcams)
#
# Usage
# -----
#   sbatch scripts/analyze/visualize_hierarchy_imagenet.sh
#   sbatch scripts/analyze/visualize_hierarchy_imagenet.sh --model-filter bottleneck_topk
#   sbatch scripts/analyze/visualize_hierarchy_imagenet.sh --force-recompute
#   sbatch scripts/analyze/visualize_hierarchy_imagenet.sh --skip-existing
#   sbatch scripts/analyze/visualize_hierarchy_imagenet.sh --ablations ./outputs/imagenet/ablations
#
# Forwarded args (see src/analyze/visualize_hierarchy_imagenet.py):
#   --max-samples N      : val images scanned for top-K + fixed selection (default 1000)
#   --n-fixed N          : fixed images shown across the tree (default 4)
#   --top-k K            : top-K activating images per node (default 4)
#   --batch-size B       : eval batch size (default 64)
#   --num-workers W      : DataLoader workers (default 4)
#   --model-filter STR   : only process runs whose name contains STR
#   --force-recompute    : ignore cached .npy files and re-extract everything
#   --skip-existing      : skip runs that already have figures
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_ARGS="${@}"   # forward all CLI arguments to the python script

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting viz_hier_imagenet (job $SLURM_JOB_ID)"
echo "  extra_args: $EXTRA_ARGS"

python src/analyze/visualize_hierarchy_imagenet.py \
    --outputs-dir ./outputs/imagenet \
    --data-root   ./data/tiny_imagenet \
    --save-dir    ./outputs/analysis_imagenet \
    --image-size  64 \
    --max-samples 1000 \
    --n-fixed     4 \
    --top-k       4 \
    --batch-size  64 \
    --num-workers 4 \
    $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished viz_hier_imagenet (job $SLURM_JOB_ID)"
