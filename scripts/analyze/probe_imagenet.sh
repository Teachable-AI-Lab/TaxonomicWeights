#!/bin/bash
#SBATCH --job-name=probe_imagenet
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/probe_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/probe_imagenet_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ─────────────────────────────────────────────────────────────────────────────
# Tiny-ImageNet probing analysis: linear probing, KNN, and sparse interpretability
# over ALL runs in both outputs/imagenet/ and outputs/imagenet/main/.
#
# Usage
# -----
#   sbatch scripts/analyze/probe_imagenet.sh
#   sbatch scripts/analyze/probe_imagenet.sh --model-filter bottleneck_topk
#   sbatch scripts/analyze/probe_imagenet.sh --skip-knn --max-samples 2000
#   sbatch scripts/analyze/probe_imagenet.sh --ablations ./outputs/imagenet/ablations
#
# Any extra arguments are forwarded to probe_imagenet.py, e.g.:
#   --force-recompute      : ignore cached latents
#   --ablations DIR        : also process ablation runs from DIR
#   --skip-sparsity        : skip sparsity analysis (faster)
#   --skip-knn             : skip KNN probing
#   --max-samples N        : cap images used for probing (default 10000)
#   --model-filter STR     : only process runs whose name contains STR
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_ARGS="${@}"   # forward all CLI arguments to the python script

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting probe_imagenet (job $SLURM_JOB_ID)"
echo "  extra_args: $EXTRA_ARGS"

python src/analyze/probe_imagenet.py \
    --outputs-dir ./outputs/imagenet \
    --data-root   ./data/tiny_imagenet \
    --save-dir    ./outputs/analysis_imagenet \
    --image-size  64 \
    --max-samples 10000 \
    --knn-k 1 5 20 \
    $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished probe_imagenet (job $SLURM_JOB_ID)"
