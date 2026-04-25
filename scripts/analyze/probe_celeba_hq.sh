#!/bin/bash
#SBATCH --job-name=probe_celeba_hq
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/probe_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/probe_celeba_hq_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ─────────────────────────────────────────────────────────────────────────────
# CelebA-HQ probing analysis: linear probing, KNN, and sparse interpretability
# over ALL runs in both outputs/celeba_hq/ and outputs/celeba_hq/main/.
#
# Usage
# -----
#   sbatch scripts/analyze/probe_celeba_hq.sh
#   sbatch scripts/analyze/probe_celeba_hq.sh --model-filter bottleneck_topk
#   sbatch scripts/analyze/probe_celeba_hq.sh --skip-knn --max-samples 2000
#
# Any extra arguments are forwarded to probe_celeba_hq.py, e.g.:
#   --force-recompute   : ignore cached latents
#   --include-ablations : also process ablation runs
#   --skip-sparsity     : skip sparsity analysis (faster)
#   --skip-knn          : skip KNN probing
#   --max-samples N     : cap images used for probing (default 5000)
#   --model-filter STR  : only process runs whose name contains STR
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_ARGS="${@}"   # forward all CLI arguments to the python script

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting probe_celeba_hq (job $SLURM_JOB_ID)"
echo "  extra_args: $EXTRA_ARGS"

python src/analyze/probe_celeba_hq.py \
    --outputs-dir ./outputs/celeba_hq \
    --data-root   ./data \
    --save-dir    ./outputs/analysis_celeba_hq \
    --image-size  256 \
    --max-samples 5000 \
    --knn-k 1 5 20 \
    $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished probe_celeba_hq (job $SLURM_JOB_ID)"
