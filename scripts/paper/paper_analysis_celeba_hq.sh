#!/bin/bash
#SBATCH --job-name=paper_celeba
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/slurm_outputs/paper_analysis_celeba_%j.out
#SBATCH --error=slurm/slurm_errors/paper_analysis_celeba_%j.err
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle

# Usage:
#   sbatch scripts/paper/paper_analysis_celeba_hq.sh configs/celeba_hq/my_model.json
#   Optionally pass a checkpoint as a second argument:
#   sbatch scripts/paper/paper_analysis_celeba_hq.sh configs/celeba_hq/my_model.json outputs/celeba_hq/run/ckpt.pt

set -euo pipefail

CONFIG="${1:?Usage: sbatch paper_analysis_celeba_hq.sh <config_path> [checkpoint_path]}"
CHECKPOINT="${2:-}"

cd /nethome/ksingara3/flash/TaxonomicWeights
source /nethome/ksingara3/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "Config     : $CONFIG"
echo "Checkpoint : ${CHECKPOINT:-<auto>}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CKPT_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    CKPT_ARGS="--checkpoint $CHECKPOINT"
fi

python src/paper/paper_analysis_celeba.py \
    --config "$CONFIG" \
    $CKPT_ARGS \
    --celeba-root ./data/celeba_fallback \
    --data-root   ./data/celeba_hq \
    --n-edit      4 \
    --topk-edit   32 \
    --n-recon     8 \
    --num-workers 4 \
    --probe-batch-size 256 \
    --device      cuda
