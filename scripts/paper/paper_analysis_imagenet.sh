#!/bin/bash
#SBATCH --job-name=paper_imagenet
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/slurm_outputs/paper_analysis_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/paper_analysis_imagenet_%j.err
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle

# Usage:
#   sbatch scripts/paper/paper_analysis_imagenet.sh configs/imagenet/my_model.json
#   Optionally pass a checkpoint as a second argument:
#   sbatch scripts/paper/paper_analysis_imagenet.sh configs/imagenet/my_model.json outputs/imagenet/run/ckpt.pt

set -euo pipefail

CONFIG="${1:?Usage: sbatch paper_analysis_imagenet.sh <config_path> [checkpoint_path]}"
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

python src/paper/paper_analysis_imagenet.py \
    --config         "$CONFIG" \
    $CKPT_ARGS \
    --n-edit         4 \
    --topk-edit      32 \
    --n-recon        8 \
    --num-workers    4 \
    --max-val-images 5000 \
    --probe-n-classes 50 \
    --probe-per-class 200 \
    --device         cuda
