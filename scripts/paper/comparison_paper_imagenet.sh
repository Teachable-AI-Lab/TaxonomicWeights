#!/bin/bash
#SBATCH --job-name=cmp_imagenet
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/slurm_outputs/comparison_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/comparison_imagenet_%j.err
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle

# Usage:
#   sbatch scripts/paper/comparison_paper_imagenet.sh
#   Optional: override paper dir and/or save dir:
#   sbatch scripts/paper/comparison_paper_imagenet.sh outputs/paper/imagenet outputs/paper/imagenet/comparison

set -euo pipefail

PAPER_DIR="${1:-outputs/paper/imagenet}"
SAVE_DIR="${2:-${PAPER_DIR}/comparison}"

cd /nethome/ksingara3/flash/TaxonomicWeights
source /nethome/ksingara3/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "Paper dir  : $PAPER_DIR"
echo "Save dir   : $SAVE_DIR"
echo "Node       : $(hostname)"

python src/paper/comparison_paper_imagenet.py \
    --paper-dir "$PAPER_DIR" \
    --save-dir  "$SAVE_DIR"
