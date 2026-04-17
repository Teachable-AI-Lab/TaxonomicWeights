#!/bin/bash
#SBATCH --job-name=cmp_celeba
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/slurm_outputs/comparison_celeba_%j.out
#SBATCH --error=slurm/slurm_errors/comparison_celeba_%j.err
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle

# Usage:
#   sbatch scripts/paper/comparison_paper_celeba_hq.sh
#   Optional: override paper dir and/or save dir:
#   sbatch scripts/paper/comparison_paper_celeba_hq.sh outputs/paper/celeba_hq outputs/paper/celeba_hq/comparison

set -euo pipefail

PAPER_DIR="${1:-outputs/paper/celeba_hq}"
SAVE_DIR="${2:-${PAPER_DIR}/comparison}"

cd /nethome/ksingara3/flash/TaxonomicWeights
source /nethome/ksingara3/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "Paper dir  : $PAPER_DIR"
echo "Save dir   : $SAVE_DIR"
echo "Node       : $(hostname)"

python src/paper/comparison_paper_celeba.py \
    --paper-dir "$PAPER_DIR" \
    --save-dir  "$SAVE_DIR"
