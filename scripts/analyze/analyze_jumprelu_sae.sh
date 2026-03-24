#!/bin/bash
#SBATCH --job-name=analyze_jumprelu_sae
#SBATCH --output=slurm/slurm_outputs/analyze_jumprelu_sae_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_jumprelu_sae_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Usage: sbatch scripts/analyze/analyze_jumprelu_sae.sh configs/jumprelu_sae_celeba_hq.json
#        sbatch scripts/analyze/analyze_jumprelu_sae.sh configs/jumprelu_sae_cifar10.json
CONFIG="${1:-configs/jumprelu_sae_celeba_hq.json}"

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_jumprelu_sae config=$CONFIG (job $SLURM_JOB_ID)"

python src/analyze/analyze_jumprelu_sae.py --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_jumprelu_sae config=$CONFIG (job $SLURM_JOB_ID)"
