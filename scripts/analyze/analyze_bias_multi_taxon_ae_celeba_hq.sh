#!/bin/bash
#SBATCH --job-name=analyze_bias_multi_taxon_ae
#SBATCH --output=slurm/slurm_outputs/analyze_bias_multi_taxon_ae_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_bias_multi_taxon_ae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Usage: sbatch scripts/analyze/analyze_bias_multi_taxon_ae_celeba_hq.sh
#        sbatch scripts/analyze/analyze_bias_multi_taxon_ae_celeba_hq.sh configs/celeba_hq/bias_multi_taxon_ae_celeba_hq.json
CONFIG="${1:-configs/celeba_hq/bias_multi_taxon_ae_celeba_hq.json}"

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_bias_multi_taxon_ae config=$CONFIG (job $SLURM_JOB_ID)"

python src/analyze/analyze_bias_multi_taxon_ae_celeba_hq.py --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_bias_multi_taxon_ae config=$CONFIG (job $SLURM_JOB_ID)"
