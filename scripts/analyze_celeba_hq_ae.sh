#!/bin/bash
#SBATCH --job-name=analyze_celeba_hq_ae
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron,clippy
#SBATCH --output=TaxonomicWeights/slurm/slurm_outputs/analyze_celeba_hq_ae.out
#SBATCH --error=TaxonomicWeights/slurm/slurm_errors/analyze_celeba_hq_ae.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate taxon-weights
cd ~/flash/TaxonomicWeights
export PYTHONPATH=$(pwd)

# Usage:
#   sbatch scripts/analyze_celeba_hq_ae.sh /path/to/checkpoint.pt [optional_config.json]
# If no args are provided, defaults are used.

CHECKPOINT_PATH=${1:-"outputs/celebahq/training/celebahq_baseline/final_model.pt"}
CONFIG_FILE=${2:-"configs/celebahq_ae.json"}

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo "Starting CelebA-HQ Autoencoder analysis at $(date)"
echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Using config: $CONFIG_FILE"

srun python tests/analyze_celeba_hq_ae.py --config "$CONFIG_FILE" --checkpoint "$CHECKPOINT_PATH"

echo "Analysis script completed at $(date)"
