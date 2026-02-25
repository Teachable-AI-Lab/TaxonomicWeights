#!/bin/bash
#SBATCH --job-name=train_celeba_hq_ae
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron,clippy
#SBATCH --output=TaxonomicWeights/slurm/slurm_outputs/train_celeba_hq_ae_%j.out
#SBATCH --error=TaxonomicWeights/slurm/slurm_errors/train_celeba_hq_ae_%j.err
#SBATCH --account="overcap"
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate taxon-weights
cd ~/flash/TaxonomicWeights
export PYTHONPATH=$(pwd)

# Config file - can be overridden by command line argument
CONFIG_FILE=${1:-"configs/celebahq_ae.json"}
# Resume flag - pass "auto" as 2nd arg to resume from latest checkpoint
RESUME_FLAG=${2:-""}

echo "=== SLURM JOB ID: $SLURM_JOB_ID ==="
echo "Starting CelebA-HQ Autoencoder training at $(date)"
echo "Using config: $CONFIG_FILE"

RESUME_ARG=""
if [ -n "$RESUME_FLAG" ]; then
    RESUME_ARG="--resume $RESUME_FLAG"
    echo "Resume mode: $RESUME_FLAG"
fi

srun python tests/train_celeba_hq_ae.py --config "$CONFIG_FILE" $RESUME_ARG

echo "Training script completed at $(date)"
