#!/bin/bash
#SBATCH --job-name=train_celeba_hq_ae
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron,clippy
#SBATCH --output=TaxonomicWeights/slurm/slurm_outputs/train_celeba_hq_ae_attn.out
#SBATCH --error=TaxonomicWeights/slurm/slurm_errors/train_celeba_hq_ae_attn.err
#SBATCH --account="overcap"
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate aryan_exp
cd /nethome/aroy389/flash/aroy389/taxon/TaxonomicWeights
export PYTHONPATH=$(pwd)

# Config file - can be overridden by command line argument
CONFIG_FILE=${1:-"configs/celeba_hq_attention.json"}

echo "Starting CelebA-HQ Autoencoder training at $(date)"
echo "Using config: $CONFIG_FILE"

srun python tests/train_celeba_hq_ae_attention.py --config "$CONFIG_FILE"

echo "Training script completed at $(date)"
