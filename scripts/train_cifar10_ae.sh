#!/bin/bash
#SBATCH --job-name=train_cifar10_ae
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron,clippy
#SBATCH --output=TaxonomicWeights/slurm/slurm_outputs/train_cifar10_ae.out
#SBATCH --error=TaxonomicWeights/slurm/slurm_errors/train_cifar10_ae.err
#SBATCH --account="overcap"
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate taxon-weights
cd ~/flash/TaxonomicWeights
export PYTHONPATH=$(pwd)

# Config file - can be overridden by command line argument
CONFIG_FILE=${1:-"configs/cifar10_standard.json"}

echo "Starting CIFAR-10 Taxonomic Autoencoder training at $(date)"
echo "Using config: $CONFIG_FILE"

srun python tests/train_cifar10_ae.py --config "$CONFIG_FILE"

echo "Training script completed at $(date)"
