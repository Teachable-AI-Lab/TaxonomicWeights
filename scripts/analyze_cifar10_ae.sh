#!/bin/bash
#SBATCH --job-name=analyze_cifar10_ae
#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron,clippy
#SBATCH --output=TaxonomicWeights/slurm/slurm_outputs/analyze_cifar10_ae.out
#SBATCH --error=TaxonomicWeights/slurm/slurm_errors/analyze_cifar10_ae.err
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

# Config file or checkpoint path - can be overridden by command line argument
# If argument looks like a .json file, treat it as config, otherwise as checkpoint
ARG=${1:-"configs/cifar10_standard.json"}

echo "Starting CIFAR-10 Taxonomic Autoencoder analysis at $(date)"

echo "Using config file: $ARG"
srun python tests/analyze_cifar10_ae.py --config "$ARG"

echo "Analysis script completed at $(date)"
