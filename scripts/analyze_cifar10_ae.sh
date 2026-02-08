#!/bin/bash
#SBATCH --job-name=analyze_cifar10_ae
#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --gpus-per-node=a40
#SBATCH --exclude=spot,heistotron,clippy
#SBATCH --output=TaxonomicWeights/slurm/slurm_outputs/analyze_cifar10_ae.out
#SBATCH --error=TaxonomicWeights/slurm/slurm_errors/analyze_cifar10_ae.err
#SBATCH --partition="tail-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate rag-cobweb
cd ~/flash/TaxonomicWeights
export PYTHONPATH=$(pwd)

# Config file or checkpoint path - can be overridden by command line argument
# If argument looks like a .json file, treat it as config, otherwise as checkpoint
ARG=${1:-"configs/cifar10_default.json"}

echo "Starting CIFAR-10 Taxonomic Autoencoder analysis at $(date)"

if [[ "$ARG" == *.json ]]; then
    echo "Using config file: $ARG"
    srun python tests/analyze_cifar10_ae.py --config "$ARG"
else
    echo "Using checkpoint: $ARG (with default architecture)"
    # Architecture parameters must match training configuration
    srun python tests/analyze_cifar10_ae.py \
        --checkpoint "$ARG" \
        --encoder-kernel-sizes 5 5 5 \
        --decoder-kernel-sizes 6 6 5 \
        --encoder-strides 2 2 \
        --decoder-strides 2 2 1 \
        --use-maxpool
fi

echo "Analysis script completed at $(date)"
