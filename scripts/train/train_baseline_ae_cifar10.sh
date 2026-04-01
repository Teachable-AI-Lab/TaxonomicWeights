#!/bin/bash
#SBATCH --job-name=train_baseline_cifar10
#SBATCH --output=slurm/slurm_outputs/train_baseline_ae_cifar10_%j.out
#SBATCH --error=slurm/slurm_errors/train_baseline_ae_cifar10_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=8:00:00
#SBATCH --qos=short

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

python src/train/cifar10/train_baseline_ae.py \
    --config          ./configs/cifar10/baseline_ae_cifar10.json \
    --output-dir      ./outputs/baseline_ae_cifar10_r18 \
    --data-root       ./data \
    --batch-size      128 \
    --num-workers     4 \
    --resnet-variant  18 \
    --stem-stride     1 \
    --no-use-stem-maxpool \
    --epochs          90 \
    --learning-rate   3e-4 \
    --weight-decay    1e-4 \
    --warmup-epochs   3 \
    --save-every      5
