#!/bin/bash
#SBATCH --job-name=compare_celeba_hq
#SBATCH --output=slurm/slurm_outputs/compare_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/compare_celeba_hq_%j.err
#SBATCH --partition=overcap
#SBATCH --account=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --qos=short

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

python src/compare/compare_celeba_hq.py \
    --outputs-dir    ./outputs \
    --save-dir       ./outputs/comparison_celeba_hq \
    --data-root      ./data/celeba_hq \
    --image-size     256 \
    --batch-size     16 \
    --num-workers    8 \
    --val-split      0.05 \
    --n-latent-batches 25 \
    --n-recon-batches  10 \
    --n-recon-images   6 \
    --device         cuda
