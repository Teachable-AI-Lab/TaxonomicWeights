#!/bin/bash
#SBATCH --job-name=compare_cifar10
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/compare_cifar10_%j.out
#SBATCH --error=slurm/slurm_errors/compare_cifar10_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --qos=short

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

python src/compare/compare_cifar10.py \
    --outputs-dir       ./outputs/cifar10 \
    --save-dir          ./outputs/cifar10/comparison \
    --data-root         ./data \
    --batch-size        128 \
    --n-latent-batches  20 \
    --n-recon-batches   10 \
    --n-recon-images    8 \
    --device            cuda
