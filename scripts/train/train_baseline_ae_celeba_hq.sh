#!/bin/bash
#SBATCH --job-name=train_baseline_celeba_hq
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_baseline_ae_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/train_baseline_ae_celeba_hq_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --qos=short

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

python src/train/celeba_hq/train_baseline_ae.py \
    --config          ./configs/celeba_hq/baseline_ae_celeba_hq.json \
    --output-dir      ./outputs/baseline_ae_celeba_hq_r18 \
    --data-root       ./data/celeba_hq \
    --image-size      256 \
    --batch-size      32 \
    --num-workers     8 \
    --val-split       0.05 \
    --resnet-variant  18 \
    --stem-stride     2 \
    --use-stem-maxpool \
    --epochs          90 \
    --learning-rate   3e-4 \
    --weight-decay    1e-4 \
    --warmup-epochs   3 \
    --save-every      5
