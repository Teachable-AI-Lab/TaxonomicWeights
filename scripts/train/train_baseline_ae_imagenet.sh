#!/bin/bash
#SBATCH --job-name=train_baseline_imagenet
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_baseline_ae_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/train_baseline_ae_imagenet_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=1-00:00:00
#SBATCH --qos=short

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

python src/train/imagenet/train_baseline_ae.py \
    --config          ./configs/imagenet/baseline_ae_imagenet.json \
    --output-dir      ./outputs/baseline_ae_imagenet_r18 \
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
