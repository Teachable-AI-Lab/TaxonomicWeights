#!/bin/bash
#SBATCH --job-name=test_jumprelu_sae
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/test_jumprelu_sae_%j.out
#SBATCH --error=slurm/slurm_errors/test_jumprelu_sae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Architecture test (CelebA-HQ config: 256×256, stem maxpool on) ─────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing JumpReLU SAE — CelebA-HQ config (job $SLURM_JOB_ID)"

python src/test/test_jumprelu_sae.py \
    --image-size 256 \
    --batch-size 2 \
    --resnet-variant 18 \
    --stage-channels 64 128 256 512 \
    --stage-strides 1 2 2 2 \
    --target-l0 64.0 \
    --bandwidth 0.001 \
    --theta-init 0.1 \
    --stem-stride 2 \
    --use-stem-maxpool \
    --device cuda

# ── Architecture test (CIFAR-10 config: 32×32, no stem maxpool) ────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing JumpReLU SAE — CIFAR-10 config"

python src/test/test_jumprelu_sae.py \
    --image-size 32 \
    --batch-size 4 \
    --resnet-variant 18 \
    --stage-channels 64 128 256 512 \
    --stage-strides 1 2 2 2 \
    --target-l0 64.0 \
    --bandwidth 0.001 \
    --theta-init 0.1 \
    --stem-stride 1 \
    --no-use-stem-maxpool \
    --device cuda

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished test_jumprelu_sae (job $SLURM_JOB_ID)"
