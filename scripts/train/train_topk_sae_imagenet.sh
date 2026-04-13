#!/bin/bash
#SBATCH --job-name=train_topk_sae_imagenet
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_topk_sae_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/train_topk_sae_imagenet_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Run training ───────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_topk_sae_imagenet (job $SLURM_JOB_ID)"

python src/train/imagenet/train_topk_sae.py \
    --config configs/imagenet/topk_sae_imagenet.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_topk_sae_imagenet (job $SLURM_JOB_ID)"
