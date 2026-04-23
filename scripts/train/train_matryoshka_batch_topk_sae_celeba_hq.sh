#!/bin/bash
#SBATCH --job-name=train_mat_topk_celeba
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_matryoshka_batch_topk_sae_celeba_%j.out
#SBATCH --error=slurm/slurm_errors/train_matryoshka_batch_topk_sae_celeba_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Run training ───────────────────────────────────────────────────────────────
CONFIG="${1:-configs/celeba_hq/matryoshka_batch_topk_sae_celeba_hq.json}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_matryoshka_batch_topk_sae_celeba_hq config=$CONFIG (job $SLURM_JOB_ID)"

python src/train/celeba_hq/train_matryoshka_batch_topk_sae.py \
    --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_matryoshka_batch_topk_sae_celeba_hq config=$CONFIG (job $SLURM_JOB_ID)"
