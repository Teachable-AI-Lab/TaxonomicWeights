#!/bin/bash
#SBATCH --job-name=cache_gemma2b
#SBATCH --output=slurm/slurm_outputs/cache_gemma2b_%j.out
#SBATCH --error=slurm/slurm_errors/cache_gemma2b_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-24:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── HuggingFace token (Gemma-2 is a gated model) ─────────────────────────────
export HF_TOKEN="$(cat hf_token | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ── Cache activations (Gemma-2-2B, layer 12) ─────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting activation caching (job $SLURM_JOB_ID)"

python src/data/cache_activations.py \
    --model-name gemma-2-2b \
    --hook-layer 12 \
    --n-tokens 500000000 \
    --batch-size 2048 \
    --context-length 1024 \
    --dtype bfloat16 \
    --output-dir ./cached_activations/gemma2b_layer12

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished activation caching (job $SLURM_JOB_ID)"
