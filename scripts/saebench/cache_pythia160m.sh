#!/bin/bash
#SBATCH --job-name=cache_pythia160m
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/cache_pythia160m_%j.out
#SBATCH --error=slurm/slurm_errors/cache_pythia160m_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Cache activations (Pythia-160M, layer 8) ─────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting activation caching (job $SLURM_JOB_ID)"

python src/data/cache_activations.py \
    --model-name pythia-160m-deduped \
    --hook-layer 8 \
    --n-tokens 500000000 \
    --batch-size 4096 \
    --context-length 1024 \
    --dtype float32 \
    --output-dir ./cached_activations/pythia160m_layer8

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished activation caching (job $SLURM_JOB_ID)"
