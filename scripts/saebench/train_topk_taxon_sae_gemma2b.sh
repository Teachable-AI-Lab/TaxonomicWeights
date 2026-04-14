#!/bin/bash
#SBATCH --job-name=train_topk_taxon_gemma2b
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_topk_taxon_gemma2b_%j.out
#SBATCH --error=slurm/slurm_errors/train_topk_taxon_gemma2b_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── HuggingFace token (Gemma-2 is a gated model) ─────────────────────────────
export HF_TOKEN="$(cat hf_token | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ── Train TopKTaxonSAE (Gemma-2-2B, layer 12) ────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TopKTaxonSAE Gemma-2B training (job $SLURM_JOB_ID)"

python src/train/saebench/train_topk_taxon_sae.py \
    --config configs/saebench/topk_taxon_sae_gemma2b.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished TopKTaxonSAE Gemma-2B training (job $SLURM_JOB_ID)"
