#!/bin/bash
#SBATCH --job-name=compare_gemma_saebench
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/compare_gemma_saebench_%j.out
#SBATCH --error=slurm/slurm_errors/compare_gemma_saebench_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ── HuggingFace token (Gemma-2 is a gated model) ─────────────────────────────
export HF_TOKEN="$(cat hf_token | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ── Compare Gemma-2-2B SAE variants (TaxonSAE vs MultiTaxonSAE) ─────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Gemma-2B comparison (job $SLURM_JOB_ID)"

python src/eval/compare_saebench.py \
    --models-dir outputs/saebench/gemma2b_layer12 \
    --model-name gemma-2-2b \
    --include-baselines \
    --baseline-width 4k \
    --eval-types core sparse_probing scr tpp absorption \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished Gemma-2B comparison (job $SLURM_JOB_ID)"
