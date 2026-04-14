#!/bin/bash
#SBATCH --job-name=eval_multi_taxon_gemma2b
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/eval_multi_taxon_gemma2b_%j.out
#SBATCH --error=slurm/slurm_errors/eval_multi_taxon_gemma2b_%j.err
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
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ── HuggingFace token (Gemma-2 is a gated model) ─────────────────────────────
export HF_TOKEN="$(cat hf_token | tr -d '[:space:]')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

# ── Evaluate MultiTaxonSAE on SAEBench (Gemma-2-2B) ─────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MultiTaxonSAE Gemma-2B SAEBench eval (job $SLURM_JOB_ID)"

python src/eval/run_saebench_evals.py \
    --config configs/saebench/multi_taxon_sae_gemma2b.json \
    --eval-types core sparse_probing scr tpp absorption taxonomy \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished MultiTaxonSAE Gemma-2B SAEBench eval (job $SLURM_JOB_ID)"
