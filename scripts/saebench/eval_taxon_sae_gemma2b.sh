#!/bin/bash
#SBATCH --job-name=eval_taxon_gemma2b
#SBATCH --output=slurm/slurm_outputs/eval_taxon_gemma2b_%j.out
#SBATCH --error=slurm/slurm_errors/eval_taxon_gemma2b_%j.err
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

# ── Evaluate TaxonSAE on SAEBench (Gemma-2-2B) ──────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TaxonSAE Gemma-2B SAEBench eval (job $SLURM_JOB_ID)"

python src/eval/run_saebench_evals.py \
    --config configs/saebench/taxon_sae_gemma2b.json \
    --output-dir outputs/saebench/gemma2b_layer12/eval_results \
    --eval-types core sparse_probing scr tpp absorption \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished TaxonSAE Gemma-2B SAEBench eval (job $SLURM_JOB_ID)"
