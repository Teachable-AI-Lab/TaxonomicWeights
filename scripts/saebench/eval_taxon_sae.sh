#!/bin/bash
#SBATCH --job-name=eval_taxon_saebench
#SBATCH --output=slurm/slurm_outputs/eval_taxon_saebench_%j.out
#SBATCH --error=slurm/slurm_errors/eval_taxon_saebench_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ── Evaluate TaxonSAE on SAEBench (with baselines) ──────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TaxonSAE SAEBench evaluation (job $SLURM_JOB_ID)"

python src/eval/run_saebench_evals.py \
    --config configs/saebench/taxon_sae_pythia160m.json \
    --output-dir outputs/saebench/pythia160m_layer8/eval_results \
    --eval-types core sparse_probing scr tpp absorption \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished TaxonSAE SAEBench evaluation (job $SLURM_JOB_ID)"
