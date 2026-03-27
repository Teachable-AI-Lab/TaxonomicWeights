#!/bin/bash
#SBATCH --job-name=compare_pythia_saebench
#SBATCH --output=slurm/slurm_outputs/compare_pythia_saebench_%j.out
#SBATCH --error=slurm/slurm_errors/compare_pythia_saebench_%j.err
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

# ── Compare Pythia-160M SAE variants (ours + baselines) ──────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Pythia-160M comparison (job $SLURM_JOB_ID)"

python src/eval/compare_saebench.py \
    --configs configs/saebench/taxon_sae_pythia160m.json \
              configs/saebench/multi_taxon_sae_pythia160m.json \
    --output-dir outputs/saebench/pythia160m_layer8/comparison \
    --include-baselines \
    --baseline-width 4k \
    --eval-types core sparse_probing scr tpp absorption \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished Pythia-160M comparison (job $SLURM_JOB_ID)"
