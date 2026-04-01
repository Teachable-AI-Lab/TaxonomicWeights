#!/bin/bash
#SBATCH --job-name=train_bias_multi_taxon_ae_cifar10
#SBATCH --output=slurm/slurm_outputs/train_bias_multi_taxon_ae_cifar10_%j.out
#SBATCH --error=slurm/slurm_errors/train_bias_multi_taxon_ae_cifar10_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# ── Environment ───────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ─────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${1:-configs/cifar10/bias_multi_taxon_ae_cifar10.json}"

# ── Run training ──────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_bias_multi_taxon_ae_cifar10 (job $SLURM_JOB_ID) config=$CONFIG"

python src/train/cifar10/train_bias_multi_taxon_ae.py \
    --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_bias_multi_taxon_ae_cifar10 (job $SLURM_JOB_ID)"
