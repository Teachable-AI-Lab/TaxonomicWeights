#!/bin/bash
#SBATCH --job-name=train_multi_taxon_ae_celeba_hq
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_multi_taxon_ae_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/train_multi_taxon_ae_celeba_hq_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00

# ── Environment ───────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ─────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${1:-configs/celeba_hq/multi_taxon_ae_celeba_hq.json}"

# ── Run training ──────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_multi_taxon_ae_celeba_hq (job $SLURM_JOB_ID) config=$CONFIG"

python src/train/celeba_hq/train_multi_taxon_ae.py \
    --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_multi_taxon_ae_celeba_hq (job $SLURM_JOB_ID)"
