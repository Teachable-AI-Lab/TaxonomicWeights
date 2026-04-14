#!/bin/bash
#SBATCH --job-name=train_taxon_sae
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_taxon_sae_%j.out
#SBATCH --error=slurm/slurm_errors/train_taxon_sae_%j.err
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

# ── Train TaxonSAE (Pythia-160M, layer 8) ────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TaxonSAE training (job $SLURM_JOB_ID)"

python src/train/saebench/train_taxon_sae.py \
    --config configs/saebench/taxon_sae_pythia160m.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished TaxonSAE training (job $SLURM_JOB_ID)"
