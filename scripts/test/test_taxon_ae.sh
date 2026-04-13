#!/bin/bash
#SBATCH --job-name=test_taxon_ae
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/test_taxon_ae_%j.out
#SBATCH --error=slurm/slurm_errors/test_taxon_ae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Run test ───────────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting test_taxon_ae (job $SLURM_JOB_ID)"

python src/test/test_taxon_ae.py \
    --image-size 256 \
    --batch-size 2 \
    --resnet-variant 18 \
    --stage-taxonomy-layers 5 6 7 8 \
    --stage-strides 1 2 2 2 \
    --temperature 1.0 \
    --device cuda

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished test_taxon_ae (job $SLURM_JOB_ID)"
