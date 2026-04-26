#!/bin/bash
#SBATCH --job-name=train_bottleneck_topk_multi_taxon_ae_imagenet
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_bottleneck_topk_multi_taxon_ae_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/train_bottleneck_topk_multi_taxon_ae_imagenet_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ── Environment ───────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ─────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="${1:-configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K4_L6_gate1.json}"

# ── Run training ──────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_bottleneck_topk_multi_taxon_ae_imagenet (job $SLURM_JOB_ID) config=$CONFIG"

python src/train/imagenet/train_bottleneck_topk_multi_taxon_ae.py \
    --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_bottleneck_topk_multi_taxon_ae_imagenet (job $SLURM_JOB_ID)"
