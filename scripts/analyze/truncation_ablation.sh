#!/bin/bash
#SBATCH --job-name=truncation_ablation
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/truncation_ablation_%j.out
#SBATCH --error=slurm/slurm_errors/truncation_ablation_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Usage: sbatch scripts/analyze/truncation_ablation.sh <config.json>
CONFIG="${1:-}"

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting truncation_ablation config=$CONFIG (job $SLURM_JOB_ID)"

python -m src.analyze.truncation_ablation --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished truncation_ablation config=$CONFIG (job $SLURM_JOB_ID)"
