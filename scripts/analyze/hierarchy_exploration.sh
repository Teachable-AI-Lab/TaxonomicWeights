#!/bin/bash
#SBATCH --job-name=hierarchy_exploration
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/hierarchy_exploration_%j.out
#SBATCH --error=slurm/slurm_errors/hierarchy_exploration_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Usage: sbatch scripts/analyze/hierarchy_exploration.sh <config.json> [--num-batches N]
CONFIG="${1:-}"
shift || true

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting hierarchy_exploration config=$CONFIG (job $SLURM_JOB_ID)"

python -m src.analyze.hierarchy_exploration --config "$CONFIG" --num-batches 50 "$@"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished hierarchy_exploration config=$CONFIG (job $SLURM_JOB_ID)"
