#!/bin/bash
#SBATCH --job-name=multi_taxon_trunc_ablation
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/multi_taxon_truncation_ablation_%j.out
#SBATCH --error=slurm/slurm_errors/multi_taxon_truncation_ablation_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Usage: sbatch scripts/analyze/multi_taxon_truncation_ablation.sh <config.json>
CONFIG="${1:-}"

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting multi_taxon_truncation_ablation config=$CONFIG (job $SLURM_JOB_ID)"

python -m src.analyze.multi_taxon_truncation_ablation --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished multi_taxon_truncation_ablation config=$CONFIG (job $SLURM_JOB_ID)"
