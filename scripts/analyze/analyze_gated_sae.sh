#!/bin/bash
#SBATCH --job-name=analyze_gated_sae
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/analyze_gated_sae_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_gated_sae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Usage: sbatch scripts/analyze/analyze_gated_sae.sh configs/gated_sae_celeba_hq.json
#        sbatch scripts/analyze/analyze_gated_sae.sh configs/gated_sae_cifar10.json
CONFIG="${1:-configs/gated_sae_celeba_hq.json}"

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_gated_sae config=$CONFIG (job $SLURM_JOB_ID)"

python src/analyze/analyze_gated_sae.py --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_gated_sae config=$CONFIG (job $SLURM_JOB_ID)"
