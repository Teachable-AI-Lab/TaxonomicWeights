#!/bin/bash
#SBATCH --job-name=analyze_baseline_ae
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/analyze_baseline_ae_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_baseline_ae_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=2:00:00
#SBATCH --qos=short

# Usage: sbatch scripts/analyze/analyze_baseline_ae.sh configs/baseline_ae_celeba_hq.json
#        sbatch scripts/analyze/analyze_baseline_ae.sh configs/baseline_ae_cifar10.json
CONFIG="${1:-configs/baseline_ae_celeba_hq.json}"

cd /nethome/ksingara3/flash/TaxonomicWeights

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_baseline_ae config=$CONFIG (job $SLURM_JOB_ID)"

python src/analyze/analyze_baseline_ae.py --config "$CONFIG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_baseline_ae config=$CONFIG (job $SLURM_JOB_ID)"
