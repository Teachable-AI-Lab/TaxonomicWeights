#!/bin/bash
#SBATCH --job-name=train_bottleneck_topk_taxon_ae_celeba_hq
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_bottleneck_topk_taxon_ae_celeba_hq_%j.out
#SBATCH --error=slurm/slurm_errors/train_bottleneck_topk_taxon_ae_celeba_hq_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
cd /nethome/ksingara3/flash/TaxonomicWeights

CONFIG="${1:-configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L6.json}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting train_bottleneck_topk_taxon_ae_celeba_hq (job $SLURM_JOB_ID) config=$CONFIG"
python src/train/celeba_hq/train_bottleneck_topk_taxon_ae.py --config "$CONFIG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished train_bottleneck_topk_taxon_ae_celeba_hq (job $SLURM_JOB_ID)"
