#!/bin/bash
#SBATCH --job-name=analyze_nodkl_ablation
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/analyze_nodkl_ablation_%j.out
#SBATCH --error=slurm/slurm_errors/analyze_nodkl_ablation_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR="${1:-./outputs/celeba_hq/taxon_ae_celeba_hq_r18_nodkl_ablation}"

# ── Run analysis ───────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting analyze_nodkl_ablation (job $SLURM_JOB_ID) output_dir=$OUTPUT_DIR"

python src/analyze/analyze_nodkl_ablation.py \
    --output-dir "$OUTPUT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished analyze_nodkl_ablation (job $SLURM_JOB_ID)"
