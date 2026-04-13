#!/bin/bash
#SBATCH --job-name=compare_pythia_saebench
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/compare_pythia_saebench_%j.out
#SBATCH --error=slurm/slurm_errors/compare_pythia_saebench_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ── Compare Pythia-160M SAE variants (ours + baselines) ──────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Pythia-160M comparison (job $SLURM_JOB_ID)"

python src/eval/compare_saebench.py \
    --models-dir outputs/saebench/pythia160m_layer8 \
    --model-name pythia-160m-deduped \
    --include-baselines \
    --baseline-width 4k \
    --eval-types core sparse_probing scr tpp absorption \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished Pythia-160M comparison (job $SLURM_JOB_ID)"
