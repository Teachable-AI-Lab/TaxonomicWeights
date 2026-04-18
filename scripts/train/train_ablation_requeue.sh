#!/bin/bash
#SBATCH --job-name=train_ablation
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/train_ablation_%j.out
#SBATCH --error=slurm/slurm_errors/train_ablation_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@120

# Usage: sbatch scripts/train/train_ablation_requeue.sh <train_script.py> <config.json>
#   e.g. sbatch scripts/train/train_ablation_requeue.sh \
#            src/train/celeba_hq/train_multi_taxon_ae.py \
#            configs/celeba_hq/ablations/multi_taxon/multi_taxon_ae_celeba_hq_r18_dkl_1e-02_temp_0p001.json

TRAIN_SCRIPT="${1:?Usage: sbatch train_ablation_requeue.sh <train_script.py> <config.json>}"
CONFIG="${2:?Usage: sbatch train_ablation_requeue.sh <train_script.py> <config.json>}"

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Auto-resume: find latest.pt under the config's output_dir subtree ─────────
BASE_OUT=$(python -c "
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get('output', {}).get('output_dir', ''))
" "$CONFIG")

RESUME_ARG=""
if [[ -n "$BASE_OUT" ]]; then
    LATEST=$(ls -t "${BASE_OUT}"*/checkpoints/latest.pt 2>/dev/null | head -1)
    if [[ -n "$LATEST" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming from $LATEST"
        RESUME_ARG="--resume $LATEST"
    fi
fi

# ── Run training ───────────────────────────────────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $TRAIN_SCRIPT (job $SLURM_JOB_ID) config=$CONFIG"

python "$TRAIN_SCRIPT" --config "$CONFIG" $RESUME_ARG

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $TRAIN_SCRIPT (job $SLURM_JOB_ID)"
