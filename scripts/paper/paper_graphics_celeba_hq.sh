#!/bin/bash
#SBATCH --job-name=paper_graphics_celeba
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/paper_graphics_celeba_%j.out
#SBATCH --error=slurm/slurm_errors/paper_graphics_celeba_%j.err
#SBATCH --partition=tail-lab
#SBATCH --account=tail-lab
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-04:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights

# ── Default arguments (override from command line) ────────────────────────────
DATA_ROOT="${DATA_ROOT:-./data/celeba_hq}"
CELEBA_ATTRS_ROOT="${CELEBA_ATTRS_ROOT:-./data/celeba_fallback}"
OUT_DIR="${OUT_DIR:-paper_graphics_celeba_hq}"
MAX_IMAGES="${MAX_IMAGES:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. "--skip-hierarchy --force-recompute"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting paper_graphics_celeba_hq (job $SLURM_JOB_ID)"
echo "  data-root:        $DATA_ROOT"
echo "  celeba-attrs-root:$CELEBA_ATTRS_ROOT"
echo "  out-dir:          $OUT_DIR"
echo "  max-images:       $MAX_IMAGES"

python src/paper/paper_graphics_celeba_hq.py \
    --data-root       "$DATA_ROOT"            \
    --celeba-attrs-root "$CELEBA_ATTRS_ROOT"  \
    --out-dir         "$OUT_DIR"              \
    --max-images      "$MAX_IMAGES"           \
    --batch-size      "$BATCH_SIZE"           \
    --num-workers     "$NUM_WORKERS"          \
    --image-size      "$IMAGE_SIZE"           \
    $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished paper_graphics_celeba_hq (job $SLURM_JOB_ID)"
