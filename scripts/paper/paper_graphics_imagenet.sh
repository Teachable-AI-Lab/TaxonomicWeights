#!/bin/bash
#SBATCH --job-name=paper_graphics_imagenet
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/paper_graphics_imagenet_%j.out
#SBATCH --error=slurm/slurm_errors/paper_graphics_imagenet_%j.err
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
DATA_ROOT="${DATA_ROOT:-./data/imagenet}"
OUT_DIR="${OUT_DIR:-paper_graphics_imagenet}"
MAX_IMAGES="${MAX_IMAGES:-3000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
N_PROBE_TRAIN="${N_PROBE_TRAIN:-5000}"
N_TSNE_SAMPLES="${N_TSNE_SAMPLES:-3000}"
STEER_SCALE="${STEER_SCALE:-10.0}"
N_STEER_CLASSES="${N_STEER_CLASSES:-6}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. "--skip-hierarchy --force-recompute"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting paper_graphics_imagenet (job $SLURM_JOB_ID)"
echo "  data-root:     $DATA_ROOT"
echo "  out-dir:       $OUT_DIR"
echo "  max-images:    $MAX_IMAGES"
echo "  n-probe-train: $N_PROBE_TRAIN"
echo "  n-tsne-samples:$N_TSNE_SAMPLES"
echo "  steer-scale:   $STEER_SCALE"
echo "  n-steer-classes:$N_STEER_CLASSES"

python src/paper/paper_graphics_imagenet.py \
    --data-root        "$DATA_ROOT"       \
    --out-dir          "$OUT_DIR"         \
    --max-images       "$MAX_IMAGES"      \
    --batch-size       "$BATCH_SIZE"      \
    --num-workers      "$NUM_WORKERS"     \
    --image-size       "$IMAGE_SIZE"      \
    --n-probe-train    "$N_PROBE_TRAIN"   \
    --n-tsne-samples   "$N_TSNE_SAMPLES"  \
    --steer-scale      "$STEER_SCALE"     \
    --n-steer-classes  "$N_STEER_CLASSES" \
    $EXTRA_ARGS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished paper_graphics_imagenet (job $SLURM_JOB_ID)"
