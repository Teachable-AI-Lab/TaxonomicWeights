#!/bin/bash
#SBATCH --job-name=eval_topk_multi_taxon_saebench
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/eval_topk_multi_taxon_saebench_%j.out
#SBATCH --error=slurm/slurm_errors/eval_topk_multi_taxon_saebench_%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-08:00:00

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate taxon-weights
export PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
cd /nethome/ksingara3/flash/TaxonomicWeights
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ── Evaluate TopKMultiTaxonSAE on SAEBench ───────────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TopKMultiTaxonSAE SAEBench evaluation (job $SLURM_JOB_ID)"

python src/eval/run_saebench_evals.py \
    --config configs/saebench/topk_multi_taxon_sae_pythia160m.json \
    --eval-types core sparse_probing scr tpp absorption taxonomy \
    --save-activations

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished TopKMultiTaxonSAE SAEBench evaluation (job $SLURM_JOB_ID)"
