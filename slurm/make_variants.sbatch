#!/bin/bash
#SBATCH --job-name=tar-make-variants
#SBATCH --output=/scratch/USER/results/02-extension/make_variants.log
#SBATCH --error=/scratch/USER/results/02-extension/make_variants.err
#SBATCH --partition=sporc
#SBATCH --time=02:00:00
#SBATCH --mem=60G

# CPU-only: pruning needs no GPU; skipping gres frees GPU queue slots.
# Env-fix stack (verbatim from notes/cluster-ops/README.md)
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/USER/hf
export TOKENIZERS_PARALLELISM=false
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME=/home/USER/miniforge3/envs/jrp/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/USER/miniforge3/envs/jrp/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/home/USER/miniforge3/envs/jrp/lib
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE=/scratch/USER/flashinfer_cache
export TRITON_CACHE_DIR=/scratch/USER/triton_cache

# Create output dirs
mkdir -p /scratch/USER/models/tar_mag10
mkdir -p /scratch/USER/models/tar_mag30
mkdir -p /scratch/USER/models/tar_mag50
mkdir -p /scratch/USER/results/02-extension

# Run pruning jobs sequentially
cd /home/USER/jollen-rapid-project

echo "Starting magnitude pruning variants at $(date)"

# 10% sparsity
echo "Pruning 10% sparsity..."
/home/USER/miniforge3/envs/jrp/bin/python projects/02-compression-tampering/compress_variants.py \
    --sparsity 0.10 \
    --model lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 \
    --out /scratch/USER/models/tar_mag10

# 30% sparsity
echo "Pruning 30% sparsity..."
/home/USER/miniforge3/envs/jrp/bin/python projects/02-compression-tampering/compress_variants.py \
    --sparsity 0.30 \
    --model lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 \
    --out /scratch/USER/models/tar_mag30

# 50% sparsity
echo "Pruning 50% sparsity..."
/home/USER/miniforge3/envs/jrp/bin/python projects/02-compression-tampering/compress_variants.py \
    --sparsity 0.50 \
    --model lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 \
    --out /scratch/USER/models/tar_mag50

echo "Completed at $(date)"
