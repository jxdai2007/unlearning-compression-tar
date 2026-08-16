#!/bin/bash
#SBATCH --job-name=tar-wmdpk5-__VARIANT__
#SBATCH --output=/scratch/USER/results/02-extension/__VARIANT__=wmdp_bio_k5.log
#SBATCH --error=/scratch/USER/results/02-extension/__VARIANT__=wmdp_bio_k5.err
#SBATCH --partition=sporc
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=60G

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

cd /home/USER/jollen-rapid-project/projects/02-compression-tampering

echo "WMDP-bio k=5 for __VARIANT__ at $(date)"

/home/USER/miniforge3/envs/jrp/bin/python wmdp_k5_eval.py \
    --model-args "__MODEL_ARGS__" \
    --out "/scratch/USER/results/02-extension/__VARIANT__=wmdp_bio/k5"

echo "Completed at $(date)"
