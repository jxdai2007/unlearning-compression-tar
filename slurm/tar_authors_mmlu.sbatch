#!/bin/bash
#SBATCH --job-name=jrp02-tarmmlu
#SBATCH --partition=sporc-gpu
#SBATCH --account=reu-aisocial
#SBATCH --qos=qos_tier3
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/jrp02-tarmmlu-%j.out
set -x
# Project 02 retain-side replication: the TAR authors' own MMLU harness (5-shot, Hendrycks
# data.tar) against lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2. This is the leg that can resolve
# the retain discrepancy: n=12,749 -> 1 SE ~0.44 pp, so the +3.11 pp gap vs our lm-eval number
# is ~7 SE if it stands. Their harness, their protocol, their defaults (nfe=5, seed 42).
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/USER/hf
export TOKENIZERS_PARALLELISM=false
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export PATH=/home/USER/miniforge3/envs/jrp/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --module so the package-relative imports resolve: eval.py does `from ...modules.utils import
# fix_seed`, which is 3 levels up -- so the REPO must be an importable package, not just
# red_teaming. The repo dir has a hyphen (not importable), so an underscore symlink
# tamper_resistance -> tamper-resistance sits in /scratch/USER/repos and we launch from
# there: module tamper_resistance.red_teaming.mmlu_eval.eval. Running -m from inside the repo
# ("red_teaming.mmlu_eval.eval") dies with "attempted relative import beyond top-level package"
# (verified the hard way, job 21457644). Config is the authors' own 2-GPU FSDP config.
# model_type: the TAR checkpoint ships weights and config ONLY -- no tokenizer files (verified:
# the cache snapshot holds config.json, generation_config.json, 7 shards, nothing else). The
# authors' default model_type (meta-llama/Meta-Llama-3-8B-Instruct) is gated; NousResearch's
# mirror is the substitution run_gate.py already uses, with the provenance check logged there
# (tokenizer.json byte-identical, len(tokenizer)=128256). --eos_pad_token is required for
# Llama-3 (their README). Output mmlu_jrp02_tarv2_authors/ lands under /scratch/USER/repos/.
cd /scratch/USER/repos
accelerate launch --config_file /scratch/USER/repos/tamper-resistance/configs/accel_config_2_gpu.yaml \
    --module tamper_resistance.red_teaming.mmlu_eval.eval \
    --model_name lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2 \
    --model_type NousResearch/Meta-Llama-3-8B-Instruct \
    --eos_pad_token \
    --path_to_data /scratch/USER/repos/tamper-resistance/red_teaming/mmlu_eval/data \
    --save_file_dir jrp02_tarv2_authors \
    --batch_size 4
# batch_size 4, not the authors' default 8: their default assumes 80GB cards. At bs=8 the fp32
# conversion of full-sequence logits (8 x 4096 x 128256 x 4B = 15.66 GiB, accelerate
# operations.py _convert_to_fp32) OOMs beside the sharded model on a 40GB A100 (verified, job
# 21458944). bs only affects throughput and padding, not per-sample scores.
echo "TARMMLU_EXIT=$?"
