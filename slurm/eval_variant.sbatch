#!/bin/bash
#SBATCH --job-name=tar-eval-__VARIANT__-__TASK__
#SBATCH --output=/scratch/USER/results/02-extension/__VARIANT__=__TASK__.log
#SBATCH --error=/scratch/USER/results/02-extension/__VARIANT__=__TASK__.err
#SBATCH --partition=sporc
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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

# Create output dir
mkdir -p /scratch/USER/results/02-extension/__VARIANT__=__TASK__

cd /home/USER/jollen-rapid-project

echo "Starting evaluation for __VARIANT__ on __TASK__ at $(date)"

# Run lm-eval with num_fewshot 5 (override task YAML silent pin)
/home/USER/miniforge3/envs/jrp/bin/python -m lm_eval \
    --model hf \
    --model_args __MODEL_ARGS__ \
    --tasks __TASK__ \
    --num_fewshot 5 \
    --output_path /scratch/USER/results/02-extension/__VARIANT__=__TASK__

echo "Completed at $(date)"

# Verify num_fewshot override took effect (silent bug check)
echo "Checking that n-shot override took effect..."
if /home/USER/miniforge3/envs/jrp/bin/python -c "
import json
import sys
path = '/scratch/USER/results/02-extension/__VARIANT__=__TASK__/results.json'
try:
    with open(path) as f:
        data = json.load(f)
    # Check first task config
    for task_key in data['results']:
        if 'n-shot' in data['results'][task_key].get('config', {}):
            n_shot = data['results'][task_key]['config']['n-shot']
            if n_shot != 5:
                print(f'ERROR: n-shot is {n_shot}, not 5')
                sys.exit(1)
        break
    print('n-shot verification passed')
except Exception as e:
    print(f'Could not verify: {e}')
    sys.exit(1)
"; then
    echo "num_fewshot=5 verified in output"
else
    echo "WARNING: num_fewshot=5 not verified in output - silent override bug"
fi
