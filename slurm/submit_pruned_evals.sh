#!/bin/bash
# Submit 6 pruned variant eval jobs (mag10/30/50 x 2 tasks)
# Run after make_variants job (21460506) completes

cd /home/USER/jollen-rapid-project/projects/02-compression-tampering/slurm

# mag10 + MMLU
sed 's|__MODEL_ARGS__|pretrained=/scratch/USER/models/tar_mag10,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|mmlu|g; s|__VARIANT__|mag10|g' eval_variant.sbatch | sbatch

# mag10 + WMDP-Bio
sed 's|__MODEL_ARGS__|pretrained=/scratch/USER/models/tar_mag10,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|wmdp_bio|g; s|__VARIANT__|mag10|g' eval_variant.sbatch | sbatch

# mag30 + MMLU
sed 's|__MODEL_ARGS__|pretrained=/scratch/USER/models/tar_mag30,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|mmlu|g; s|__VARIANT__|mag30|g' eval_variant.sbatch | sbatch

# mag30 + WMDP-Bio
sed 's|__MODEL_ARGS__|pretrained=/scratch/USER/models/tar_mag30,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|wmdp_bio|g; s|__VARIANT__|mag30|g' eval_variant.sbatch | sbatch

# mag50 + MMLU
sed 's|__MODEL_ARGS__|pretrained=/scratch/USER/models/tar_mag50,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|mmlu|g; s|__VARIANT__|mag50|g' eval_variant.sbatch | sbatch

# mag50 + WMDP-Bio
sed 's|__MODEL_ARGS__|pretrained=/scratch/USER/models/tar_mag50,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|wmdp_bio|g; s|__VARIANT__|mag50|g' eval_variant.sbatch | sbatch

echo "Submitted 6 pruned variant eval jobs"
