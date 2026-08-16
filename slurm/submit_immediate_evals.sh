#!/bin/bash
# Submit 6 immediate eval jobs (intact, 8bit, 4bit x 2 tasks)

cd /home/USER/jollen-rapid-project/projects/02-compression-tampering/slurm

# Intact + MMLU
sed 's|__MODEL_ARGS__|pretrained=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|mmlu|g; s|__VARIANT__|intact|g' eval_variant.sbatch | sbatch

# Intact + WMDP-Bio
sed 's|__MODEL_ARGS__|pretrained=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct|g; s|__TASK__|wmdp_bio|g; s|__VARIANT__|intact|g' eval_variant.sbatch | sbatch

# 8bit + MMLU
sed 's|__MODEL_ARGS__|pretrained=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct,load_in_8bit=True|g; s|__TASK__|mmlu|g; s|__VARIANT__|8bit|g' eval_variant.sbatch | sbatch

# 8bit + WMDP-Bio
sed 's|__MODEL_ARGS__|pretrained=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct,load_in_8bit=True|g; s|__TASK__|wmdp_bio|g; s|__VARIANT__|8bit|g' eval_variant.sbatch | sbatch

# 4bit + MMLU
sed 's|__MODEL_ARGS__|pretrained=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct,load_in_4bit=True,bnb_4bit_quant_type=nf4|g; s|__TASK__|mmlu|g; s|__VARIANT__|4bit|g' eval_variant.sbatch | sbatch

# 4bit + WMDP-Bio
sed 's|__MODEL_ARGS__|pretrained=lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2,dtype=float16,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct,load_in_4bit=True,bnb_4bit_quant_type=nf4|g; s|__TASK__|wmdp_bio|g; s|__VARIANT__|4bit|g' eval_variant.sbatch | sbatch

echo "Submitted 6 immediate eval jobs"
