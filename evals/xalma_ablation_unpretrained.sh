#!/bin/bash

# Ablation Study: Test X-ALMA with UNPRETRAINED base model
# This measures the contribution of X-ALMA module WITHOUT pretrain stage
# Expected: Lower performance compared to pretrained base

OUTPUT_DIR=${1:-"./outputs/xalma_ablation_unpretrained"}

echo "=========================================="
echo "X-ALMA Ablation Study: UNPRETRAINED Base"
echo "=========================================="
echo "Base Model: meta-llama/Llama-2-13b-hf (NO pretrain)"
echo "Adapter:    haoranxu/X-ALMA-13B-Group5"
echo "Output:     $OUTPUT_DIR"
echo "=========================================="
echo ""

accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model meta-llama/Llama-2-13b-hf \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs en-cs,cs-en \
    --mmt_data_path placeholder \
    --override_test_data_path haoranxu/WMT23-Test \
    --per_device_eval_batch_size 1 \
    --output_dir $OUTPUT_DIR \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --chat_style

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
