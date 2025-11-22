#!/bin/bash

# Quick test script - translate just a few sentences to verify setup
# This is faster than running full evaluation on FLORES-200

echo "=========================================="
echo "Quick Test: X-ALMA Ablation Study"
echo "=========================================="
echo ""

EXPERIMENT=${1:-"unpretrained"}

case $EXPERIMENT in
    "unpretrained")
        echo "Testing UNPRETRAINED base (meta-llama/Llama-2-13b-hf)"
        BASE_MODEL="meta-llama/Llama-2-13b-hf"
        OUTPUT_DIR="./outputs/quick_test_unpretrained"
        ;;
    "pretrained")
        echo "Testing PRETRAINED base (haoranxu/ALMA-13B-Pretrain)"
        BASE_MODEL="haoranxu/ALMA-13B-Pretrain"
        OUTPUT_DIR="./outputs/quick_test_pretrained"
        ;;
    *)
        echo "Usage: bash $0 [unpretrained|pretrained]"
        exit 1
        ;;
esac

echo "Base Model: $BASE_MODEL"
echo "Adapter:    haoranxu/X-ALMA-13B-Group5"
echo "Output:     $OUTPUT_DIR"
echo ""
echo "This will translate only the first 10 examples for quick testing."
echo "=========================================="
echo ""

accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model $BASE_MODEL \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs cs-en \
    --mmt_data_path placeholder \
    --override_test_data_path haoranxu/WMT23-Test \
    --max_test_samples 10 \
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
echo "Quick test completed!"
echo "Check the output: $OUTPUT_DIR/test-cs-en.txt"
echo "=========================================="
