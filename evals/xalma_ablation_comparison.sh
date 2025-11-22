#!/bin/bash

# Comparison script: Run both pretrained and unpretrained experiments
# Then compare the results to measure pretrain contribution

echo "=========================================="
echo "X-ALMA Ablation Study: Full Comparison"
echo "=========================================="
echo ""
echo "This script will run TWO experiments:"
echo "  1. X-ALMA with UNPRETRAINED base (Llama-2-13b-hf)"
echo "  2. X-ALMA with PRETRAINED base (ALMA-13B-Pretrain)"
echo ""
echo "You can then compare BLEU/COMET scores to measure"
echo "the contribution of the pretrain stage."
echo ""
echo "=========================================="
echo ""

# Run unpretrained experiment
echo ">>> Running Experiment 1: UNPRETRAINED base..."
bash evals/xalma_ablation_unpretrained.sh ./outputs/xalma_unpretrained

echo ""
echo ">>> Experiment 1 completed."
echo ""
echo ">>> Running Experiment 2: PRETRAINED base..."
bash evals/xalma_ablation_pretrained.sh ./outputs/xalma_pretrained

echo ""
echo "=========================================="
echo "Both experiments completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Unpretrained: ./outputs/xalma_unpretrained/"
echo "  Pretrained:   ./outputs/xalma_pretrained/"
echo ""
echo "Compare the test-*.txt files to see the difference!"
echo ""
echo "To calculate BLEU scores, you can use:"
echo "  cat ./outputs/xalma_unpretrained/test-cs-en.txt | sacrebleu reference.txt"
echo "  cat ./outputs/xalma_pretrained/test-cs-en.txt | sacrebleu reference.txt"
echo ""
echo "=========================================="
