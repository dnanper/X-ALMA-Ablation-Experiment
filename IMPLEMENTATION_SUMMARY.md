# X-ALMA Ablation Study - Implementation Summary

## Changes Made

### 1. Modified Files

#### `utils/arguments.py`

- Added new argument: `--custom_base_model`
- This allows specifying a custom base model for ablation studies
- When set, `model_name_or_path` becomes the adapter path

#### `run_llmmt.py`

- Imported `load_model_specific` function
- Added logic to check `custom_base_model` argument
- If set, uses `load_model_specific()` instead of `load_model()`

#### `utils/utils.py` (created earlier)

- Added `load_model_specific()` function
- Loads base model and adapter separately
- Merges them using PEFT

### 2. New Scripts Created

#### Evaluation Scripts:

- `evals/xalma_ablation_unpretrained.sh` - Test with unpretrained base
- `evals/xalma_ablation_pretrained.sh` - Test with pretrained base
- `evals/xalma_ablation_comparison.sh` - Run both experiments
- `evals/xalma_quick_test.sh` - Quick test with 10 samples

#### Documentation:

- `ABLATION_USAGE.md` - Complete usage guide

## How It Works

### Normal X-ALMA Usage (No Changes):

```bash
accelerate launch run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --do_predict ...
```

→ Loads merged model as usual

### Ablation Study Usage (NEW):

```bash
accelerate launch run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model meta-llama/Llama-2-13b-hf \
    --do_predict ...
```

→ Loads `meta-llama/Llama-2-13b-hf` as base, applies `X-ALMA-13B-Group5` as adapter

## Usage Examples

### Quick Start (10 samples):

```bash
# Test unpretrained
bash evals/xalma_quick_test.sh unpretrained

# Test pretrained
bash evals/xalma_quick_test.sh pretrained
```

### Full Evaluation:

```bash
# Run both experiments
bash evals/xalma_ablation_comparison.sh
```

### Custom Evaluation:

```bash
accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model meta-llama/Llama-2-13b-hf \
    --do_predict \
    --language_pairs en-cs,cs-en \
    --override_test_data_path haoranxu/FLORES-200 \
    --output_dir ./outputs/my_test \
    --chat_style \
    --bf16
```

## Expected Results

| Experiment   | Base Model        | Expected BLEU        |
| ------------ | ----------------- | -------------------- |
| Unpretrained | Llama-2-13b-hf    | Lower (e.g., 25-30)  |
| Pretrained   | ALMA-13B-Pretrain | Higher (e.g., 35-40) |

The gap between these scores shows the **contribution of pretrain stage**.

## Benefits

1. **Backward Compatible**: Original usage still works
2. **Flexible**: Can test any base model with any adapter
3. **Easy to Use**: Just add `--custom_base_model` argument
4. **Integrated**: Works with all existing features (DeepSpeed, multi-GPU, etc.)

## Base Models You Can Try

- `meta-llama/Llama-2-13b-hf` - Original Llama-2 (unpretrained)
- `haoranxu/ALMA-13B-Pretrain` - ALMA pretrained (normal)
- `meta-llama/Llama-2-13b-chat-hf` - Instruction-tuned version
- `meta-llama/Llama-2-7b-hf` - 7B version (with 7B adapters)

## Adapters Available

All X-ALMA-13B groups:

- Group 1: Germanic languages (de, nl, da, is, no, sv, af)
- Group 2: Romance languages (es, pt, it, ca, ro, gl)
- Group 3: Slavic languages (ru, uk, bg, mk, sr)
- Group 4: Southeast Asian + French (vi, th, id, ms, mg, fr)
- Group 5: Eastern European (cs, pl, hu, el, lt, lv)
- Group 6: East Asian + Finnic (zh, ja, ko, ka, fi, et)
- Group 7: Indian languages (hi, gu, mr, ne, ur)
- Group 8: Turkic + Arabic (tr, az, kk, ky, uz, ar, he, fa)

## Notes

- Base model and adapter must match in size (7B or 13B)
- Requires 24GB+ GPU memory
- Use `--bf16` or `--fp16` for memory efficiency
- Tokenizer is loaded from adapter path, not base model
