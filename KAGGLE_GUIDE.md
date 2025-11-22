# X-ALMA Ablation Study on Kaggle

## Setup Guide

### 1. Create New Kaggle Notebook

- Go to https://www.kaggle.com/code
- Click "New Notebook"
- Choose **GPU** accelerator (T4 or P100)

### 2. Install Dependencies

```python
# Cell 1: Clone repository
!git clone https://github.com/dnanper/X-ALMA-Ablation-Experiment.git
%cd X-ALMA-Ablation-Experiment
```

```python
# Cell 2: Install dependencies
!bash install_alma.sh
```

### 3. Run Ablation Study

#### Option A: Unpretrained Base (Test pretrain contribution)

```python
# Cell 3: Run with unpretrained base
!accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model meta-llama/Llama-2-13b-hf \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs en-cs,cs-en \
    --mmt_data_path placeholder \
    --override_test_data_path haoranxu/WMT23-Test \
    --per_device_eval_batch_size 1 \
    --output_dir ./outputs/unpretrained \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --chat_style
```

#### Option B: Pretrained Base (Normal X-ALMA)

```python
# Cell 4: Run with pretrained base
!accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model haoranxu/ALMA-13B-Pretrain \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs en-cs,cs-en \
    --mmt_data_path placeholder \
    --override_test_data_path haoranxu/WMT23-Test \
    --per_device_eval_batch_size 1 \
    --output_dir ./outputs/pretrained \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --chat_style
```

### 4. View Results

```python
# Cell 5: Display translations from unpretrained base
print("=" * 80)
print("UNPRETRAINED BASE RESULTS (cs-en)")
print("=" * 80)
with open('./outputs/unpretrained/test-cs-en.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:10], 1):
        print(f"{i}. {line.strip()}")
```

```python
# Cell 6: Display translations from pretrained base
print("=" * 80)
print("PRETRAINED BASE RESULTS (cs-en)")
print("=" * 80)
with open('./outputs/pretrained/test-cs-en.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:10], 1):
        print(f"{i}. {line.strip()}")
```

### 5. Calculate BLEU Scores (Optional)

```python
# Cell 7: Install sacrebleu
!pip install sacrebleu
```

```python
# Cell 8: Calculate BLEU
# Note: You need reference file from WMT23
# This is just an example - you'll need to get the actual references

from sacrebleu import corpus_bleu

# Read hypotheses
with open('./outputs/unpretrained/test-cs-en.txt', 'r') as f:
    hyps_unpretrained = [line.strip() for line in f.readlines()]

with open('./outputs/pretrained/test-cs-en.txt', 'r') as f:
    hyps_pretrained = [line.strip() for line in f.readlines()]

# If you have references:
# refs = [...]  # Your reference translations
# bleu_unpretrained = corpus_bleu(hyps_unpretrained, [refs])
# bleu_pretrained = corpus_bleu(hyps_pretrained, [refs])
#
# print(f"BLEU (Unpretrained): {bleu_unpretrained.score:.2f}")
# print(f"BLEU (Pretrained): {bleu_pretrained.score:.2f}")
# print(f"Gap: {bleu_pretrained.score - bleu_unpretrained.score:.2f}")
```

### 6. Download Results

```python
# Cell 9: Zip and download results
!zip -r results.zip outputs/
from IPython.display import FileLink
FileLink('results.zip')
```

## Quick Test (10 samples only)

If you want to test quickly with just 10 samples:

```python
# Quick test - unpretrained
!accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model meta-llama/Llama-2-13b-hf \
    --do_predict \
    --low_cpu_mem_usage \
    --language_pairs cs-en \
    --mmt_data_path placeholder \
    --override_test_data_path haoranxu/WMT23-Test \
    --max_test_samples 10 \
    --per_device_eval_batch_size 1 \
    --output_dir ./outputs/quick_unpretrained \
    --predict_with_generate \
    --max_new_tokens 256 \
    --max_source_length 256 \
    --bf16 \
    --seed 42 \
    --num_beams 5 \
    --overwrite_cache \
    --overwrite_output_dir \
    --chat_style
```

## Tips for Kaggle

1. **GPU Selection**: Use T4 (free) or P100 (better performance)

2. **Session Time**: Kaggle has 12-hour limit for GPU sessions

   - Quick test (~10-15 min)
   - Full WMT23 (~1-2 hours per experiment)

3. **Disk Space**: Kaggle gives 20GB workspace

   - Models will download to cache (~26GB total for both base models)
   - May need to run one at a time and clear cache between runs

4. **Clear Cache Between Runs**:

```python
!rm -rf ~/.cache/huggingface/hub
```

5. **Internet Access**: Enable Internet in Kaggle settings

   - Settings → Internet → On

6. **Persistence**: Save outputs to Kaggle Datasets to persist results

```python
!mkdir -p /kaggle/working/outputs
# Results will be available in Output tab
```

## Kaggle Settings Checklist

- ✅ Accelerator: GPU (T4 or P100)
- ✅ Internet: On
- ✅ Persistence: On (if you want to save)

## Expected Runtime

| Experiment | Samples | Time (T4) | Time (P100) |
| ---------- | ------- | --------- | ----------- |
| Quick test | 10      | ~10 min   | ~5 min      |
| Full WMT23 | ~2000   | ~2 hours  | ~1 hour     |

## Memory Usage

- Unpretrained base: ~24GB GPU + ~15GB System
- Pretrained base: ~24GB GPU + ~15GB System

If OOM error, reduce batch size or use fp16:

```python
--per_device_eval_batch_size 1
--fp16  # instead of --bf16
```
