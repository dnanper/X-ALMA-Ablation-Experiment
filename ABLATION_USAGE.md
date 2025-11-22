# X-ALMA Ablation Study: Đo Contribution của Pretrain Stage

## Giới thiệu

Tool này giúp bạn đánh giá **mức độ đóng góp của pretrain stage** vào hiệu năng X-ALMA bằng cách thay đổi base model.

## Cách sử dụng

### Option 1: Chạy từng experiment riêng lẻ

#### Test với unpretrained base (đo contribution):

```bash
bash evals/xalma_ablation_unpretrained.sh
```

#### Test với pretrained base (normal X-ALMA):

```bash
bash evals/xalma_ablation_pretrained.sh
```

### Option 2: Chạy cả 2 và so sánh

```bash
bash evals/xalma_ablation_comparison.sh
```

### Option 3: Custom với accelerate launch

```bash
# Unpretrained base
accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model meta-llama/Llama-2-13b-hf \
    --do_predict \
    --language_pairs en-cs,cs-en \
    --override_test_data_path haoranxu/FLORES-200 \
    --output_dir ./outputs/my_experiment \
    --chat_style \
    --bf16

# Pretrained base
accelerate launch --config_file configs/deepspeed_eval_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path haoranxu/X-ALMA-13B-Group5 \
    --custom_base_model haoranxu/ALMA-13B-Pretrain \
    --do_predict \
    --language_pairs en-cs,cs-en \
    --override_test_data_path haoranxu/FLORES-200 \
    --output_dir ./outputs/my_experiment \
    --chat_style \
    --bf16
```

## Tham số quan trọng

### `--custom_base_model` (MỚI)

Chỉ định base model custom. Khi dùng argument này:

- `--model_name_or_path`: Sẽ được coi là **adapter path** (X-ALMA module)
- `--custom_base_model`: Base model để load adapter lên

**Ví dụ:**

```bash
--model_name_or_path haoranxu/X-ALMA-13B-Group5    # Adapter
--custom_base_model meta-llama/Llama-2-13b-hf      # Base (unpretrained)
```

### Base models có thể dùng:

- `meta-llama/Llama-2-13b-hf` - Unpretrained base từ Meta
- `haoranxu/ALMA-13B-Pretrain` - Pretrained base (normal X-ALMA)
- `meta-llama/Llama-2-7b-hf` - 7B version (nếu dùng adapter 7B)

### X-ALMA adapters:

- `haoranxu/X-ALMA-13B-Group{1-8}` - 8 language groups
- Ví dụ: Group 5 = Czech, Polish, Hungarian, Greek, Lithuanian, Latvian

## Kết quả

Sau khi chạy, kết quả sẽ nằm trong `output_dir/`:

- `test-cs-en.txt` - Translations từ Czech sang English
- `test-en-cs.txt` - Translations từ English sang Czech

## So sánh kết quả

```bash
# Tính BLEU scores
cat ./outputs/xalma_unpretrained/test-cs-en.txt | sacrebleu reference.txt
cat ./outputs/xalma_pretrained/test-cs-en.txt | sacrebleu reference.txt

# Hoặc dùng COMET
comet-score -s source.txt -t ./outputs/xalma_unpretrained/test-cs-en.txt -r reference.txt
comet-score -s source.txt -t ./outputs/xalma_pretrained/test-cs-en.txt -r reference.txt
```

## Kết quả mong đợi

- **Unpretrained base**: BLEU thấp hơn → Chứng minh pretrain quan trọng
- **Pretrained base**: BLEU cao hơn → Performance bình thường của X-ALMA
- **Gap càng lớn** → Pretrain stage càng đóng góp nhiều

## Code changes

### Trong `utils/arguments.py`:

- Thêm argument `--custom_base_model`

### Trong `run_llmmt.py`:

- Import `load_model_specific`
- Check `custom_base_model` argument
- Nếu có, load base + adapter riêng
- Nếu không, load model bình thường

### Trong `utils/utils.py`:

- Thêm hàm `load_model_specific()` để load base + adapter

## Yêu cầu hệ thống

- GPU: 24GB+ VRAM (RTX 3090/4090 hoặc A100)
- Dependencies: Đã cài qua `install_alma.sh`
- Disk space: ~50GB cho models

## Troubleshooting

**Q: Out of memory**

```bash
# Thử giảm batch size
--per_device_eval_batch_size 1

# Hoặc dùng fp16 thay vì bf16
--fp16  # thay cho --bf16
```

**Q: Size mismatch error**

- Đảm bảo base model và adapter cùng size (7B hoặc 13B)

**Q: Slow loading**

```bash
# Thêm flags này
--low_cpu_mem_usage
--multi_gpu_one_model  # Nếu có nhiều GPU
```
