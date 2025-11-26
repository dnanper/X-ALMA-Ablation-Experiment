# Ablation Experiment

Thư mục này chứa các script để chạy thí nghiệm ablation study cho X-ALMA model.

## 📁 Cấu trúc

```
ablation/
├── runs/
│   ├── inference.py      # Script dịch văn bản bằng X-ALMA
│   ├── evaluation.py     # Script tính BLEU và COMET-22
│   └── requirements.txt  # Dependencies cần thiết
└── results/              # Thư mục chứa kết quả đầu ra
```

## 🔧 Cài đặt

### 1. Cài đặt dependencies

```bash
cd ablation/runs
pip install -r requirements.txt
```

### 2. Yêu cầu hệ thống

- **GPU**: Cần GPU với ít nhất 16GB VRAM (để load model 13B với 4-bit quantization)
- **Python**: 3.8+
- **CUDA**: Được cài đặt và hoạt động với PyTorch

## 🚀 Cách chạy

### Bước 1: Inference (Dịch văn bản)

File `inference.py` sẽ:

- Load model X-ALMA với adapter
- Dịch test set từ WMT23
- Lưu kết quả vào thư mục `outputs/`

**Chỉnh sửa tham số trong `inference.py`:**

```python
load_and_translate(
    base_model_path="haoranxu/ALMA-13B-Pretrain",  # Base model
    adapter_path="haoranxu/X-ALMA-13B-Group1",      # Adapter path
    output_dir="./outputs/pretrained",               # Thư mục đầu ra
    lang_pair="de-en",                               # Cặp ngôn ngữ
    max_samples=None,                                # None = dịch hết, hoặc số nguyên
    use_5shot=True,                                  # Dùng 5-shot prompting
    chat_style=True                                  # Dùng chat template
)
```

**Chạy inference:**

```bash
cd ablation/runs
python inference.py
```

**Các cặp ngôn ngữ được hỗ trợ:**

- `cs-en`, `en-cs` (Czech ↔ English)
- `de-en`, `en-de` (German ↔ English)
- `is-en`, `en-is` (Icelandic ↔ English)
- `ru-en`, `en-ru` (Russian ↔ English)
- `zh-en`, `en-zh` (Chinese ↔ English)

**Output files:**

- `test-{src}-{tgt}.txt` - Bản dịch
- `test-{lang_pair}.ref` - Reference (ground truth)
- `test-{lang_pair}.src` - Source text
- `test-{lang_pair}.debug.txt` - Raw model output (để debug)

### Bước 2: Evaluation (Tính metrics)

File `evaluation.py` sẽ tính:

- **BLEU score**: Metric đánh giá n-gram overlap
- **COMET-22**: Neural metric (primary metric trong ALMA paper)

**Chỉnh sửa đường dẫn trong `evaluation.py`:**

```python
# Thay đổi đường dẫn đến output files của bạn
pretrained_bleu = calculate_bleu(
    f"./outputs/pretrained/test-{lang_pair}.txt",  # Bản dịch
    f"./outputs/pretrained/test-{lang_pair}.ref"   # Reference
)

pretrained_comet = calculate_comet(
    f"./outputs/pretrained/test-{lang_pair}.src",  # Source
    f"./outputs/pretrained/test-{lang_pair}.txt",  # Bản dịch
    f"./outputs/pretrained/test-{lang_pair}.ref",  # Reference
    comet_model_path
)
```

**Chạy evaluation:**

```bash
cd ablation/runs
python evaluation.py
```

**Output mẫu:**

```
==============================================================
EVALUATION METRICS (BLEU + COMET-22)
==============================================================

DE-EN:
------------------------------------------------------------
  BLEU:
  Pretrained:   28.45

  COMET-22:
  Pretrained:   0.8234
```

## 📊 Kết quả

Kết quả sẽ được lưu trong `ablation/results/`:

- Screenshots các metrics
- Bảng so sánh giữa các variants

## 💡 Tips

### Debug khi model không dịch đúng:

1. Kiểm tra file `.debug.txt` để xem raw output của model
2. Kiểm tra số dòng giữa hypothesis và reference có khớp không
3. Thử giảm `max_samples` để test nhanh hơn

### Tối ưu memory:

- Giảm `batch_size` trong `calculate_comet()` nếu bị OOM
- Dùng `max_samples` để test trên subset nhỏ trước

### Thay đổi model/adapter:

```python
# Ví dụ: test với base model khác
load_and_translate(
    base_model_path="meta-llama/Llama-2-13b-hf",
    adapter_path="path/to/your/adapter",
    output_dir="./outputs/your_experiment",
    lang_pair="en-zh"
)
```

## 🔍 So sánh với thí nghiệm gốc

Scripts này replicate pipeline từ `run_llmmt.py` và `evals/` để:

- Dùng đúng prompt format (5-shot từ `Filtered-5-shot/`)
- Dùng đúng chat template
- Post-process output giống hệt như code gốc

## 📝 Notes

- **COMET-22 là metric chính** được dùng trong ALMA paper
- Model sẽ tự động download lần đầu (khoảng 1-2GB)
- Inference trên full test set mất khoảng 30-60 phút tùy GPU

## ❓ Troubleshooting

### Lỗi: "Line count mismatch"

→ Model sinh ra output rỗng cho một số câu. Kiểm tra `.debug.txt` và có thể cần điều chỉnh prompt.

### Lỗi: "CUDA out of memory"

→ Giảm `max_length`, `max_new_tokens` hoặc dùng GPU lớn hơn.

### Lỗi: "5-shot file not found"

→ Đảm bảo thư mục `human_written_data/Filtered-5-shot/` tồn tại với file `shots.{lang_pair}.json`

---

**Happy experimenting! 🚀**
