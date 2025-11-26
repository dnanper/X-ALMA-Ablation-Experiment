import sacrebleu
from comet import download_model, load_from_checkpoint
import re

def read_and_clean_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for i, line in enumerate(lines):
        # 1. Xóa khoảng trắng đầu cuối
        text = line.strip()
        
        # 2. Xóa tag [/INST] nếu model lỡ sinh ra (Clean rác)
        # Regex này tìm cụm [/INST] và xóa nó đi
        text = re.sub(r'\[/?INST\]', '', text)
        
        # 3. Xóa khoảng trắng dư thừa sau khi xóa tag
        text = text.strip()
        
        # 4. Xử lý dòng trống
        if text == "":
            print(f"Warning: Line {i+1} is empty (Model generated nothing).")
            # Vẫn append chuỗi rỗng để giữ đúng thứ tự dòng với Reference
            cleaned_lines.append("") 
        else:
            cleaned_lines.append(text)
            
    return cleaned_lines
def calculate_bleu(hyp_file, ref_file):
    hypotheses = read_and_clean_file(hyp_file)
    with open(ref_file, 'r', encoding='utf-8') as f:
        references = [[line.strip()] for line in f]
    if len(hypotheses) != len(references):
        print(f"   CRITICAL ERROR: Line count mismatch!")
        print(f"   Hyp: {len(hypotheses)}")
        print(f"   Ref: {len(references)}")
        print("   -> Scores will be meaningless due to misalignment.")
        # Cắt bớt để chạy tạm (nhưng kết quả sẽ không tin cậy)
        min_len = min(len(hypotheses), len(references))
        hypotheses = hypotheses[:min_len]
        references = references[:min_len]
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return bleu.score

def calculate_comet(src_file, hyp_file, ref_file, model_path):
    hypotheses = read_and_clean_file(hyp_file)
    with open(src_file, 'r', encoding='utf-8') as f:
        sources = [line.strip() for line in f]
    with open(ref_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    if len(hypotheses) != len(references):
        print(f"   CRITICAL ERROR: Line count mismatch!")
        print(f"   Hyp: {len(hypotheses)}")
        print(f"   Ref: {len(references)}")
        print("   -> Scores will be meaningless due to misalignment.")
        # Cắt bớt để chạy tạm (nhưng kết quả sẽ không tin cậy)
        min_len = min(len(hypotheses), len(references))
        hypotheses = hypotheses[:min_len]
        references = references[:min_len]
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
    model = load_from_checkpoint(model_path)
    scores = model.predict(data, batch_size=8, gpus=1)
    return scores.system_score

print("\n" + "="*60)
print("EVALUATION METRICS (BLEU + COMET-22)")
print("="*60)

print("Downloading COMET-22 model...")
comet_model_path = download_model("Unbabel/wmt22-comet-da")

for lang_pair in ["zh-en", "en-zh", "ru-en", "en-ru"]:
    print(f"\n{lang_pair.upper()}:")
    print("-" * 60)
    
    
    pretrained_bleu = calculate_bleu(
        f"/kaggle/input/x-alma-zh-en-and-en-zh-and-ru-en-and-en-ru/test-{lang_pair}.txt",
        f"/kaggle/input/x-alma-zh-en-and-en-zh-and-ru-en-and-en-ru/test-{lang_pair}.ref"
    )
    
    src_lang = lang_pair.split('-')[0]
    
    print("Calculating COMET scores (this may take a while)...")
    
    pretrained_comet = calculate_comet(
        f"/kaggle/input/x-alma-zh-en-and-en-zh-and-ru-en-and-en-ru/test-{lang_pair}.src",
        f"/kaggle/input/x-alma-zh-en-and-en-zh-and-ru-en-and-en-ru/test-{lang_pair}.txt",
        f"/kaggle/input/x-alma-zh-en-and-en-zh-and-ru-en-and-en-ru/test-{lang_pair}.ref",
        comet_model_path
    )
    
    print(f"\n  BLEU:")
    print(f"  Pretrained:   {pretrained_bleu:.2f}")
    
    print(f"\n  COMET-22:")
    print(f"  Pretrained:   {pretrained_comet:.4f}")

print("\n" + "="*60)
print("Note: COMET-22 is the primary metric used in ALMA paper")
print("="*60)