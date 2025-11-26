import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import os
import numpy as np

# Language name mapping from utils/utils.py
LANG_TABLE = {
    "en": "English",
    "cs": "Czech",
    "de": "German",
    "is": "Icelandic",
    "ru": "Russian",
    "zh": "Chinese",
}

def get_prompt(source_lang, target_lang, source_text, shots_eval_dict=None):
    """
    Generate prompt exactly as in run_llmmt.py -> get_prompt() -> get_prompt_few_shot()
    """
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    
    # With 5-shot examples if provided
    if shots_eval_dict and f"{source_lang}-{target_lang}" in shots_eval_dict:
        shots = shots_eval_dict[f"{source_lang}-{target_lang}"]
        prefix = f"Translate this from {src_fullname} to {tgt_fullname}:"
        shot_prompt = ""
        for shot in shots:
            shot_src = shot['source']
            shot_tgt = shot['target']
            shot_prompt += f"\n{src_fullname}: " + shot_src + f"\n{tgt_fullname}: " + shot_tgt
        suffix = f"\n{tgt_fullname}:"
        prompt = prefix + shot_prompt + f"\n{src_fullname}: " + source_text + suffix
    else:
        # 0-shot: simple format
        prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
        suffix = f"\n{tgt_fullname}:"
        prompt = prefix + source_text + suffix
    
    return prompt

def apply_chat_template(tokenizer, prompt):
    """
    Apply chat template if model uses chat format (for X-ALMA with chat_style=True)
    """
    chat_format = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=True)

def clean_outputstring(output, key_word, split_idx=1):
    """
    Post-process output exactly as in run_llmmt.py -> clean_outputstring()
    Extracts translation after the target language suffix
    """
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif len(out) > 1 and out[1].strip() != "":
            return out[1].strip()
        elif len(out) > 2 and out[2].strip() != "":
            return out[2].strip()
    except:
        pass
    
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        return ""

def load_and_translate(base_model_path, adapter_path, output_dir, lang_pair="cs-en", 
                      max_samples=None, use_5shot=True, chat_style=True):
    """
    Load model and translate WMT23 dataset exactly as run_llmmt.py does
    
    Args:
        base_model_path: Base model (e.g., "meta-llama/Llama-2-13b-hf")
        adapter_path: Adapter path (e.g., "haoranxu/X-ALMA-13B-Group5")
        output_dir: Output directory for results
        lang_pair: Language pair (e.g., "cs-en")
        max_samples: Max samples to translate (None for all)
        use_5shot: Use 5-shot prompting (default: True)
        chat_style: Use chat template (default: True for X-ALMA)
    """
    src_lang, tgt_lang = lang_pair.split("-")
    src_name = LANG_TABLE[src_lang]
    tgt_name = LANG_TABLE[tgt_lang]
    
    # Load 5-shot examples if enabled
    shots_eval_dict = None
    if use_5shot:
        # Load from human_written_data/Filtered-5-shot/
        import json
        shot_file = f"human_written_data/Filtered-5-shot/shots.{lang_pair}.json"
        if os.path.exists(shot_file):
            with open(shot_file, 'r', encoding='utf-8') as f:
                shots_eval_dict = {lang_pair: json.load(f)}
            print(f"Loaded 5-shot examples from {shot_file}")
        else:
            print(f"Warning: 5-shot file not found: {shot_file}, using 0-shot")
            use_5shot = False
    
    # Load model (same as load_model_specific in utils/utils.py)
    print(f"\n{'='*60}")
    print(f"Loading base: {base_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        device_map="auto",  
        torch_dtype=torch.float16,       
    )
    
    # Load tokenizer from adapter path
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, legacy=False, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Set padding side to left (same as tokenize_function_test)
    tokenizer.padding_side = "left"
    
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    
    # Load WMT23 dataset
    print(f"Loading WMT23-Test {lang_pair}...")
    dataset = load_dataset("haoranxu/WMT23-Test", lang_pair, split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Translating {len(dataset)} samples...")
    print(f"  - 5-shot: {use_5shot}")
    print(f"  - Chat template: {chat_style}")
    
    # Debug: check dataset keys
    if len(dataset) > 0:
        print(f"Dataset keys: {list(dataset[0].keys())}")
        # Check the structure of the first item
        first_item = dataset[0]
        if lang_pair in first_item:
            print(f"Item structure: {type(first_item[lang_pair])}")
            if isinstance(first_item[lang_pair], dict):
                print(f"Sub-keys: {list(first_item[lang_pair].keys())}")
    
    translations = []
    decoded_preds = []
    
    # Translate (same as trainer.predict logic in run_llmmt.py)
    for item in tqdm(dataset):
        # 1. Generate prompt (same as tokenize_function_test)
        # WMT23-Test dataset has nested structure: item[lang_pair][src_lang] and item[lang_pair][tgt_lang]
        pair_data = item[lang_pair]
        source_text = pair_data[src_lang]
        prompt = get_prompt(src_lang, tgt_lang, source_text, shots_eval_dict)
        
        # 2. Apply chat template if enabled
        if chat_style:
            prompt = apply_chat_template(tokenizer, prompt)
        
        # 3. Tokenize with left padding
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=256,  # max_source_length
            padding="max_length",
            truncation=True,
            add_special_tokens=True if not chat_style else False
        ).to(model.device)
        
        # 4. Generate (same as trainer.predict)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # data_args.max_new_tokens
                num_beams=1,         # data_args.num_beams
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 5. Decode (same as run_llmmt.py line 347-350)
        preds = outputs.cpu().numpy()
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_pred = tokenizer.decode(preds[0], skip_special_tokens=True)
        decoded_preds.append(decoded_pred)
        
        # 6. Post-process (same as run_llmmt.py line 352-363)
        decoded_pred = decoded_pred.strip()
        
        # Clean output using suffix (same as get_key_suffix + clean_outputstring)
        suffix = f"\n{tgt_name}:"
        split_idx = len(shots_eval_dict[lang_pair]) + 1 if shots_eval_dict else 1
        translation = clean_outputstring(decoded_pred, suffix, split_idx)
        translations.append(translation)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save translations (same format as run_llmmt.py)
    output_file = os.path.join(output_dir, f"test-{src_lang}-{tgt_lang}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for trans in translations:
            f.writelines([trans, "\n"])
    
    # Save references
    ref_file = os.path.join(output_dir, f"test-{lang_pair}.ref")
    with open(ref_file, 'w', encoding='utf-8') as f:
        refs = [item[lang_pair][tgt_lang] for item in dataset]
        f.write('\n'.join(refs))
    
    # Save sources (for COMET)
    src_file = os.path.join(output_dir, f"test-{lang_pair}.src")
    with open(src_file, 'w', encoding='utf-8') as f:
        srcs = [item[lang_pair][src_lang] for item in dataset]
        f.write('\n'.join(srcs))
    
    # Save raw decoded outputs (for debugging)
    debug_file = os.path.join(output_dir, f"test-{lang_pair}.debug.txt")
    with open(debug_file, 'w', encoding='utf-8') as f:
        for pred in decoded_preds:
            f.writelines([pred.replace('\n', '\\n'), "\n"])
    
    print(f"\n✓ Saved to {output_file}")
    print(f"\nSample (first 3):")
    for i in range(min(3, len(dataset))):
        print(f"\n--- Example {i+1} ---")
        pair_data = dataset[i][lang_pair]
        print(f"SRC: {pair_data[src_lang]}")
        print(f"HYP: {translations[i]}")
        print(f"REF: {pair_data[tgt_lang]}")
    
    return translations

print("\n" + "="*60)
print("EXPERIMENT: PRETRAINED BASE (ALMA-13B-Pretrain)")
print("="*60)

load_and_translate(
    base_model_path="haoranxu/ALMA-13B-Pretrain",
    adapter_path="haoranxu/X-ALMA-13B-Group1",
    output_dir="./outputs/pretrained",
    lang_pair="de-en",
    use_5shot=True,
    chat_style=True
)

load_and_translate(
    base_model_path="haoranxu/ALMA-13B-Pretrain",
    adapter_path="haoranxu/X-ALMA-13B-Group1",
    output_dir="./outputs/pretrained",
    lang_pair="en-de",
    use_5shot=True,
    chat_style=True
)