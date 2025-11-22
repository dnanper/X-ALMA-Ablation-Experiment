#!/usr/bin/env python
# coding=utf-8

"""
Example script to test load_model_specific() function for ablation studies.

This script demonstrates how to load X-ALMA with different base models to measure
the contribution of pretrain stage.

Usage examples:

1. Test with unpretrained base (to measure pretrain contribution):
   python example_load_model_specific.py \
       --base_model_path meta-llama/Llama-2-13b-hf \
       --adapter_path haoranxu/X-ALMA-13B-Group5

2. Test with pretrained base (normal X-ALMA):
   python example_load_model_specific.py \
       --base_model_path haoranxu/ALMA-13B-Pretrain \
       --adapter_path haoranxu/X-ALMA-13B-Group5

3. Test with different base model size:
   python example_load_model_specific.py \
       --base_model_path meta-llama/Llama-2-7b-hf \
       --adapter_path haoranxu/X-ALMA-13B-Group5
"""

import logging
import sys
import torch
from transformers import AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

# Add parent directory to path to import utils
sys.path.append('.')
from utils.utils import load_model_specific, LANG_TABLE
from utils.arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class CustomArguments:
    """
    Arguments for custom model loading.
    """
    base_model_path: str = field(
        metadata={
            "help": "HuggingFace path to base model. Examples: "
                    "'meta-llama/Llama-2-13b-hf' (unpretrained), "
                    "'haoranxu/ALMA-13B-Pretrain' (pretrained)"
        }
    )
    adapter_path: str = field(
        metadata={
            "help": "HuggingFace path to adapter/module. Examples: "
                    "'haoranxu/X-ALMA-13B-Group5'"
        }
    )
    test_translation: bool = field(
        default=True,
        metadata={"help": "Whether to run a test translation"}
    )
    source_lang: str = field(
        default="cs",
        metadata={"help": "Source language code for test translation"}
    )
    target_lang: str = field(
        default="en",
        metadata={"help": "Target language code for test translation"}
    )
    test_sentence: str = field(
        default="Dobrý den, jak se máte?",
        metadata={"help": "Test sentence for translation"}
    )


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Parse arguments
    parser = HfArgumentParser((CustomArguments, ModelArguments, DataTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        custom_args, model_args, data_args = parser.parse_json_file(
            json_file=sys.argv[1]
        )
    else:
        custom_args, model_args, data_args = parser.parse_args_into_dataclasses()

    # Load tokenizer from adapter path (it should have the correct tokenizer)
    logger.info(f"Loading tokenizer from {custom_args.adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        custom_args.adapter_path,
        padding_side='left',
        use_fast=model_args.use_fast_tokenizer,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with specific base and adapter
    logger.info("=" * 80)
    logger.info("Loading model with load_model_specific()")
    logger.info("=" * 80)
    
    model = load_model_specific(
        base_model_path=custom_args.base_model_path,
        adapter_model_path=custom_args.adapter_path,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        logger=logger,
    )
    
    logger.info("=" * 80)
    logger.info("Model loaded successfully!")
    logger.info("=" * 80)

    # Run test translation if requested
    if custom_args.test_translation:
        logger.info("\n" + "=" * 80)
        logger.info("Running test translation")
        logger.info("=" * 80)
        
        src_lang = custom_args.source_lang
        tgt_lang = custom_args.target_lang
        test_sent = custom_args.test_sentence
        
        # Create prompt
        src_fullname = LANG_TABLE.get(src_lang, src_lang)
        tgt_fullname = LANG_TABLE.get(tgt_lang, tgt_lang)
        
        prompt = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: {test_sent}\n{tgt_fullname}:"
        
        # For X-ALMA, use chat template
        if model_args.chat_style:
            chat_style_prompt = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                chat_style_prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        logger.info(f"\nPrompt:\n{prompt}")
        
        # Tokenize
        input_ids = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            max_length=512, 
            truncation=True
        ).input_ids
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            model = model.cuda()
        
        # Generate
        logger.info("\nGenerating translation...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                num_beams=5,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        
        # Decode
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        logger.info(f"\n{'=' * 80}")
        logger.info("Translation Result:")
        logger.info(f"{'=' * 80}")
        logger.info(f"Source ({src_lang}): {test_sent}")
        logger.info(f"Target ({tgt_lang}): {outputs[0]}")
        logger.info(f"{'=' * 80}\n")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
