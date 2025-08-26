import json
import torch
import datasets
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
from utils.CORD_util import JSONParseEvaluator, mask_values
from utils.FT_util import process_json


def run_finetune(quantize, epochs, train_batch_size, eval_batch_size, r, alpha, output_dir):
    # Quantize config selection
    bnb_config = None
    if quantize == "4-bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantize == "8-bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto", quantization_config=bnb_config,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    processor.tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Load dataset
    train = datasets.load_dataset("naver-clova-ix/cord-v2", split="train").cast_column(
        "image", datasets.Image(decode=False)
    )
    test = datasets.load_dataset("naver-clova-ix/cord-v2", split="test").cast_column(
        "image", datasets.Image(decode=False)
    )

    # Preprocess
    def preprocess(example):
        # Parse
        ann = json.loads(example["ground_truth"])["gt_parse"]
        masked = mask_values(ann)

        # Build prompt string
        template_str = json.dumps(masked, indent=2)
        prompt = (f"Parse the document by filling and returning *only* valid JSON matching this template "
                  f"exactly:\n{template_str}")

        # Tokenize prompt + image
        proc = processor(
            text=prompt,
            images=Image.open(BytesIO(example["image"]["bytes"])),
            return_tensors="pt",
        )
        prompt_ids = proc.input_ids[0]
        prompt_mask = proc.attention_mask[0]

        # Tokenize target JSON (no special BOS/EOS)
        target_str = json.dumps(ann, indent=2)
        target_ids = processor.tokenizer(target_str, add_special_tokens=False, return_tensors="pt").input_ids[0]

        # Concatenate
        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, torch.ones_like(target_ids)], dim=0)

        # Labels
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Map the dataset
    train_tokenized = train.map(preprocess, remove_columns=train.column_names, batched=False)
    test_tokenized = test.map(preprocess, remove_columns=test.column_names, batched=False)
    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Metrics
    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        pred_ids = np.where(
            pred_ids != -100,
            pred_ids,
            processor.tokenizer.pad_token_id,
        )
        label_ids = np.where(
            label_ids != -100,
            label_ids,
            processor.tokenizer.pad_token_id,
        )

        # Decode
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        evaluator = JSONParseEvaluator()

        scores = []
        for pred_str, ref_str in zip(preds, refs):
            second_json_start = process_json(pred_str)
            if second_json_start == -1:
                raise ValueError("Could not find the second JSON object in the prediction.")
            pred_str = pred_str[second_json_start:]
            try:
                pred_json = json.loads(pred_str)
                ref_json = json.loads(ref_str)
            except json.JSONDecodeError:
                scores.append(0.0)
                continue
            scores.append(evaluator.cal_acc(pred_json, ref_json))
        return {"ted_accuracy": float(sum(scores) / len(scores))}

    # Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_accumulation_steps=10,
        predict_with_generate=True,
        learning_rate=2e-4,
        fp16=True,
        remove_unused_columns=False,
        logging_steps=100,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with optional quantization")
    parser.add_argument("--quantize", default=None, choices=[None, "8-bit", "4-bit"],
                        help="Quantization level: '8-bit', '4-bit', or omit for no quantization.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Per-device evaluation batch size.")
    parser.add_argument("--r", type=int, default=8, help="r.")
    parser.add_argument("--alpha", type=int, default=16, help="alpha.")
    parser.add_argument("--output_dir", default="qwen_lora_ft", help="directory to save the trained model.")

    args = parser.parse_args()

    run_finetune(
        quantize=args.quantize,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        r=args.r,
        alpha=args.alpha,
        output_dir=args.output_dir
    )