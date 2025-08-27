import json
import torch
import numpy as np
from datasets import Dataset
from collections import Counter
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from utils.util import anls_score
from utils.FT_util import process_json


def run_finetune(quantize, epochs, train_batch_size, eval_batch_size, r, alpha, output_dir, train_size, eval_size):
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
        attn_implementation="flash_attention_2", local_files_only=True)
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

    # Create and load dataset
    def load_custom_docs(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)["documents"]
        records = []
        for data in docs:
            folder_name = data["hash"]
            fields = data['fields']
            images = data['images']
            for page in range(len(images)):
                image_name = images[page]["hash"]
                img_path = f"data/dataset_2/imgs/{folder_name}/{image_name}.png"
                page_fields = [field for field in fields if field['page'] == page]
                # Select only those fields that have at least 2 fields
                if len(page_fields) <= 1:
                    continue

                # Build JSON template
                name_counts = Counter(f["name"] for f in page_fields)
                repeated_names = {n for n, c in name_counts.items() if c > 1}
                gt_template = {}
                for field in page_fields:
                    name = field["name"]
                    value = field["value"]
                    if name in repeated_names:
                        grp = field.get("group", 0) + 1
                        key = f"{name} {grp}"
                    else:
                        key = name

                    gt_template[key] = value

                json_template = {k: "" for k in gt_template}
                template_str = json.dumps(json_template, indent=2)
                prompt = ("Parse the document by filling the template JSON and returning *only* valid JSON matching "
                          "this template exactly(If any field value contains internal double quotes, remove those "
                          "internal double quotes so they don't break the JSON.):\n") + template_str
                target_str = json.dumps(gt_template, indent=2)

                records.append({"image_path": img_path, "prompt": prompt, "target_str": target_str})
        return Dataset.from_list(records)

    if train_size == "full":
        train = load_custom_docs("data/dataset_2/train.json")
    else:
        train = load_custom_docs("data/dataset_2/train.json").select(range(int(train_size)))
    if eval_size == "full":
        test = load_custom_docs("data/dataset_2/test.json")
    else:
        test = load_custom_docs("data/dataset_2/test.json").select(range(int(eval_size)))

    def _build_inputs(prompt_text, image_path, add_gen_prompt):
        RESIZE_H, RESIZE_W = 800, 1200
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "resized_height": RESIZE_H, "resized_width": RESIZE_W},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen_prompt)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        return inputs

    # Preprocess
    def preprocess_train(ex):
        inputs = _build_inputs(ex["prompt"], ex["image_path"], add_gen_prompt=False)
        prompt_ids = inputs["input_ids"][0]
        prompt_mask = inputs["attention_mask"][0]
        tgt_ids = processor.tokenizer(ex["target_str"], add_special_tokens=False, return_tensors="pt").input_ids[0]

        input_ids = torch.cat([prompt_ids, tgt_ids], dim=0)
        attention_mask = torch.cat([prompt_mask, torch.ones_like(tgt_ids)], dim=0)
        labels = torch.cat([torch.full_like(prompt_ids, -100), tgt_ids], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
        }

    def preprocess_eval(ex):
        inputs = _build_inputs(ex["prompt"], ex["image_path"], add_gen_prompt=True)
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": torch.full_like(inputs["input_ids"][0], -100),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "target_str": ex["target_str"],
        }

    # Map the dataset
    train_tok = train.map(preprocess_train, remove_columns=train.column_names, batched=False, writer_batch_size=200)
    test_tok = test.map(preprocess_eval, remove_columns=test.column_names, batched=False, writer_batch_size=200)

    # Save refs then drop the column so collator stacks only tensors
    EVAL_REFS = test_tok["target_str"]
    test_tok = test_tok.remove_columns(["target_str"])

    # Make Datasets return tensors for ALL needed columns
    train_tok.set_format(type="torch",
                         columns=["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"])
    test_tok.set_format(type="torch",
                        columns=["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"])

    # Metrics
    def compute_metrics(eval_pred):
        pred_ids, _ = eval_pred
        pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

        doc_scores, all_pair_scores = [], []
        for pred_str, ref_str in zip(preds, EVAL_REFS):
            start, end = process_json(pred_str)
            if start == -1 or end == -1:
                doc_scores.append(0.0)
                continue

            pred_str = pred_str[start:end]
            try:
                pred_json = json.loads(pred_str)
                ref_json = json.loads(ref_str)
            except json.JSONDecodeError:
                doc_scores.append(0.0)
                continue

            pair_scores = []
            for name, gt in ref_json.items():
                pred_val = pred_json.get(name, "")
                pair_scores.append(anls_score(str(pred_val).lower(), str(gt).lower()))

            doc_scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
            all_pair_scores.extend(pair_scores)

        anls_micro = float(np.mean(all_pair_scores)) if all_pair_scores else 0.0
        anls_macro = float(np.mean(doc_scores)) if doc_scores else 0.0
        return {"anls_micro": anls_micro, "anls_macro": anls_macro}

    # Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        generation_max_length=4096,
        eval_accumulation_steps=10,
        predict_with_generate=True,
        learning_rate=2e-4,
        fp16=True,
        remove_unused_columns=False,
        logging_steps=500,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with optional quantization")
    parser.add_argument("--quantize", default=None, choices=[None, "8-bit", "4-bit"],
                        help="Quantization level: '8-bit', '4-bit', or omit for no quantization.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Per-device evaluation batch size.")
    parser.add_argument("--r", type=int, default=8, help="Parameter r in LoRA.")
    parser.add_argument("--alpha", type=int, default=16, help="Parameter alpha in LoRA.")
    parser.add_argument("--output_dir", default="qwen_lora_ft", help="Directory to save the trained model.")
    parser.add_argument("--train_size", default="full", help="Number of training samples.")
    parser.add_argument("--eval_size", default=50, help="Number of evaluation samples.")

    args = parser.parse_args()

    run_finetune(
        quantize=args.quantize,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        r=args.r,
        alpha=args.alpha,
        output_dir=args.output_dir,
        train_size=args.train_size,
        eval_size=args.eval_size
    )