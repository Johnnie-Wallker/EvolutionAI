import json
import os
import base64
import re
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from transformers import (logging, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoModel, AutoTokenizer,
                          AutoProcessor, GenerationConfig, BitsAndBytesConfig)
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from utils.Internvl_util import load_image
from utils.CORD_util import JSONParseEvaluator

logging.set_verbosity_error()


def run_model(model_name: str, quantize: str, FT_root: str) -> None:
    """
    Run a single vision-LLM model over the test split and save per-document JSON outputs + scores.

    Args:
        model_name: one of "Qwen2.5-VL-3B-Instruct", "InternVL3-2B-Instruct", "Phi-4-multimodal-instruct", "GPT-4o"
        quantize: None, "8-bit", or "4-bit" — if set, applies bits-and-bytes quantization.
        FT_root: root folder where the lora module is saved.
    """
    # Quantization config
    bnb_config = None
    if quantize == "8-bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "4-bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Instantiate model
    if model_name == "Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"Qwen/{model_name}", torch_dtype="auto", device_map="auto", quantization_config=bnb_config,
            attn_implementation="flash_attention_2", local_files_only=True
        )
        if FT_root is not None:
            model = PeftModel.from_pretrained(model, FT_root)
        processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}", local_files_only=True)
        model_type = 'qwen'

    elif model_name == "InternVL3-2B-Instruct":
        model = AutoModel.from_pretrained(f"OpenGVLab/{model_name}", torch_dtype=torch.bfloat16, device_map="auto",
                                          low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True,
                                          quantization_config=bnb_config, local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(f"OpenGVLab/{model_name}", trust_remote_code=True, use_fast=False,
                                                  local_files_only=True)
        model_type = "internvl"

    elif model_name == "Phi-4-multimodal-instruct":
        processor = AutoProcessor.from_pretrained(f"microsoft/{model_name}", trust_remote_code=True,
                                                  local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(f"microsoft/{model_name}", device_map="auto", torch_dtype="auto",
                                                     trust_remote_code=True, _attn_implementation="flash_attention_2",
                                                     local_files_only=True, quantization_config=bnb_config).eval()
        generation_config = GenerationConfig.from_pretrained(f"microsoft/{model_name}", local_files_only=True)
        model_type = "phi"

    elif model_name == "GPT-4o":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        model_type = "gpt"

    else:
        raise ValueError(f"Unrecognized model_name: {model_name!r}")

    # Prepare output directory
    if not quantize:
        if not FT_root:
            out_dir = os.path.join("results", "SROIE", "default", model_name)
        else:
            out_dir = os.path.join("results", f"SROIE_{FT_root}", "default", model_name)
    else:
        if not FT_root:
            out_dir = os.path.join("results", "SROIE", quantize, model_name)
        else:
            out_dir = os.path.join("results", f"SROIE_{FT_root}", quantize, model_name)

    os.makedirs(out_dir, exist_ok=True)

    doc_scores = []
    total_TP, total_FP, total_FN = 0, 0, 0

    annotations_dir = "data/SROIE/annotations"
    images_dir = "data/SROIE/images"

    json_files = [f for f in os.listdir(annotations_dir)]
    pbar = tqdm(json_files, desc=f"{model_name}")
    for file_name in pbar:
        # Load directory
        base_name = os.path.splitext(file_name)[0]
        img_path = os.path.join(images_dir, base_name + ".jpg")
        out_path = os.path.join(out_dir, f"{base_name}.json")

        # Load existing if present
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as inf:
                saved = json.load(inf)
            score = saved.get('score', 0.0)

        else:
            # Load dataset
            with open(os.path.join(annotations_dir, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)

            # Create json template
            masked = {k: "" for k in data}
            template_str = json.dumps(masked, indent=2)
            prompt = ("Parse the document by filling the template JSON and returning *only* valid JSON matching this "
                      "template exactly. The returned values should be in exactly the same format as appeared on the "
                      "document image(Capital, date format, number of decimals for total etc.):\n") + template_str

            # Generate JSON via chosen model type
            if model_type == "qwen":
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt}
                    ]
                }]

                # Preparation for inference
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt", ).to("cuda")

                # Run modal
                max_retries = 3
                for attempt in range(max_retries):
                    generated_ids = model.generate(**inputs, max_new_tokens=4096)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    raw = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                    json_str = m.group(1) if m else raw.strip()
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, list):
                            if attempt == max_retries - 1:
                                result = masked
                            else:
                                continue
                        break
                    except json.JSONDecodeError:
                        if attempt == max_retries - 1:
                            result = masked
            elif model_type == "internvl":
                pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=4096, do_sample=True)
                question = f"<image>\n{prompt}" + template_str

                # Run Modal
                max_retries = 3
                for attempt in range(max_retries):
                    raw = model.chat(tokenizer, pixel_values, question, generation_config)
                    m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                    json_str = m.group(1) if m else raw.strip()
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, list):
                            if attempt == max_retries - 1:
                                result = masked
                            else:
                                continue
                        break
                    except json.JSONDecodeError:
                        if attempt == max_retries - 1:
                            result = masked

            elif model_name == "GPT-4o":
                with open(img_path, "rb") as imgf:
                    b64 = base64.b64encode(imgf.read()).decode()
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        {"type": "input_text", "text": prompt}
                    ]
                }]

                # Run model
                max_retries = 3
                for attempt in range(max_retries):
                    response = client.responses.create(model="gpt-4o", input=message)
                    raw = response.output_text
                    # extract code‐blocked JSON if present
                    m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                    json_str = m.group(1) if m else raw
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, list):
                            if attempt == max_retries - 1:
                                result = masked
                            else:
                                continue
                        break
                    except json.JSONDecodeError:
                        if attempt == max_retries - 1:
                            result = masked

            else:
                prompt = f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
                image = Image.open(img_path).convert("RGB")
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                # Run model
                max_retries = 3
                for attempt in range(max_retries):
                    gen_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=4096,
                                             num_logits_to_keep=1)
                    # Strip off the prompt tokens:
                    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
                    raw = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                        0].strip()
                    # Extract code‐blocked JSON if present
                    m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                    json_str = m.group(1) if m else raw
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, list):
                            if attempt == max_retries - 1:
                                result = masked
                            else:
                                continue
                        break
                    except json.JSONDecodeError:
                        if attempt == max_retries - 1:
                            result = masked

            # Compute scores
            evaluator = JSONParseEvaluator()
            doc_score = evaluator.cal_acc(result, data)
            if doc_score == 1:
                score = 1
            else:
                score = 0

            # Save detailed per-doc result
            to_save = {
                "file_name": base_name,
                "gt_answer": data,
                "pred": result,
                "score": score,
                "TED_acc": doc_score
            }
            with open(out_path, "w", encoding="utf-8") as outf:
                json.dump(to_save, outf, indent=2, ensure_ascii=False)

        # Update running metrics
        if score == 1:
            total_TP += 1
        else:
            total_FP += 1
            total_FN += 1

        precision = total_TP / (total_TP + total_FP) if total_TP + total_FP != 0 else 0
        recall = total_TP / (total_TP + total_FN) if total_TP + total_FN != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        doc_scores.append(doc_score)
        mean_score = float(np.mean(doc_scores))

        pbar.set_postfix({
            "F1 score": f"{f1:.3f}",
            "mean TED acc": f"{mean_score:.3f}"
        })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run vision-LLM models over CORD.")
    parser.add_argument("--model_name", required=True,
                        choices=["Qwen2.5-VL-3B-Instruct", "InternVL3-2B-Instruct",
                                 "Phi-4-multimodal-instruct", "GPT-4o"],
                        help="Model name to run.")
    parser.add_argument("--quantize", default=None, choices=[None, "8-bit", "4-bit"],
                        help="Quantization level: '8-bit', '4-bit', or omit for no quantization.")
    parser.add_argument("--FT_root", default=None,
                        help="Directory where the trained LoRA module is stored.")

    args = parser.parse_args()
    run_model(args.model_name, args.quantize, args.FT_root)