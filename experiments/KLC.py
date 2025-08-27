import json
import os
import base64
import fitz
import re
import torch
import math
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from openai import OpenAI
from transformers import (logging, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM,
                          AutoModelForImageTextToText, AutoProcessor, GenerationConfig, BitsAndBytesConfig)
from qwen_vl_utils import process_vision_info
from utils.KLC_util import get_confidence
from utils.util import anls_score

logging.set_verbosity_error()


def run_model(model_name: str, quantize: str, FT_root: str, size: int) -> None:
    """
    Run a single vision-LLM model over the test split and save per-document JSON outputs + scores.

    Args:
        model_name: one of "Qwen2.5-VL-3B-Instruct", "InternVL3-2B-Instruct", "Phi-4-multimodal-instruct", "GPT-4o"
        quantize: None, "8-bit", or "4-bit" — if set, applies bits-and-bytes quantization.
        FT_root: root folder where the lora module is saved.
        size: None or an int, size of dataset to run.
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

    # Load dataset
    with open("data/kleister-charity/train/data.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Instantiate model
    if model_name == "Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"Qwen/{model_name}", torch_dtype="auto", device_map="auto", quantization_config=bnb_config,
            attn_implementation="flash_attention_2", local_files_only=True).eval()
        processor = AutoProcessor.from_pretrained(f"Qwen/{model_name}", local_files_only=True)
        model_type = 'qwen'

    elif model_name == "InternVL3-2B-Instruct":
        model_checkpoint = "OpenGVLab/InternVL3-2B-hf"
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map="auto",
                                                            torch_dtype=torch.bfloat16, local_files_only=True)
        model_type = "internvl"

    elif model_name == "Phi-4-multimodal-instruct":
        processor = AutoProcessor.from_pretrained(f"microsoft/{model_name}", trust_remote_code=True,
                                                  local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(f"microsoft/{model_name}", device_map="auto", torch_dtype="auto",
                                                     trust_remote_code=True, _attn_implementation="flash_attention_2",
                                                     local_files_only=True, quantization_config=bnb_config)
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
            out_dir = os.path.join("results", "KLC", "default", model_name)
        else:
            out_dir = os.path.join("results", f"KLC_{FT_root}", "default", model_name)
    else:
        if not FT_root:
            out_dir = os.path.join("results", "KLC", quantize, model_name)
        else:
            out_dir = os.path.join("results", f"KLC_{FT_root}", quantize, model_name)

    os.makedirs(out_dir, exist_ok=True)

    total_TP, total_FP, total_FN = 0, 0, 0
    total_TP_anls, total_FP_anls, total_FN_anls = 0, 0, 0

    if size is None:
        pbar = tqdm(dataset, desc=model_name, position=0, dynamic_ncols=True)
    else:
        pbar = tqdm(dataset[:size], desc=model_name, position=0, dynamic_ncols=True)
    for data in pbar:
        pdf_path = os.path.join("data", "kleister-charity", "documents", data["filename"])
        if not os.path.exists(pdf_path):
            continue

        hash_value, _ = os.path.splitext(data['filename'])
        out_path = os.path.join(out_dir, f"{hash_value}.json")

        # Load existing if present
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as inf:
                saved = json.load(inf)
            TP = saved.get('TP_count(confidence)', 0.0)
            FP = saved.get('FP_count(confidence)', 0.0)
            FN = saved.get('FN_count(confidence)', 0.0)
            TP_anls = saved.get('TP_count(anls)', 0.0)
            FP_anls = saved.get('FP_count(anls)', 0.0)
            FN_anls = saved.get('FN_count(anls)', 0.0)

        else:
            # Build JSON template
            fields = data["fields"]
            template = {k: "" for k in fields}
            json_template = json.dumps(template, indent=2)
            prompt = (f"Read the document and extract the following information of interest: (if applicable)\n"
                      f"address__post_town — post town of the address of the charitable organization (in upper-case "
                      f"letters)\naddress__postcode — postcode of the address of the charitable organization (in "
                      f"upper-case letters)\naddress__street_line — street line of the address of the charitable "
                      f"organization (in upper-case letters)\ncharity_name — the name of the charitable organization "
                      f"(in proper case)\ncharity_number — the registered number of the charitable organization\n"
                      f"income_annually_in_british_pounds — the annual income in British Pounds of the charitable "
                      f"organization\nreport_date — the reporting date of the annual document of the charitable "
                      f"organization (in YYYY-MM-DD format)\nspending_annually_in_british_pounds — the annual spending "
                      f"in British Pounds of the charitable organization\nNote that the information may or may not be "
                      f"on this page, if there is no such information, just return '' for that field(s). Return *only* "
                      f"valid JSON matching this template exactly, do not make changes to the JSON structure, your "
                      f"output JSON should be exactly the template but with values filled:\n{json_template}")

            agg = {f: [] for f in fields.keys()}
            doc = fitz.open(pdf_path)

            TP, FP, FN = 0, 0, 0
            TP_anls, FP_anls, FN_anls = 0, 0, 0
            total_pages = len(doc)
            for page_idx, page in enumerate(doc, start=1):
                pbar.set_description(f"{model_name} | Page {page_idx}/{total_pages}")
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
                img_bytes = pix.tobytes("png")
                img = Image.open(BytesIO(img_bytes))
                # Generate JSON via chosen model type
                if model_type == "qwen":
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
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
                        gen = model.generate(**inputs, max_new_tokens=1024, output_scores=True,
                                             return_dict_in_generate=True)
                        # Decode just the generated tokens
                        in_len = inputs["input_ids"].shape[1]
                        gen_ids = gen.sequences[:, in_len:]
                        raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

                        # extract code‐blocked JSON if present
                        m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                        json_str = m.group(1) if m else raw.strip()
                        try:
                            result = json.loads(json_str)
                            if not isinstance(result, dict):
                                if attempt == max_retries - 1:
                                    result = template
                                else:
                                    continue

                            # Decode token by token
                            tok_ids = gen_ids[0].tolist()
                            tok_strs = processor.tokenizer.convert_ids_to_tokens(tok_ids, skip_special_tokens=True)
                            trans = str.maketrans({"Ġ": " ", "Ċ": "\n"})
                            tok_strs = [t.translate(trans) for t in tok_strs]

                            # Token logprobs
                            transition_scores = model.compute_transition_scores(
                                gen.sequences, gen.scores, normalize_logits=True
                            )[0]
                            logprobs = transition_scores.detach().cpu().tolist()
                            tok_probs = [math.exp(logprob) for logprob in logprobs]
                            confidence = get_confidence(tok_strs, tok_probs)
                            break

                        except json.JSONDecodeError as e:
                            if attempt == max_retries - 1:
                                result = template

                if model_type == "internvl":
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    }]

                    # Preparation for inference
                    inputs = (processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                            return_dict=True, return_tensors="pt")
                              .to(model.device, dtype=torch.bfloat16))

                    # Run modal
                    max_retries = 3
                    for attempt in range(max_retries):
                        gen = model.generate(**inputs, max_new_tokens=1024, output_scores=True,
                                             return_dict_in_generate=True)

                        # Decode just the generated tokens
                        in_len = inputs["input_ids"].shape[1]
                        gen_ids = gen.sequences[:, in_len:]
                        raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                        # extract code‐blocked JSON if present
                        m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                        json_str = m.group(1) if m else raw.strip()
                        try:
                            result = json.loads(json_str)
                            if not isinstance(result, dict):
                                if attempt == max_retries - 1:
                                    result = json_template
                                else:
                                    continue
                            # Decode token by token
                            tok_ids = gen_ids[0].tolist()
                            tok_strs = processor.tokenizer.convert_ids_to_tokens(tok_ids, skip_special_tokens=True)
                            trans = str.maketrans({"Ġ": " ", "Ċ": "\n"})
                            tok_strs = [t.translate(trans) for t in tok_strs]

                            # Token logprobs
                            transition_scores = model.compute_transition_scores(
                                gen.sequences, gen.scores, normalize_logits=True
                            )[0]
                            logprobs = transition_scores.detach().cpu().tolist()
                            tok_probs = [math.exp(logprob) for logprob in logprobs]
                            confidence = get_confidence(tok_strs, tok_probs)
                            break
                        except json.JSONDecodeError as e:
                            if attempt == max_retries - 1:
                                result = template

                elif model_name == "GPT-4o":
                    b64_image = base64.b64encode(img_bytes).decode("utf-8")
                    message = [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                    # Run model
                    max_retries = 3
                    for attempt in range(max_retries):
                        response = client.chat.completions.create(model="gpt-4o", messages=message, logprobs=True,
                                                                  top_logprobs=0)
                        raw = response.choices[0].message.content

                        # extract code‐blocked JSON if present
                        m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                        json_str = m.group(1) if m else raw
                        try:
                            result = json.loads(json_str)
                            if not isinstance(result, dict):
                                if attempt == max_retries - 1:
                                    result = template
                                else:
                                    continue

                            tokens = response.choices[0].logprobs.content
                            tok_strs = [bytes(t.bytes).decode("utf-8") for t in tokens]
                            tok_probs = [math.exp(t.logprob) for t in tokens]
                            confidence = get_confidence(tok_strs, tok_probs)
                            break

                        except json.JSONDecodeError:
                            if attempt == max_retries - 1:
                                result = template

                else:
                    question = f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
                    inputs = processor(text=question, images=img, return_tensors="pt").to(model.device)
                    # Run model
                    max_retries = 3
                    for attempt in range(max_retries):
                        gen = model.generate(**inputs, generation_config=generation_config, max_new_tokens=1024,
                                             num_logits_to_keep=1, output_scores=True, return_dict_in_generate=True)
                        # Decode just the generated tokens
                        in_len = inputs["input_ids"].shape[1]
                        gen_ids = gen.sequences[:, in_len:]
                        raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

                        # Extract code‐blocked JSON if present
                        m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
                        json_str = m.group(1) if m else raw
                        try:
                            result = json.loads(json_str)
                            if not isinstance(result, dict):
                                if attempt == max_retries - 1:
                                    result = template
                                else:
                                    continue

                            # Decode token by token
                            tok_ids = gen_ids[0].tolist()
                            tok_strs = processor.tokenizer.convert_ids_to_tokens(tok_ids, skip_special_tokens=True)
                            trans = str.maketrans({"Ġ": " ", "Ċ": "\n"})
                            tok_strs = [t.translate(trans) for t in tok_strs]

                            # Token logprobs
                            transition_scores = model.compute_transition_scores(
                                gen.sequences, gen.scores, normalize_logits=True)[0]
                            logprobs = transition_scores.detach().cpu().tolist()
                            tok_probs = [math.exp(logprob) for logprob in logprobs]
                            confidence = get_confidence(tok_strs, tok_probs)
                            break

                        except json.JSONDecodeError:
                            if attempt == max_retries - 1:
                                result = template

                # Aggregate answer
                for f in fields.keys():
                    agg[f].append({"page": page_idx, "value": result.get(f), "conf": confidence.get(f)})

            doc.close()

            # Get the best response by confidence
            best = {}
            for k, rows in agg.items():
                cand = [r for r in rows if r["value"]]
                best[k] = None if not cand else max(
                    cand, key=lambda r: r["conf"] if r["conf"] is not None else -math.inf
                )["value"]

            # Get the best response by ANLS
            best_by_anls = {}
            for k, rows in agg.items():
                gt_txt = fields[k]
                best_score = -1.0
                best_val = None
                for r in rows:
                    v = (r.get("value") or "").strip()
                    if not v:
                        continue
                    s = anls_score(v, gt_txt)
                    if s > best_score:
                        best_score = s
                        best_val = v
                best_by_anls[k] = best_val

            # Normalise keys
            underscore_keys = ("address__post_town", "address__postcode", "address__street_line", "charity_name")
            money_keys = ("income_annually_in_british_pounds", "spending_annually_in_british_pounds")

            # Normalise best by confidence
            normalized = {}
            for k, v in best.items():
                if not v or (isinstance(v, str) and v.strip().lower() == "none"):
                    normalized[k] = v
                    continue
                if k in underscore_keys:
                    normalized[k] = re.sub(r"\s+", "_", v.strip())
                elif k in money_keys:
                    cleaned = re.sub(r"[^\d.]", "", v)
                    if not cleaned:
                        normalized[k] = v
                    else:
                        num = float(cleaned)
                        normalized[k] = f"{num:.2f}"
                else:
                    normalized[k] = v

            # Normalise best by ANLS
            normalized_anls = {}
            for k, v in best_by_anls.items():
                if not v or (isinstance(v, str) and v.strip().lower() == "none"):
                    normalized_anls[k] = v
                    continue
                if k in underscore_keys:
                    normalized_anls[k] = re.sub(r"\s+", "_", v.strip())
                elif k in money_keys:
                    cleaned = re.sub(r"[^\d.]", "", v)
                    if not cleaned:
                        normalized_anls[k] = v
                    else:
                        num = float(cleaned)
                        normalized_anls[k] = f"{num:.2f}"
                else:
                    normalized_anls[k] = v

            # Compute scores
            for field in fields:
                gt = fields[field]
                pred = normalized[field]
                pred_anls = normalized_anls[field]
                if gt == pred:
                    TP += 1
                else:
                    FP += 1
                    FN += 1
                if gt == pred_anls:
                    TP_anls += 1
                else:
                    FP_anls += 1
                    FN_anls += 1

            # Save to cache
            cache_data = {
                "filename": hash_value,
                "aggregated_answer": agg,
                "gt_answer": fields,
                "best(confidence)": normalized,
                "best(anls)": normalized_anls,
                "TP_count(confidence)": TP,
                "FP_count(confidence)": FP,
                "FN_count(confidence)": FN,
                "TP_count(anls)": TP_anls,
                "FP_count(anls)": FP_anls,
                "FN_count(anls)": FN_anls

            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

        # Update running metrics
        total_TP += TP
        total_FP += FP
        total_FN += FN
        precision = total_TP / (total_TP + total_FP) if total_TP + total_FP != 0 else 0
        recall = total_TP / (total_TP + total_FN) if total_TP + total_FN != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        total_TP_anls += TP_anls
        total_FP_anls += FP_anls
        total_FN_anls += FN_anls
        precision_anls = total_TP_anls / (total_TP_anls + total_FP_anls) if total_TP_anls + total_FP_anls != 0 else 0
        recall_anls = total_TP_anls / (total_TP_anls + total_FN_anls) if total_TP_anls + total_FN_anls != 0 else 0
        f1_anls = 2 * precision_anls * recall_anls / (
                precision_anls + recall_anls) if precision_anls + recall_anls != 0 else 0

        pbar.set_postfix({
            "F1 Score(confidence)": f"{f1:.3f}",
            "F1 Score(anls)": f"{f1_anls:.3f}"
        })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run vision-LLM models over dataset1.")
    parser.add_argument("--model_name", required=True,
                        choices=["Qwen2.5-VL-3B-Instruct", "InternVL3-2B-Instruct",
                                 "Phi-4-multimodal-instruct", "GPT-4o"],
                        help="Model name to run.")
    parser.add_argument("--quantize", default=None, choices=[None, "8-bit", "4-bit"],
                        help="Quantization level: '8-bit', '4-bit', or omit for no quantization.")
    parser.add_argument("--FT_root", default=None,
                        help="Directory where the trained LoRA module is stored.")
    parser.add_argument("--size", type=int, default=100, help="Data size to run, None for full data.")

    args = parser.parse_args()
    run_model(args.model_name, args.quantize, args.FT_root, args.size)
