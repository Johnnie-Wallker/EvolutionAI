import json
import os
import base64
import torch
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
from transformers import (logging, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoModel, AutoTokenizer,
                          AutoProcessor, GenerationConfig, BitsAndBytesConfig)
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from utils.Internvl_util import load_image
from utils.LLM_util import llm_judge

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
    DATA_JSON = "data/DocVQA/qas/val.json"
    IMG_DIR = "data/DocVQA/images"
    with open(DATA_JSON, encoding="utf-8") as f:
        dataset = json.load(f)["data"]

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
            out_dir = os.path.join("results", "DocVQA", "default", model_name)
        else:
            out_dir = os.path.join("results", f"DocVQA_{FT_root}", "default", model_name)
    else:
        if not FT_root:
            out_dir = os.path.join("results", "DocVQA", quantize, model_name)
        else:
            out_dir = os.path.join("results", f"DocVQA_{FT_root}", quantize, model_name)

    os.makedirs(out_dir, exist_ok=True)

    total, correct = 0, 0

    if size is None:
        pbar = tqdm(dataset, desc=model_name, position=0, dynamic_ncols=True)
    else:
        pbar = tqdm(dataset[:size], desc=model_name, position=0, dynamic_ncols=True)
    for qa in pbar:
        qid = qa["questionId"]
        out_path = os.path.join(out_dir, f"{qid}.json")

        # Load existing if present
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as inf:
                saved = json.load(inf)
            total += 1
            best_score = saved.get('best_score') or 0.0
            best_score = float(best_score)

        else:
            # Build prompt
            question = qa["question"]
            gt_answers = qa["answers"]
            page_ids = qa["page_ids"]
            replies = {}

            for pid in page_ids:
                img_path = f"{IMG_DIR}/{pid}.jpg"
                # Generate JSON via chosen model type
                if model_type == "qwen":
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": question}
                        ]
                    }]

                    # Preparation for inference
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt",).to("cuda")

                    # Run modal
                    generated_ids = model.generate(**inputs, max_new_tokens=4096)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    pred = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                elif model_type == "internvl":
                    pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    generation_config = dict(max_new_tokens=4096, do_sample=True)
                    question = f"<image>\n{question}"
                    pred = model.chat(tokenizer, pixel_values, question, generation_config)

                elif model_name == "GPT-4o":
                    with open(img_path, "rb") as imgf:
                        b64 = base64.b64encode(imgf.read()).decode()
                    message = [{
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                            {"type": "input_text", "text": question}
                        ]
                    }]

                    # Run model
                    response = client.responses.create(model="gpt-4o", input=message)
                    pred = response.output_text

                else:
                    prompt = f"<|user|><|image_1|>{question}<|end|><|assistant|>"
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                    # Run model
                    gen_ids = model.generate(**inputs, generation_config=generation_config, max_new_tokens=4096,
                                             num_logits_to_keep=1)
                    # Strip off the prompt tokens:
                    gen_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
                    pred = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

                replies[pid] = pred

            # Save result
            result = {
                "questionId": qid,
                "question": question,
                "gt_answers": gt_answers,
                "replies": replies,
                "best_reply": "",
                "best_pid": "",
                "best_score": ""
            }
            result = llm_judge(result)
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(result, fp, ensure_ascii=False, indent=2)

            total += 1
            best_score = result.get('best_score') or 0.0
            best_score = float(best_score)

        # Update running metrics
        correct += best_score
        pbar.set_postfix(acc=f"{correct / total:.3f}")


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
    parser.add_argument("--size", type=int, default=None, help="Data size to run.")

    args = parser.parse_args()
    run_model(args.model_name, args.quantize, args.FT_root, args.size)