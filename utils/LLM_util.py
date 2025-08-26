import re
import os
import json
from openai import OpenAI


def llm_judge(result):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = f'Here is a JSON batch to evaluate:\n {result}'
    message = [
        {
            "role": "system",
            "content": "You are an evaluator that judges whether each candidate reply contains any of the "
                       "ground-truth answers (case-insensitive substring or minor punctuation variations allowed).  "
                       "Score each reply 1.0 if it contains at least one of the gt answers, otherwise 0.0.  "
                       "Finally, pick the reply with score 1.0 (if no reply got 1.0 then return blank for "
                       "best_reply and null for best_pid).  Respond ONLY with a JSON object exactly in the format "
                       "shown in the user prompt."
        },
        {"role": "user",
         "content": [
             {"type": "input_text", "text": prompt}
         ]}
    ]
    MAX_ATTEMPTS = 10
    KEYS = ("best_reply", "best_pid", "best_score")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.responses.create(model="gpt-4o", input=message)
        raw = response.output_text
        m = re.search(r"```json\s*(\{.*?})\s*```", raw, re.S)
        json_str = m.group(1) if m else raw

        try:
            json_result = json.loads(json_str)
        except json.JSONDecodeError:
            if attempt == MAX_ATTEMPTS:
                print("Process failed: never got valid JSON.")
            continue

        missing = [k for k in KEYS if k not in json_result]
        if missing:
            if attempt == MAX_ATTEMPTS:
                print(f"Process failed: missing keys {missing} in final attempt.")
            continue

        break

    for key in KEYS:
        result[key] = json_result[key]

    return result