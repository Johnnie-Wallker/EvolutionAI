from Levenshtein import distance as lev_distance
from typing import List, Dict, Any, Set
import re


def anls_score(pred: str, gt: str, tau: float = 0.5) -> float:
    # normalized Levenshtein similarity, zero if below tau
    d = lev_distance(pred, gt)
    norm = max(len(pred), len(gt), 1)
    score = max(0.0, 1.0 - d / norm)
    return score if score >= tau else 0.0


def mask_dict_values(data, mask_token=""):
    """
    Recursively masks all values in a dictionary.
    """
    masked_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively call the function for nested dictionaries
            masked_data[key] = mask_dict_values(value, mask_token)
        else:
            # Mask the value
            masked_data[key] = mask_token
    return masked_data


def transform_funsd(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # --- Helpers ----------------------------------------------------------------
    def is_valid_text(text: str) -> bool:
        # must be non-empty and contain at least one alphanumeric character
        return bool(text and re.search(r'[A-Za-z0-9]', text))

    def gather_answers(id: int) -> str:
        ans_ids = sorted(aid for aid in linked[id] if aid in answer_ids)
        if not ans_ids:
            return ""
        if len(ans_ids) == 1:
            return id_map[ans_ids[0]]['text']

        # multi-answer logic (as before)
        texts = [(aid, id_map[aid]['text']) for aid in ans_ids]
        # only keep valid answers
        texts = [(aid, txt) for aid, txt in texts if is_valid_text(txt)]
        if not texts:
            return ""

        letter = [(aid, txt) for aid, txt in texts if txt[0].isalpha()]
        others = [(aid, txt) for aid, txt in texts if not txt[0].isalpha()]
        uppercase = [t for t in letter if t[1][0].isupper()]

        if len(uppercase) == 1:
            start = uppercase[0][0]
            letter_sorted = sorted(letter, key=lambda x: x[0])
            idx = next(i for i, (aid, _) in enumerate(letter_sorted) if aid == start)
            ordered = letter_sorted[idx:] + letter_sorted[:idx] + others
        else:
            ordered = sorted(letter, key=lambda x: x[0]) + sorted(others, key=lambda x: x[0])

        return " ".join(txt for _, txt in ordered)

    # --- Build indexes ---------------------------------------------------------
    id_map = {item['id']: item for item in dataset}
    header_ids = {i for i, item in id_map.items() if item['label'] == 'header'}
    question_ids = {i for i, item in id_map.items() if item['label'] == 'question'}
    answer_ids = {i for i, item in id_map.items() if item['label'] == 'answer'}

    linked: Dict[int, Set[int]] = {}
    for item in dataset:
        i = item['id']
        pairs = item.get('linking', [])
        linked[i] = {pid for pair in pairs for pid in pair} - {i}

    result: List[Dict[str, Any]] = []

    # --- 1) Top-level Q&A (no header) ------------------------------------------
    for qid in sorted(question_ids):
        qtxt: object = id_map[qid]['text']
        if not is_valid_text(qtxt):                     # skip blank or symbol-only
            continue
        if linked[qid] & header_ids:                    # skip those under headers
            continue
        ans = gather_answers(qid)
        if not is_valid_text(ans):                      # skip if no valid answer
            continue
        result.append({
            "question": qtxt,
            "answer": ans
        })

    # --- 2) Headers + subfields -----------------------------------------------
    for hid in sorted(header_ids):
        htxt = id_map[hid]['text']
        if not is_valid_text(htxt):                     # optional: skip symbol-only headers
            continue

        # find child questions, process them
        fields: List[Dict[str, str]] = []
        for qid in sorted(question_ids):
            if hid not in linked[qid]:
                continue
            qtxt = id_map[qid]['text']
            if not is_valid_text(qtxt):
                continue
            ans = gather_answers(qid)
            if not is_valid_text(ans):
                continue
            fields.append({
                "question": qtxt,
                "answer": ans
            })

        if fields:  # only keep headers with ≥1 valid QA
            result.append({
                "header": htxt,
                "fields": fields
            })

    return result


def mask_answers(data):
    masked = []
    for item in data:
        if 'question' in item and 'answer' in item:
            masked.append({'question': item['question'], 'answer': ''})
        elif 'header' in item and 'fields' in item:
            masked_fields = [
                {'question': f['question'], 'answer': ''}
                for f in item['fields']
            ]
            masked.append({'header': item['header'], 'fields': masked_fields})
    return masked


def flatten_qa_list(qa_list):
    """Turn your JSON structure into [(question, answer), ...]."""
    flat = []
    for item in qa_list:
        if 'question' in item and 'answer' in item:
            flat.append((item['question'], item['answer']))
        elif 'header' in item and 'fields' in item:
            for f in item['fields']:
                flat.append((f['question'], f['answer']))
    return flat