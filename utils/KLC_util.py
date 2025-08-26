import re
import json
import math


def get_confidence(tok_strs, tok_probs):
    # Build the exact output string and per-token probabilities
    full_text = "".join(tok_strs)

    # Isolate and parse the JSON
    m = re.search(r"\{.*}", full_text, flags=re.S)
    json_text = m.group(0)
    data = json.loads(json_text)

    # Char -> token index map
    char2tok = [-1] * len(full_text)
    cursor = 0
    offsets = []
    for i, s in enumerate(tok_strs):
        start, end = cursor, cursor + len(s)
        offsets.append((start, end))
        for c in range(start, end):
            char2tok[c] = i
        cursor = end

    def span_conf(start_char, end_char):
        idxs = sorted({char2tok[c] for c in range(start_char, end_char) if char2tok[c] >= 0})
        ps = [tok_probs[i] for i in idxs]
        if not ps:
            return None
        # geometric mean
        return math.exp(sum(math.log(p) for p in ps) / len(ps))

    # Compute confidence per field
    conf = {}
    search_pos = 0
    for k, v in data.items():
        if v == "":
            conf[k] = None
            continue

        pat = re.escape(f'"{k}"') + r'\s*:\s*"' + re.escape(v) + r'"'
        m = re.search(pat, full_text[search_pos:], flags=re.S)
        if not m:
            conf[k] = None
        continue

        start = search_pos + m.start()
        end = search_pos + m.end()
        conf[k] = span_conf(start, end)
        search_pos = end

    return conf