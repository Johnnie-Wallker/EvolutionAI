def process_json(s: str) -> tuple[int, int]:
    """
    Finds the start and end index of the second JSON object in a string by balancing braces.
    Returns (start_index, end_index) of the second JSON.
    """
    # Find the start of the first JSON object
    first_brace = s.find('{')
    if first_brace == -1:
        return -1, -1

    open_braces = 0
    # Find the end of the first JSON
    for i in range(first_brace, len(s)):
        if s[i] == '{':
            open_braces += 1
        elif s[i] == '}':
            open_braces -= 1
        if open_braces == 0:
            second_brace = s.find('{', i + 1)
            if second_brace == -1:
                return -1, -1
            # --- New: find the end of the second JSON ---
            open_braces = 0
            for j in range(second_brace, len(s)):
                if s[j] == '{':
                    open_braces += 1
                elif s[j] == '}':
                    open_braces -= 1
                if open_braces == 0:
                    return second_brace, j + 1
            return second_brace, -1
    return -1, -1
