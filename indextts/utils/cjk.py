"""CJK text helpers with no audio or model imports."""

import re


def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    pattern = r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    chars = re.split(pattern, line.strip())
    return " ".join(
        word.strip().upper() if do_upper_case else word.strip()
        for word in chars
        if word.strip()
    )


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    english_pattern = re.compile(r"([A-Z]+(?:[\s'-][A-Z-]+)*)", re.IGNORECASE)
    english_sentences = english_pattern.findall(line)
    for index, sentence in enumerate(english_sentences):
        line = line.replace(sentence, f"<sent_{index}>")
    words = line.split()
    placeholder_pattern = re.compile(r"(<sent_(\d+)>)")
    for index in range(len(words)):
        for placeholder, value in placeholder_pattern.findall(words[index]):
            restored = english_sentences[int(value)]
            words[index] = words[index].replace(
                placeholder, restored.lower() if do_lower_case else restored
            )
    return "".join(words)
