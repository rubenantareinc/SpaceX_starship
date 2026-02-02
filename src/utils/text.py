import re
from typing import List


SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = SENTENCE_SPLIT_REGEX.split(text)
    return [s.strip() for s in sentences if s.strip()]
