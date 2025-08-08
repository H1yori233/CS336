import os
from typing import Any, Generator
from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse.extract.html2text
import fasttext
import regex as re
import nltk


def identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("data/lid.176.bin")
    label, confidence = model.predict(text.replace("\n", " "))
    return label[0].replace("__label__", ""), confidence[0]


def classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(
        "data/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin"
    )
    label, confidence = model.predict(text.replace("\n", " "))
    return label[0].replace("__label__", ""), confidence[0]


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(
        "data/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    )
    label, confidence = model.predict(text.replace("\n", " "))
    return label[0].replace("__label__", ""), confidence[0]


def classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def gopher_quality_filter(text: str) -> bool:
    nltk.data.find("tokenizers/punkt")
    words = nltk.word_tokenize(text)

    num_words = len(words)
    if num_words < 50 or num_words > 100000:
        return False

    total_length = sum(len(word) for word in words)
    avg_length = total_length / num_words
    if avg_length < 3 or avg_length > 10:
        return False

    lines = text.splitlines()
    num_lines = len(lines)
    if num_lines > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if (ellipsis_lines / num_lines) > 0.3:
            return False

    alphabetic_words = sum(1 for word in words if re.search(r"[a-zA-Z]", word))
    if (alphabetic_words / num_words) < 0.8:
        return False

    return True
