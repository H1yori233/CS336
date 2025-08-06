import os
from typing import Any, Generator
from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse.extract.html2text
import fasttext
import regex as re


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    # decode the byte string into a Unicode string
    html_str = None
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # https://resiliparse.chatnoir.eu/en/stable/man/parse/encoding.html
        enc = resiliparse.parse.encoding.detect_encoding(html_bytes)
        html_str = html_bytes.decode(enc, errors="replace")

    return resiliparse.extract.html2text.extract_plain_text(html_str)


def extract_text_from_warc(file_path: str | os.PathLike) -> Generator[str, None, None]:
    # https://resiliparse.chatnoir.eu/en/stable/man/fastwarc.html
    for record in ArchiveIterator(
        open(file_path, "rb"), record_types=WarcRecordType.response
    ):
        record_id = record.record_id
        html_bytes = record.reader.read()
        if not html_bytes:
            continue

        text = extract_text_from_html_bytes(html_bytes)
        if text and text.strip():
            yield record_id, text


def extract_text_from_wet(file_path: str | os.PathLike) -> Generator[str, None, None]:
    # https://resiliparse.chatnoir.eu/en/stable/man/fastwarc.html
    for record in ArchiveIterator(
        open(file_path, "rb"), record_types=WarcRecordType.conversion
    ):
        refers_to_id = record.headers.get("WARC-Refers-To")
        if not refers_to_id:
            continue

        wet_text = record.reader.read().decode("utf-8", errors="replace")
        if wet_text and wet_text.strip():
            yield refers_to_id, wet_text


def identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model("data/lid.176.bin")
    label, confidence = model.predict(text.replace("\n", " "))
    return label[0].replace("__label__", ""), confidence[0]


def mask_emails(text: str) -> tuple[str, int]:
    pattern = r"[A-Za-z0-9\.\-_]+@[A-Za-z0-9\-_]+\.[a-zA-Z]{2,}"
    # https://www.geeksforgeeks.org/python/re-subn-in-python/
    # search for a pattern and replace it with a string
    masked_text, num_masked = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return masked_text, num_masked


def mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r"(?<!\w)(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?[\d\s.-]{7,}(?!\w)"
    masked_text, num_masked = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return masked_text, num_masked


def mask_ips(text: str) -> tuple[str, int]:
    pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    masked_text, num_masked = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return masked_text, num_masked
