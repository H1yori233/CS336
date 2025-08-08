import os
from typing import Any, Generator
from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse.extract.html2text


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
