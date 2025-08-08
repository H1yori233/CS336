import regex as re


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
