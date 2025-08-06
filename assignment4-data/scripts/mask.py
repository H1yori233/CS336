import os, sys, random
from tqdm import tqdm
from cs336_data.filter import extract_text_from_warc, mask_emails, mask_ips, mask_phone_numbers

WARC_FILE_PATH = "data/CC/example.warc.gz"

output_dir = "data/CC/"
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, "mask_result.txt")
try:
    f_out = open(output_filepath, "w", encoding="utf-8")
except IOError as e:
    print(f"Fatal: Could not open output file {output_filepath}. Error: {e}")
    sys.exit(1)


def log(message=""):
    print(message)
    f_out.write(message + "\n")


SAMPLE_SIZE = 20
reservoir = []
generator = extract_text_from_warc(WARC_FILE_PATH)

# Reservoir sampling
for i, (record_id, warc_text) in enumerate(generator):
    if i < SAMPLE_SIZE:
        reservoir.append((record_id, warc_text))
    else:
        j = random.randint(0, i)
        if j < SAMPLE_SIZE:
            reservoir[j] = (record_id, warc_text)

log(f"Sampled {len(reservoir)} records using reservoir sampling.")
log("=" * 100)

# Process sampled texts
for record_id, warc_text in tqdm(reservoir, desc="Masking"):
    text, num_masked = mask_emails(warc_text)
    text, num_masked = mask_phone_numbers(text)
    text, num_masked = mask_ips(text)

    log(f"record_id: {record_id}")
    log(f"warc_text: {warc_text[:200]}")
    log(f"masked_text: {text[:500]}")
    log("-" * 100)
