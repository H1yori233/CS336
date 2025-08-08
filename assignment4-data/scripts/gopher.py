import os, sys, random
from tqdm import tqdm
from cs336_data.extract import extract_text_from_warc
from cs336_data.filter import gopher_quality_filter

WARC_FILE_PATH = "data/CC/example.warc.gz"

output_dir = "data/classifiers/"
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, "gopher_result.txt")
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
    is_harmful = gopher_quality_filter(warc_text)

    log(f"record_id: {record_id}")
    log(f"warc_text: {warc_text[:500]}")
    log(f"is_harmful: {is_harmful}")
    log("-" * 100)
