import os, sys
from cs336_data.filter import extract_text_from_warc, extract_text_from_wet

WARC_FILE_PATH = "data/CC/example.warc.gz"
WET_FILE_PATH = "data/CC/example.warc.wet.gz"

output_dir = "data/CC/"
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, "compare_result.txt")
try:
    f_out = open(output_filepath, "w", encoding="utf-8")
except IOError as e:
    print(f"Fatal: Could not open output file {output_filepath}. Error: {e}")
    sys.exit(1)


def log(message=""):
    print(message)
    f_out.write(message + "\n")


warc_text_generator = extract_text_from_warc(WARC_FILE_PATH)
wet_text_generator = extract_text_from_wet(WET_FILE_PATH)

for i, (warc_record, wet_record) in enumerate(
    zip(warc_text_generator, wet_text_generator)
):
    if i > 10:
        break

    warc_id, warc_text = warc_record
    refers_to_id, wet_text = wet_record

    if warc_id != refers_to_id:
        log(f"warc_id: {warc_id} != refers_to_id: {refers_to_id}")
        continue

    log("warc_text:")
    log(warc_text)
    log("wet_text:")
    log(wet_text)
    log("-" * 100)

    i += 1
