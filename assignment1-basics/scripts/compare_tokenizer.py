import os
import sys
import time
import random
import numpy as np
from cs336_basics.tokenizer import BPETokenizer

# --- Configuration ---
# Paths for OpenWebText data
owt_train_path = "data/owt_train.txt"
owt_valid_path = "data/owt_valid.txt"
owt_vocab_path = "data/owt_train-vocab_size_32000-vocab.json"
owt_merges_path = "data/owt_train-vocab_size_32000-merges.txt"

# Paths for TinyStories data
tiny_train_path = "data/TinyStoriesV2-GPT4-train.txt"
tiny_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
tiny_vocab_path = "data/TinyStoriesV2-GPT4-train-vocab_size_10000-vocab.json"
tiny_merges_path = "data/TinyStoriesV2-GPT4-train-vocab_size_10000-merges.txt"

# --- Output File Setup ---
# All print output will be redirected to this file.
output_dir = "data"
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

# --- Tokenizer Loading ---
try:
    log("Loading tokenizers...\n")
    owt_tokenizer = BPETokenizer.from_files(
        owt_vocab_path, owt_merges_path, special_tokens=["<|endoftext|>"]
    )
    tiny_tokenizer = BPETokenizer.from_files(
        tiny_vocab_path, tiny_merges_path, special_tokens=["<|endoftext|>"]
    )
    log("Tokenizers loaded successfully.\n")
except Exception as e:
    log(f"Error loading tokenizers: {e}\n")
    f_out.close()
    sys.exit(1)


def sample_documents(file_path, num_docs=10, doc_separator="\n\n"):
    """Sample documents randomly without loading the whole file."""
    log(
        f"Sampling {num_docs} documents from {os.path.basename(file_path)}...\n"
    )
    document_positions = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        while (
            len(document_positions) < num_docs * 5
        ):  # Sample more to ensure we get enough valid ones
            random_pos = random.randint(0, file_size)
            f.seek(random_pos)
            f.readline()  # Align to next line
            pos = f.tell()
            if pos < file_size:
                document_positions.append(pos)

    sampled_positions = random.sample(
        list(set(document_positions)), min(num_docs, len(document_positions))
    )

    documents = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for pos in sorted(sampled_positions):
            f.seek(pos)
            text = f.read(4096)  # Read a chunk
            doc = text.split(doc_separator)[0]
            if doc.strip():
                documents.append(doc.strip())

    log(f"Sampled {len(documents)} documents.\n")
    return documents


# --- Document Sampling ---
log("\n--- SAMPLING DOCUMENTS ---\n")
tiny_samples = sample_documents(tiny_train_path, 10, doc_separator="<|endoftext|>")
owt_samples = sample_documents(owt_train_path, 10, doc_separator="<|endoftext|>")

# --- Encoding Sampled Documents ---
log("\n--- ENCODING SAMPLES ---\n")

log("\nTinyStories samples w/ TinyStories tokenizer:\n")
for i, doc in enumerate(tiny_samples[:3]):
    tokens = tiny_tokenizer.encode(doc)
    log(f"  Doc {i+1}: {len(doc)} chars -> {len(tokens)} tokens\n")
    log(f"    Decoded: {tiny_tokenizer.decode(tokens[:50])}...\n")

log("\nOpenWebText samples w/ OpenWebText tokenizer:\n")
for i, doc in enumerate(owt_samples[:3]):
    tokens = owt_tokenizer.encode(doc)
    log(f"  Doc {i+1}: {len(doc)} chars -> {len(tokens)} tokens\n")
    log(f"    Decoded: {owt_tokenizer.decode(tokens[:50])}...\n")

# --- Cross-Tokenization Analysis ---
log("\n--- CROSS-TOKENIZATION ANALYSIS ---\n")
log("Tokenizing OpenWebText samples with both tokenizers:\n")
owt_with_tiny_tokens = []
owt_with_owt_tokens = []

for i, doc in enumerate(owt_samples[:5]):
    tiny_tokens = tiny_tokenizer.encode(doc)
    owt_tokens = owt_tokenizer.encode(doc)
    owt_with_tiny_tokens.extend(tiny_tokens)
    owt_with_owt_tokens.extend(owt_tokens)

    log(f"  Doc {i+1}:\n")
    log(f"    TinyTokenizer: {len(tiny_tokens)} tokens\n")
    log(f"    OWTTokenizer:  {len(owt_tokens)} tokens\n")
    if len(tiny_tokens) > 0:
        log(
            f"    Compression Ratio (OWT/Tiny): {len(owt_tokens)/len(tiny_tokens):.3f}\n"
        )

log("\nOverall Stats (first 5 OWT docs):\n")
log(f"  Total Chars: {sum(len(doc) for doc in owt_samples[:5])}\n")
log(f"  Total Tokens (TinyTokenizer): {len(owt_with_tiny_tokens)}\n")
log(f"  Total Tokens (OWTTokenizer):  {len(owt_with_owt_tokens)}\n")
if len(owt_with_tiny_tokens) > 0:
    log(
        f"  Overall Compression Ratio: {len(owt_with_owt_tokens)/len(owt_with_tiny_tokens):.3f}\n"
    )

# --- Throughput Estimation ---
log("\n--- THROUGHPUT ESTIMATION ---\n")


def estimate_throughput(tokenizer, sample_text, tokenizer_name):
    """Estimate tokenizer throughput."""
    test_text = sample_text * 10

    tokenizer.encode(test_text[:1000])  # Warm-up

    start_time = time.time()
    tokens = tokenizer.encode(test_text)
    end_time = time.time()

    elapsed_time = end_time - start_time
    if elapsed_time == 0:
        elapsed_time = 1e-9
    chars_processed = len(test_text)
    mb_per_sec = chars_processed / (1024 * 1024) / elapsed_time
    tokens_per_sec = len(tokens) / elapsed_time

    log(f"\n{tokenizer_name} Tokenizer Performance:\n")
    log(f"  {chars_processed:,} chars in {elapsed_time:.3f}s\n")
    log(
        f"  Throughput: {mb_per_sec:.2f} MB/s ({tokens_per_sec:,.0f} tokens/s)\n"
    )
    return mb_per_sec


tiny_sample_text = "".join(tiny_samples)
owt_sample_text = "".join(owt_samples)

tiny_mb_sec = estimate_throughput(tiny_tokenizer, tiny_sample_text, "TinyStories")
owt_mb_sec = estimate_throughput(owt_tokenizer, owt_sample_text, "OpenWebText")

# --- Pile Dataset Estimation ---
log(f"\n--- PILE DATASET ESTIMATION ---\n")
pile_size_gb = 825
pile_size_mb = pile_size_gb * 1024

for name, mb_per_sec in [("TinyStories", tiny_mb_sec), ("OpenWebText", owt_mb_sec)]:
    if mb_per_sec > 0:
        time_seconds = pile_size_mb / mb_per_sec
        time_hours = time_seconds / 3600
        time_days = time_hours / 24
        log(f"\nEstimating for {name} tokenizer:\n")
        log(
            f"  Est. time for The Pile (825GB): {time_hours:.1f} hours ({time_days:.1f} days)\n"
        )

# --- Encoding Full Datasets ---
log("\n--- ENCODING FULL DATASETS ---\n")


def encode_and_save_dataset(file_path, tokenizer, output_path, dataset_name):
    """Encode and save a dataset using streaming with encode_iterable."""
    log(f"\nEncoding {dataset_name} from {os.path.basename(file_path)}...\n")
    start_time = time.time()
    
    tokens = []
    line_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        # Use encode_iterable for memory-efficient processing
        for token_id in tokenizer.encode_iterable(f):
            tokens.append(token_id)
            
            # Log progress periodically
            if len(tokens) % 1000000 == 0:
                log(f"  Progress: {len(tokens)//10000}k tokens processed...")
    
    # Convert to numpy array
    max_token_id = max(tokens) if tokens else 0
    dtype = np.uint16 if max_token_id < 65536 else np.uint32
    tokens_array = np.array(tokens, dtype=dtype)

    np.save(output_path, tokens_array)
    end_time = time.time()

    log(f"  Finished: {len(tokens):,} tokens in {end_time - start_time:.2f}s.\n")
    log(f"  Saved to {output_path}.\n")
    log(f"  Array info: shape={tokens_array.shape}, dtype={tokens_array.dtype}.\n")
    return len(tokens)


os.makedirs("data/encoded", exist_ok=True)

# Only encode validation sets to avoid memory issues with large training sets
datasets_to_encode = [
    (
        tiny_valid_path,
        tiny_tokenizer,
        "data/encoded/tiny_valid_tokens.npy",
        "TinyStories Valid",
    ),
    (
        owt_valid_path,
        owt_tokenizer,
        "data/encoded/owt_valid_tokens.npy",
        "OpenWebText Valid",
    ),
]

encoded_tokens = {}
for path, tokenizer, out_path, name in datasets_to_encode:
    encoded_tokens[name] = encode_and_save_dataset(path, tokenizer, out_path, name)

# --- Summary and Verification ---
log("\n--- SUMMARY ---\n")
log("\nDataset Encoding Summary:\n")
for name, count in encoded_tokens.items():
    log(f"  {name}: {count:,} tokens\n")

log("\nFile Verification:\n")
for _, _, filename, _ in datasets_to_encode:
    filepath = filename
    if os.path.exists(filepath):
        arr = np.load(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        log(
            f"  {os.path.basename(filepath)}: shape={arr.shape}, dtype={arr.dtype}, size={file_size_mb:.1f}MB\n"
        )
    else:
        log(f"  {os.path.basename(filepath)}: NOT FOUND!\n")

log("\nScript finished.\n")
f_out.close()

print(f"Comparison results saved to {output_filepath}")
