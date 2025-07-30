import os
import sys
import time
import random
import numpy as np
from cs336_basics.tokenizer import BPETokenizer

owt_train_path = "data/owt_train.txt"
owt_valid_path = "data/owt_valid.txt"
owt_vocab_path = "data/owt_train-vocab_size_32000-vocab.json"
owt_merges_path = "data/owt_train-vocab_size_32000-merges.txt"

tiny_train_path = "data/TinyStoriesV2-GPT4-train.txt"
tiny_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
tiny_vocab_path = "data/TinyStoriesV2-GPT4-train-vocab_size_10000-vocab.json"
tiny_merges_path = "data/TinyStoriesV2-GPT4-train-vocab_size_10000-merges.txt"

try:
    owt_tokenizer = BPETokenizer.from_files(owt_vocab_path, owt_merges_path)
    tiny_tokenizer = BPETokenizer.from_files(tiny_vocab_path, tiny_merges_path)
except Exception as e:
    print(f"Error loading tokenizers: {e}")
    sys.exit(1)

def sample_documents(file_path, num_docs=10, doc_separator="\n\n"):
    """Sample random documents from a text file without loading entire file into memory."""
    print(f"Sampling {num_docs} documents from {file_path}")
    
    # First pass: count total documents
    document_positions = []
    current_pos = 0
    buffer_size = 1024 * 1024  # 1MB buffer
    leftover = ""
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            
            # Combine with leftover from previous chunk
            text = leftover + chunk
            
            # Find document boundaries
            parts = text.split(doc_separator)
            
            # Process all complete documents (all but the last part)
            for i in range(len(parts) - 1):
                if parts[i].strip():  # Skip empty documents
                    document_positions.append(current_pos)
                current_pos += len(parts[i]) + len(doc_separator)
            
            # Keep the last part as leftover for next iteration
            leftover = parts[-1]
            current_pos += len(leftover)
    
    # Handle the final leftover if it's a valid document
    if leftover.strip():
        document_positions.append(current_pos - len(leftover))
    
    print(f"Found {len(document_positions)} documents")
    
    if len(document_positions) == 0:
        print("No documents found!")
        return []
    
    # Sample random document positions
    num_to_sample = min(num_docs, len(document_positions))
    sampled_positions = random.sample(document_positions, num_to_sample)
    
    # Read the sampled documents
    documents = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for pos in sorted(sampled_positions):
            f.seek(pos)
            # Read until we find the document separator or EOF
            doc_text = ""
            while True:
                char = f.read(1)
                if not char:  # EOF
                    break
                doc_text += char
                if doc_text.endswith(doc_separator):
                    doc_text = doc_text[:-len(doc_separator)]
                    break
            
            if doc_text.strip():
                documents.append(doc_text.strip())
    
    print(f"Successfully sampled {len(documents)} documents")
    return documents

# Sample 10 documents from TinyStories and OpenWebText
print("=" * 60)
print("SAMPLING DOCUMENTS")
print("=" * 60)

tiny_samples = sample_documents(tiny_train_path, 10)
# owt_samples = sample_documents(owt_train_path, 10)

print(f"TinyStories samples: {len(tiny_samples)} documents")
# print(f"OpenWebText samples: {len(owt_samples)} documents")

# Using trained tokenizer to encode these sampled documents into integer IDs
print("\n" + "=" * 60)
print("ENCODING SAMPLED DOCUMENTS")
print("=" * 60)

print("\nTinyStories samples with TinyStories tokenizer:")
for i, doc in enumerate(tiny_samples[:3]):  # Show first 3 for brevity
    tokens = tiny_tokenizer.encode(doc)
    print(f"Document {i+1} ({len(doc)} chars -> {len(tokens)} tokens):")
    print(f"  First 20 tokens: {tokens[:20]}")
    print(f"  Decoded: {tiny_tokenizer.decode(tokens[:50])}...")
    print()

# print("\nOpenWebText samples with OpenWebText tokenizer:")
# for i, doc in enumerate(owt_samples[:3]):  # Show first 3 for brevity
#     tokens = owt_tokenizer.encode(doc)
#     print(f"Document {i+1} ({len(doc)} chars -> {len(tokens)} tokens):")
#     print(f"  First 20 tokens: {tokens[:20]}")
#     print(f"  Decoded: {owt_tokenizer.decode(tokens[:50])}...")
#     print()

# # Tokenize OpenWebText sample with the TinyStories tokenizer
# print("\n" + "=" * 60)
# print("CROSS-TOKENIZATION ANALYSIS")
# print("=" * 60)

# print("\nOpenWebText samples with TinyStories tokenizer:")
# owt_with_tiny_tokens = []
# owt_with_owt_tokens = []

# for i, doc in enumerate(owt_samples[:5]):  # Analyze first 5 documents
#     tiny_tokens = tiny_tokenizer.encode(doc)
#     owt_tokens = owt_tokenizer.encode(doc)
    
#     owt_with_tiny_tokens.extend(tiny_tokens)
#     owt_with_owt_tokens.extend(owt_tokens)
    
#     print(f"Document {i+1}:")
#     print(f"  With TinyStories tokenizer: {len(tiny_tokens)} tokens")
#     print(f"  With OpenWebText tokenizer: {len(owt_tokens)} tokens")
#     print(f"  Compression ratio: {len(owt_tokens)/len(tiny_tokens):.3f}")
#     print()

# total_chars = sum(len(doc) for doc in owt_samples[:5])
# print(f"Total characters: {total_chars}")
# print(f"Total tokens (TinyStories tokenizer): {len(owt_with_tiny_tokens)}")
# print(f"Total tokens (OpenWebText tokenizer): {len(owt_with_owt_tokens)}")
# print(f"Overall compression ratio: {len(owt_with_owt_tokens)/len(owt_with_tiny_tokens):.3f}")

# # Estimate the throughput of your tokenizer
# print("\n" + "=" * 60)
# print("THROUGHPUT ESTIMATION")
# print("=" * 60)

# def estimate_throughput(tokenizer, sample_text, tokenizer_name):
#     """Estimate tokenization throughput."""
#     # Use a reasonably sized sample for timing
#     test_text = sample_text * 100  # Repeat to get a substantial test
    
#     # Warm up
#     tokenizer.encode(test_text[:1000])
    
#     # Time the encoding
#     start_time = time.time()
#     tokens = tokenizer.encode(test_text)
#     end_time = time.time()
    
#     elapsed_time = end_time - start_time
#     chars_processed = len(test_text)
#     tokens_generated = len(tokens)
    
#     chars_per_sec = chars_processed / elapsed_time
#     tokens_per_sec = tokens_generated / elapsed_time
#     mb_per_sec = chars_processed / (1024 * 1024) / elapsed_time
    
#     print(f"\n{tokenizer_name} Tokenizer Performance:")
#     print(f"  Processed {chars_processed:,} characters in {elapsed_time:.3f} seconds")
#     print(f"  Generated {tokens_generated:,} tokens")
#     print(f"  Throughput: {chars_per_sec:,.0f} chars/sec, {tokens_per_sec:,.0f} tokens/sec")
#     print(f"  Throughput: {mb_per_sec:.2f} MB/sec")
    
#     return chars_per_sec, tokens_per_sec, mb_per_sec

# # Test with both tokenizers
# tiny_sample_text = "\n".join(tiny_samples)
# owt_sample_text = "\n".join(owt_samples)

# tiny_throughput = estimate_throughput(tiny_tokenizer, tiny_sample_text, "TinyStories")
# owt_throughput = estimate_throughput(owt_tokenizer, owt_sample_text, "OpenWebText")

# # Estimate time for The Pile dataset (825GB)
# pile_size_gb = 825
# pile_size_bytes = pile_size_gb * 1024 * 1024 * 1024
# pile_size_chars = pile_size_bytes  # Assuming roughly 1 byte per character

# print(f"\n" + "=" * 60)
# print("THE PILE DATASET ESTIMATION")
# print("=" * 60)

# for name, (chars_per_sec, tokens_per_sec, mb_per_sec) in [
#     ("TinyStories", tiny_throughput),
#     ("OpenWebText", owt_throughput)
# ]:
#     time_seconds = pile_size_chars / chars_per_sec
#     time_hours = time_seconds / 3600
#     time_days = time_hours / 24
    
#     print(f"\nWith {name} tokenizer:")
#     print(f"  Estimated time to tokenize The Pile (825GB): {time_hours:.1f} hours ({time_days:.1f} days)")

# # Encode training and development datasets and serialize as NumPy arrays
# print("\n" + "=" * 60)
# print("ENCODING FULL DATASETS")
# print("=" * 60)

# def encode_and_save_dataset(file_path, tokenizer, output_path, dataset_name):
#     """Encode a dataset and save as uint16 numpy array."""
#     print(f"\nEncoding {dataset_name} from {file_path}")
    
#     start_time = time.time()
#     all_tokens = []
    
#     # Process file in chunks to handle large files
#     chunk_size = 1024 * 1024  # 1MB chunks
    
#     with open(file_path, 'r', encoding='utf-8') as f:
#         chunk_num = 0
#         while True:
#             chunk = f.read(chunk_size)
#             if not chunk:
#                 break
            
#             tokens = tokenizer.encode(chunk)
#             all_tokens.extend(tokens)
#             chunk_num += 1
            
#             if chunk_num % 100 == 0:
#                 print(f"  Processed {chunk_num} chunks, {len(all_tokens):,} tokens so far")
    
#     # Convert to numpy array with uint16 dtype
#     # Check if any token IDs exceed uint16 range
#     max_token_id = max(all_tokens) if all_tokens else 0
#     if max_token_id >= 65536:
#         print(f"  Warning: Max token ID {max_token_id} exceeds uint16 range, using uint32")
#         tokens_array = np.array(all_tokens, dtype=np.uint32)
#     else:
#         tokens_array = np.array(all_tokens, dtype=np.uint16)
    
#     # Save the array
#     np.save(output_path, tokens_array)
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     print(f"  Encoded {len(all_tokens):,} tokens in {elapsed_time:.2f} seconds")
#     print(f"  Saved to {output_path}")
#     print(f"  Array shape: {tokens_array.shape}, dtype: {tokens_array.dtype}")
    
#     return len(all_tokens)

# # Create output directory
# os.makedirs("data/encoded", exist_ok=True)

# # Encode TinyStories datasets
# tiny_train_tokens = encode_and_save_dataset(
#     tiny_train_path, 
#     tiny_tokenizer, 
#     "data/encoded/tiny_train_tokens.npy",
#     "TinyStories Train"
# )

# tiny_valid_tokens = encode_and_save_dataset(
#     tiny_valid_path, 
#     tiny_tokenizer, 
#     "data/encoded/tiny_valid_tokens.npy",
#     "TinyStories Valid"
# )

# # Encode OpenWebText datasets
# owt_train_tokens = encode_and_save_dataset(
#     owt_train_path, 
#     owt_tokenizer, 
#     "data/encoded/owt_train_tokens.npy",
#     "OpenWebText Train"
# )

# owt_valid_tokens = encode_and_save_dataset(
#     owt_valid_path, 
#     owt_tokenizer, 
#     "data/encoded/owt_valid_tokens.npy",
#     "OpenWebText Valid"
# )

# # Summary
# print("\n" + "=" * 60)
# print("SUMMARY")
# print("=" * 60)

# print(f"\nDataset encoding completed:")
# print(f"  TinyStories Train: {tiny_train_tokens:,} tokens")
# print(f"  TinyStories Valid: {tiny_valid_tokens:,} tokens")
# print(f"  OpenWebText Train: {owt_train_tokens:,} tokens")
# print(f"  OpenWebText Valid: {owt_valid_tokens:,} tokens")

# print(f"\nEncoded files saved to data/encoded/ directory")
# print(f"All arrays use uint16 dtype (or uint32 if token IDs exceed 65535)")

# # Verify the saved files
# print(f"\nVerifying saved files:")
# for filename in ["tiny_train_tokens.npy", "tiny_valid_tokens.npy", 
#                  "owt_train_tokens.npy", "owt_valid_tokens.npy"]:
#     filepath = f"data/encoded/{filename}"
#     if os.path.exists(filepath):
#         arr = np.load(filepath)
#         file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
#         print(f"  {filename}: {arr.shape} {arr.dtype}, {file_size_mb:.1f} MB")
#     else:
#         print(f"  {filename}: File not found!")

# print("\nScript completed successfully!")