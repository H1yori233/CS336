# use huggingface tokenizers to train the tokenizer as a reference

import json
import time
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- Parameters ---
VOCAB_SIZE = 32000
INPUT_FILE = "data/owt_train.txt"
# VOCAB_SIZE = 10000
# INPUT_FILE = "data/TinyStoriesV2-GPT4-train.txt"
SPECIAL_TOKENS = ["<|endoftext|>"]

# output file path
base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
output_dir = os.path.dirname(INPUT_FILE)
merges_output_path = os.path.join(output_dir, f"{base_name}-vocab_size_{VOCAB_SIZE}-merges.txt")
vocab_output_path = os.path.join(output_dir, f"{base_name}-vocab_size_{VOCAB_SIZE}-vocab.json")
tokenizer_json_path = "bpe_tokenizer.json" # temporary file

# --- Training tokenizer ---
start_time = time.time()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
tokenizer.train(files=[INPUT_FILE], trainer=trainer)
tokenizer.save(tokenizer_json_path)

end_time = time.time()

# print statistics
elapsed_hours = (end_time - start_time) / 3600
vocab = tokenizer.get_vocab()
longest_token = max(vocab.keys(), key=len) if vocab else ""

print("finished")
print(f"Training took: {elapsed_hours:.2f} hours")
print(f"Longest token: '{longest_token}'")

# --- save merges and vocab---
print(f"\nExtracting data from {tokenizer_json_path}...")
try:
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merges_list = data.get("model", {}).get("merges", [])
    if merges_list:
        with open(merges_output_path, "w", encoding="utf-8") as f:
            for pair in merges_list:
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"Merges successfully saved to: {merges_output_path}")
    else:
        print("No merges found in the tokenizer file.")

    vocab_dict = data.get("model", {}).get("vocab", {})
    if vocab_dict:
        with open(vocab_output_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary successfully saved to: {vocab_output_path}")
    else:
        print("No vocabulary found in the tokenizer file.")

except Exception as e:
    print(f"An error occurred while extracting data: {e}")
