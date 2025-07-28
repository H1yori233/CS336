import os
from typing import IO, Any, BinaryIO, Iterable, Iterator
import collections
import regex as re
from multiprocessing import Pool
import heapq

# --- Custom Utilities ---

class _Node:
    def __init__(self, token_id: int, count_ref: dict[str, int]):
        self.id = token_id 
        self.count_ref = count_ref # point to a shared counter dict {'count': ...}
        self.prev: '_Node | None' = None
        self.next: '_Node | None' = None
        
    @property
    def count(self) -> int:
        return self.count_ref['count']

class _PriorityQueueItem:
    """
    Represents an item in the priority queue used during BPE merging.
    """
    def __init__(self, count: int, p1_bytes: bytes, p2_bytes: bytes, pair: tuple[int, int]):
        self.count = count
        self.p1_bytes = p1_bytes
        self.p2_bytes = p2_bytes
        self.pair = pair

    def __lt__(self, other: '_PriorityQueueItem') -> bool:
        # 1. Higher count has higher priority.
        if self.count != other.count:
            return self.count > other.count
        # 2. If counts are equal, compare p1_bytes lexicographically (larger is higher priority).
        if self.p1_bytes != other.p1_bytes:
            return self.p1_bytes > other.p1_bytes
        # 3. If still equal, compare p2_bytes lexicographically.
        return self.p2_bytes > other.p2_bytes


    
# --- BPE Trainer & Tokenizer ---

class BPETrainer:
    """A BPE Trainer to learn vocabulary and merges from a corpus."""
    
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
        self.special_tokens = special_tokens
        self.next_id = 0
        self.vocab_size = vocab_size
        
        # Store special token IDs to prevent them from being merged
        self.special_token_ids = set()
        
        # one-to-one mapping from special_tokens, bytestring token to ID
        for st in special_tokens:
            self.vocab[self.next_id] = st.encode("utf-8")
            self.special_token_ids.add(self.next_id)  # Mark as special token
            self.next_id += 1
        for i in range(256):
            self.vocab[self.next_id] = bytes([i])
            self.next_id += 1
            
        self.byte_string_to_id = {v: k for k, v in self.vocab.items()}
        self.pretoken_counts: dict[tuple[bytes, ...], int] = collections.defaultdict(int) # { token sequence : count }
        
        # pattern for pre-tokenization
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # This token is used for finding optimal chunk boundaries for multiprocessing.
        self.split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b'\n'
        
        # pattern for splitting special tokens
        if special_tokens:
            self.split_special_pattern = f"({ '|'.join([re.escape(st) for st in special_tokens]) })"
        else:
            self.split_special_pattern = None

    # --- Main Functions ---
    
    def pretokenize(self, input_path: str | os.PathLike):
        with open(input_path, "rb") as f:
            num_processes = 8
            boundaries = self.find_chunk_boundaries(f, num_processes, self.split_special_token)
            
            # Prepare arguments for multiprocessing
            chunk_args = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunk_args.append((start, end, input_path, self.PAT, 
                                   self.special_tokens, self.byte_string_to_id, 
                                   self.split_special_pattern))
            
            with Pool(processes=num_processes) as pool:
                chunk_results = pool.map(self.process_chunk, chunk_args)
            
            # Merge results from all chunks
            for chunk_pretoken_counts in chunk_results:
                for token_ids, count in chunk_pretoken_counts.items():
                    self.pretoken_counts[token_ids] += count
    
    def merge(self):
        '''
        More efficient implementation of merge.
        Only iterate pretoken_counts once to build the linked list of nodes,
        build a linked list of nodes, as well as the frequency (pair_counts), and index (pair_to_nodes).
        '''
        pair_counts: collections.defaultdict[tuple[int, int], int] = collections.defaultdict(int)
        pair_to_nodes: collections.defaultdict[tuple[int, int], set[_Node]] = collections.defaultdict(set)
        
        for token_id_list, count in self.pretoken_counts.items():
            if len(token_id_list) < 2:
                continue
            count_ref = {'count': count}
            head = _Node(token_id_list[0], count_ref)
            prev_node = head
            for i in range(1, len(token_id_list)):
                current_node = _Node(token_id_list[i], count_ref)
                prev_node.next = current_node
                current_node.prev = prev_node
                pair = (prev_node.id, current_node.id)
                
                # Skip pairs involving special tokens
                if prev_node.id not in self.special_token_ids and current_node.id not in self.special_token_ids:
                    pair_counts[pair] += count
                    pair_to_nodes[pair].add(prev_node)
                    
                prev_node = current_node

        pq = []
        for pair, count in pair_counts.items():
            p1_bytes, p2_bytes = self.vocab[pair[0]], self.vocab[pair[1]]
            item = _PriorityQueueItem(count, p1_bytes, p2_bytes, pair) # custom comparator
            heapq.heappush(pq, item)

        def update_stats(pair_to_update, delta, node_to_index):
            if not pair_to_update: return
            # Skip pairs involving special tokens
            if pair_to_update[0] in self.special_token_ids or pair_to_update[1] in self.special_token_ids:
                return
                
            if delta > 0:
                pair_to_nodes[pair_to_update].add(node_to_index)
            else:
                pair_to_nodes[pair_to_update].discard(node_to_index)
            pair_counts[pair_to_update] += delta
            
            # push into priority queue
            if pair_counts[pair_to_update] > 0:
                p1_b, p2_b = self.vocab[pair_to_update[0]], self.vocab[pair_to_update[1]]
                new_item = _PriorityQueueItem(pair_counts[pair_to_update], p1_b, p2_b, pair_to_update)
                heapq.heappush(pq, new_item)

        # merges
        num_merges_to_do = self.vocab_size - len(self.vocab)
        for _ in range(num_merges_to_do):
            best_pair = None
            while pq:
                item = heapq.heappop(pq)
                # lazy delete: check if frequency still matches
                if pair_counts.get(item.pair, 0) == item.count:
                    best_pair = item.pair
                    break
            
            if best_pair is None:
                break
            id1, id2 = best_pair
            
            new_id = self.allocate_id()
            token1_bytes, token2_bytes = self.vocab[id1], self.vocab[id2]
            new_token_bytes = token1_bytes + token2_bytes
            self.vocab[new_id] = new_token_bytes
            self.byte_string_to_id[new_token_bytes] = new_id
            self.merges.append((token1_bytes, token2_bytes))
            
            # update the linked list
            valid_pairs = []
            for node1 in pair_to_nodes[best_pair]:
                node2 = node1.next
                # pre-determine the valid pairs, avoid node conflict
                if node2 is not None and node1.id == id1 and node2.id == id2:
                    valid_pairs.append((node1, node2))
            
            for node1, node2 in valid_pairs:
                word_count = node1.count
                if node1.prev:
                    left_node = node1.prev
                    update_stats((left_node.id, node1.id), -word_count, left_node) # update left node
                    update_stats((left_node.id, new_id), word_count, left_node)    # update new node
                if node2.next:
                    right_node = node2.next
                    update_stats((node2.id, right_node.id), -word_count, node2) # update right node
                    update_stats((new_id, right_node.id), word_count, node1)    # update new node
                
                node1.id = new_id
                node1.next = node2.next
                if node2.next:
                    node2.next.prev = node1
            del pair_counts[best_pair]
            del pair_to_nodes[best_pair]
    
    def run_train_bpe(self, input_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.pretokenize(input_path)
        self.merge()
        return self.vocab, self.merges
    
    # --- Helper Functions ---
    
    def allocate_id(self):
        new_id = self.next_id
        self.next_id += 1
        return new_id
    
    def find_chunk_boundaries(
        self,
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_size = file_size // desired_num_chunks
        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
        
    def process_chunk(self, args):
        """Worker function to process a single chunk"""
        start, end, input_path, PAT, special_tokens, byte_string_to_id, split_special_pattern  = args
        
        chunk_pretoken_counts = collections.defaultdict(int)
        buffer_size = 10 * 1024 * 1024 
        
        with open(input_path, "rb") as f:
            f.seek(start)
            bytes_to_process = end - start
            
            while bytes_to_process > 0:
                read_size = min(buffer_size, bytes_to_process)
                chunk_bytes = f.read(read_size)
                
                if not chunk_bytes:
                    break
                    
                chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
                text_parts = re.split(split_special_pattern, chunk_str) if split_special_pattern else [chunk_str]
            
                for part in text_parts:
                    if not part:
                        continue
                    if part in special_tokens:
                        # This part is a special token. It's treated as an atomic unit.
                        token_bytes = part.encode('utf-8')
                        if token_bytes in byte_string_to_id:
                            token_id = byte_string_to_id[token_bytes]
                            # The special token forms a sequence of its own, with length 1.
                            chunk_pretoken_counts[(token_id,)] += 1
                    else:
                        # This part is a regular text segment. Apply the base pre-tokenization regex.
                        for match in re.finditer(PAT, part):
                            token_str = match.group()
                            # Convert the pre-tokenized string into a sequence of byte-level IDs.
                            current_token_ids = [
                                byte_string_to_id[bytes([b_val])] 
                                for b_val in token_str.encode("utf-8")
                            ]
                            if current_token_ids:
                                chunk_pretoken_counts[tuple(current_token_ids)] += 1
                
                bytes_to_process -= read_size
        
        return chunk_pretoken_counts    



class BPETokenizer:
    """A lightweight BPE Tokenizer for inference."""
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], 
                       special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        # sort special tokens by length in descending order
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        
        self.byte_string_to_id = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)} # { pair : rank (position in the dict) }
        
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b'\n'
        if special_tokens:
            self.split_special_pattern = f"({ '|'.join([re.escape(st) for st in special_tokens]) })"
        else:
            self.split_special_pattern = None
      
    # --- Main Functions ---
    
    @classmethod    
    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, 
                        special_tokens: list[str] | None = None):
        '''
        Class method that constructs and return a Tokenizer from a serialized vocabulary and 
        list of merges and a list of special tokens.
        '''
        
        import json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(f) # (tokens: count)
            vocab = {int(token_id): token_str.encode("utf-8") 
                     for token_str, token_id in data.items()} # (int, bytes)

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) == 2:
                    p1 = parts[0].encode("utf-8")
                    p2 = parts[1].encode("utf-8")
                    merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)
        
    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        ids = []
        text_parts = self._split_with_special_tokens(text)
        for part in text_parts:
            if not part:
                continue
            
            if part in self.special_tokens:
                token_bytes = part.encode('utf-8')
                if token_bytes in self.byte_string_to_id:
                    token_id = self.byte_string_to_id[token_bytes]
                    ids.append(token_id)
            else:
                for match in re.finditer(self.PAT, part):
                    token_str = match.group()
                    tokens = self._apply_bpe(token_str.encode('utf-8'))
                    token_ids = [self.byte_string_to_id[b] for b in tokens]
                    ids.extend(token_ids)
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        '''
        for text in iterable:
            yield from self.encode(text)
            
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text
        '''
        bytes_list = [self.vocab[id] for id in ids]
        # using errors='replace' automatically replace malformed data with the replacement marker.
        return b''.join(bytes_list).decode("utf-8", errors="replace")


    # --- Helper Functions ---

    @staticmethod
    def get_pairs(parts: list[bytes]) -> set[tuple[bytes, bytes]]:
        return set(zip(parts, parts[1:]))

    def _apply_bpe(self, tokens: list[bytes]) -> list[bytes]:
        '''Apply BPE merges to a list of byte tokens'''
        
        parts = [bytes([b]) for b in tokens] # split tokens into single bytes
        if len(parts) < 2:
            return parts
        
        while True:
            pairs = self.get_pairs(parts)
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.merge_ranks.get(pair, float('inf')))
            if best_pair not in self.merges:
                break
            
            p1, p2 = best_pair
            new_parts = []
            i = 0
            while i < len(parts):
                if parts[i] == p1 and i + 1 < len(parts) and parts[i+1] == p2:
                    new_parts.append(p1 + p2)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
            if len(parts) == 1:
                break
            
        return parts

    def _split_with_special_tokens(self, text: str) -> list[str]:
        '''
        Split text while preserving special tokens, with proper handling of overlapping tokens.
        '''
        if not self.special_tokens:
            return [text]
        
        parts = []
        i = 0
        while i < len(text):
            matched_token = None
            matched_length = 0
            
            for token in self.special_tokens:
                if text[i:].startswith(token):
                    matched_token = token
                    matched_length = len(token)
                    break
            
            if matched_token:
                if parts and not parts[-1]:
                    parts.pop()
                parts.append(matched_token)
                i += matched_length
            else:
                if not parts or parts[-1] in self.special_tokens:
                    parts.append('')
                parts[-1] += text[i]
                i += 1
        
        return [part for part in parts if part]



# --- TESTING ---

def train_and_profile_cli():
    """
    Command-line interface function to train the tokenizer and profile its performance.
    """
    
    import argparse
    import cProfile
    import pstats
    import time
    import json
    import tracemalloc
    
    tracemalloc.start()
    
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a text file.")
    parser.add_argument("input_path", type=str, help="Path to the training text file.")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Desired vocabulary size.")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=["<|endoftext|>"], help="List of special tokens.")
    args = parser.parse_args()
    
    base_path, _ = os.path.splitext(args.input_path)
    merges_file_path =  f"{base_path}-vocab_size_{args.vocab_size}-merges.txt"
    vocab_file_path =   f"{base_path}-vocab_size_{args.vocab_size}-vocab.json"
    
    # --- Profiling ---
    profiler = cProfile.Profile()
    profiler.enable()
    
    print("="*50)
    print("🚀 Initializing BPETrainer...")
    tokenizer = BPETrainer(vocab_size=args.vocab_size, special_tokens=args.special_tokens)
    print("Configuration:")
    print(f"  - Vocab Size: {args.vocab_size}")
    print(f"  - Special Tokens: {args.special_tokens}")
    print(f"  - Input File: {args.input_path}")
    print("="*50)
    
    print("\nStep 1: Pre-tokenizing input file...")
    start_time = time.time()
    tokenizer.pretokenize(args.input_path)
    end_time = time.time()
    print(f"✅ Pre-tokenization complete in {end_time - start_time:.2f} seconds.")
    print(f"   Found {len(tokenizer.pretoken_counts)} unique pre-tokenized words/chunks.")
    
    print("\nStep 2: Learning BPE merges...")
    start_time = time.time()
    tokenizer.merge()
    end_time = time.time()
    print(f"✅ Merging complete in {end_time - start_time:.2f} seconds.")
    print(f"   Learned {len(tokenizer.merges)} merges. Final vocab size: {len(tokenizer.vocab)}")
    
    # --- Find and report the longest token ---
    if tokenizer.vocab:
        longest_token_bytes = max(tokenizer.vocab.values(), key=len)
        print(f"   Longest token has {len(longest_token_bytes)} bytes.")
        try:
            readable_token = longest_token_bytes.decode('utf-8', errors='replace')
            print(f"   Longest token content: '{readable_token}'")
        except:
             print(f"   Longest token (raw bytes): {longest_token_bytes}")
    
    print("\nStep 3: Saving vocabulary and merges...")
    try:
        inverted_vocab = {v.decode('utf-8'): k for k, v in tokenizer.vocab.items()}
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
        print(f"   Vocabulary saved to {vocab_file_path}")
    except Exception as e:
        print(f"   Error saving vocabulary file: {e}")
    
    try:
        with open(merges_file_path, "w", encoding="utf-8") as f:
            for p1, p2 in tokenizer.merges:
                f.write(f"{p1.decode('utf-8')} {p2.decode('utf-8')}\n")
        print(f"   Merges saved to {merges_file_path}")
    except Exception as e:
        print(f"   Error saving merges file: {e}")
    
    print("\n🎉 Training complete!")
    print("="*50)
    profiler.disable()
    
    # --- Resource Usage ---
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n--- Resource Usage ---")
    print(f"✅ Peak memory usage: {peak_mem / 1024**2:.2f} MB")
    
    print("\n--- CProfile Performance Analysis (Top 20) ---")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)

if __name__ == "__main__":
    train_and_profile_cli()