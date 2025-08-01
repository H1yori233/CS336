import argparse
import os
import sys
import time
import collections
from tqdm import tqdm
import heapq

from cs336_basics.tokenizer import BPETrainer, _PriorityQueueItem, _Node


class BPETrainerWithProgress(BPETrainer):
    """
    An inherited BPETrainer that adds a progress bar to the merge operation
    without modifying the original class.
    """

    def merge(self):
        """
        Overridden version of merge to include a tqdm progress bar.
        The core logic is identical to the parent class's merge method.
        """
        pair_counts: collections.defaultdict[tuple[int, int], int] = (
            collections.defaultdict(int)
        )
        pair_to_nodes: collections.defaultdict[tuple[int, int], set[_Node]] = (
            collections.defaultdict(set)
        )

        for token_id_list, count in self.pretoken_counts.items():
            if len(token_id_list) < 2:
                continue
            count_ref = {"count": count}
            head = _Node(token_id_list[0], count_ref)
            prev_node = head
            for i in range(1, len(token_id_list)):
                current_node = _Node(token_id_list[i], count_ref)
                prev_node.next = current_node
                current_node.prev = prev_node
                pair = (prev_node.id, current_node.id)

                if (
                    prev_node.id not in self.special_token_ids
                    and current_node.id not in self.special_token_ids
                ):
                    pair_counts[pair] += count
                    pair_to_nodes[pair].add(prev_node)

                prev_node = current_node

        pq = []
        for pair, count in pair_counts.items():
            p1_bytes, p2_bytes = self.vocab[pair[0]], self.vocab[pair[1]]
            item = _PriorityQueueItem(count, p1_bytes, p2_bytes, pair)
            heapq.heappush(pq, item)

        def update_stats(pair_to_update, delta, node_to_index):
            if not pair_to_update:
                return
            if (
                pair_to_update[0] in self.special_token_ids
                or pair_to_update[1] in self.special_token_ids
            ):
                return

            if delta > 0:
                pair_to_nodes[pair_to_update].add(node_to_index)
            else:
                pair_to_nodes[pair_to_update].discard(node_to_index)
            pair_counts[pair_to_update] += delta

            if pair_counts[pair_to_update] > 0:
                p1_b, p2_b = (
                    self.vocab[pair_to_update[0]],
                    self.vocab[pair_to_update[1]],
                )
                new_item = _PriorityQueueItem(
                    pair_counts[pair_to_update], p1_b, p2_b, pair_to_update
                )
                heapq.heappush(pq, new_item)

        num_merges_to_do = self.vocab_size - len(self.vocab)
        progress_bar = tqdm(range(num_merges_to_do), desc="Merging tokens")
        for _ in progress_bar:
            best_pair = None
            while pq:
                item = heapq.heappop(pq)
                if pair_counts.get(item.pair, 0) == item.count:
                    best_pair = item.pair
                    break

            if best_pair is None:
                progress_bar.close()
                print("No more pairs to merge. Stopping early.")
                break

            id1, id2 = best_pair

            new_id = self.allocate_id()
            token1_bytes, token2_bytes = self.vocab[id1], self.vocab[id2]
            new_token_bytes = token1_bytes + token2_bytes
            self.vocab[new_id] = new_token_bytes
            self.byte_string_to_id[new_token_bytes] = new_id
            self.merges.append((token1_bytes, token2_bytes))

            valid_pairs = []
            for node1 in pair_to_nodes[best_pair]:
                node2 = node1.next
                if node2 is not None and node1.id == id1 and node2.id == id2:
                    valid_pairs.append((node1, node2))

            for node1, node2 in valid_pairs:
                word_count = node1.count
                if node1.prev:
                    left_node = node1.prev
                    update_stats((left_node.id, node1.id), -word_count, left_node)
                    update_stats((left_node.id, new_id), word_count, left_node)
                if node2.next:
                    right_node = node2.next
                    update_stats((node2.id, right_node.id), -word_count, node2)
                    update_stats((new_id, right_node.id), word_count, node1)

                node1.id = new_id
                node1.next = node2.next
                if node2.next:
                    node2.next.prev = node1
            del pair_counts[best_pair]
            del pair_to_nodes[best_pair]
        else:
            progress_bar.close()


def main():
    """Main function to train the BPE tokenizer."""
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer from a text file."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input text file for training.",
    )
    parser.add_argument(
        "vocab_size",
        type=int,
        help="The desired final vocabulary size.",
    )
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="A list of special tokens to add to the vocabulary. (e.g., --special_tokens <|endoftext|> <|pad|>)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found at {args.input_path}")
        sys.exit(1)

    print(f"Starting tokenizer training for: {args.input_path}")
    print(f"Target vocabulary size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")

    # Initialize the trainer
    trainer = BPETrainerWithProgress(
        vocab_size=args.vocab_size, special_tokens=args.special_tokens
    )

    # --- Time pre-tokenization ---
    print("\nStarting pre-tokenization...")
    start_time = time.monotonic()
    trainer.pretokenize(args.input_path)
    end_time = time.monotonic()
    print(f"Pre-tokenization finished in {end_time - start_time:.2f}s.")

    # --- Time merging ---
    print("\nStarting merging...")
    start_time = time.monotonic()
    trainer.merge()
    end_time = time.monotonic()
    print(f"Merging finished in {end_time - start_time:.2f}s.")

    print("\nTraining complete.")

    # --- Save the results ---
    output_base_path = os.path.splitext(args.input_path)[0]
    # Assuming the save method is added to your BPETrainer class
    if hasattr(trainer, "save"):
        trainer.save(output_base_path)
    else:
        print("\nWarning: `save` method not found in BPETrainer. Results not saved.")

    # --- Find and print the longest token ---
    if trainer.vocab:
        longest_token = max(trainer.vocab.values(), key=len)
        print(f"\nLongest token length: {len(longest_token)}")
        print(f"Longest token (bytes): {longest_token}")


if __name__ == "__main__":
    main()
