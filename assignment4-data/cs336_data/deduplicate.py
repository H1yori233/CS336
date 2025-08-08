import os
from typing import Any
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata
import regex as re
import random


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    # Ensure the output directory exists
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    line_counts = Counter()

    # Count the frequency of each line across all files
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                line_counts[line] += 1

    # Write unique lines to new files
    for input_file in input_files:
        input_file_path = Path(input_file)
        output_file_path = output_dir_path / input_file_path.name

        with open(input_file_path, "r", encoding="utf-8") as in_f, open(
            output_file_path, "w", encoding="utf-8"
        ) as out_f:
            for line in in_f:
                if line_counts[line] == 1:
                    out_f.write(line)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text)  # applying NFD
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # remove accents
    text = text.lower()  # lowercasing
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text


def get_ngrams(text: str, n: int) -> set:
    """
    For: "I love language models", n = 2
    return: {"I love", "love language", "language models"}
    """
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def hash_ngram(ngram: str, seed: int) -> int:
    combined = f"{ngram}_{seed}"
    return abs(hash(combined)) % (2**31 - 1)


def compute_minhash_signature(ngrams: set, num_hashes: int) -> list:
    signature = []
    for i in range(num_hashes):
        if ngrams:
            min_hash = min(hash_ngram(ngram, i) for ngram in ngrams)
        else:
            min_hash = 0
        signature.append(min_hash)
    return signature


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    # Ensure the output directory exists
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # MinHashing
    doc_signatures = {}
    doc_ngrams = {}
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # First, normalize the text
        content = normalize_text(content)
        ngrams_set = get_ngrams(content, ngrams)
        signature = compute_minhash_signature(ngrams_set, num_hashes)

        doc_ngrams[input_file] = ngrams_set
        doc_signatures[input_file] = signature

    # LSH
    rows_per_band = num_hashes // num_bands
    buckets = defaultdict(list)

    # doc_A: [12, 45, 66, 23, 88, 90]   num_hashes = 6
    # doc_B: [12, 45, 66, 24, 77, 90]   num_bands = 3
    # doc_C: [33, 99, 11, 22, 55, 66]   rows_per_band = num_hashes // num_bands = 2

    for doc_path, signature in doc_signatures.items():
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_hash = tuple(signature[start:end])
            buckets[(band_idx, band_hash)].append(doc_path)

    # buckets = (0, (12, 45)): [A, B], (1, (66, 23)): [A], (1, (66, 24)): [B],
    #           (2, (88, 90)): [A], (2, (77, 90)): [B],
    #           (0, (33, 99)): [C], (1, (11, 22)): [C], (2, (55, 66)): [C]

    candidate_pairs = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) > 1:
            for i in range(len(bucket_docs)):
                for j in range(i + 1, len(bucket_docs)):
                    candidate_pairs.add((bucket_docs[i], bucket_docs[j]))

    # candidate_pairs = { ('A', 'B') }

    # Jaccard Similarity
    duplicates = set()
    clusters = []
    for doc1, doc2 in candidate_pairs:
        ngrams1 = doc_ngrams[doc1]
        ngrams2 = doc_ngrams[doc2]
        similarity = jaccard_similarity(ngrams1, ngrams2)

        # clusters = {("doc1", "doc2"), ("doc2", "doc3"), ("doc4", "doc5")}
        if similarity >= jaccard_threshold:
            found_cluster = None
            for cluster in clusters:
                if doc1 in cluster or doc2 in cluster:
                    cluster.update([doc1, doc2])
                    found_cluster = cluster
                    break

            if found_cluster is None:
                clusters.append({doc1, doc2})
        # clusters = {("doc1", "doc2", "doc3"), ("doc4", "doc5")}

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i] & clusters[j]:
                    clusters[i] = clusters[i] | clusters[j]
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    # random keep one, others are duplicates
    for cluster in clusters:
        cluster_list = list(cluster)
        keep = random.choice(cluster_list)
        for doc in cluster_list:
            if doc != keep:
                duplicates.add(doc)

    for input_file in input_files:
        if input_file not in duplicates:  # check
            input_path = Path(input_file)
            output_path = output_dir_path / input_path.name

            with open(input_file, "r", encoding="utf-8") as in_f, open(
                output_path, "w", encoding="utf-8"
            ) as out_f:
                out_f.write(in_f.read())
