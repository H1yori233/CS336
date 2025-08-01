import torch
import numpy as np
import os
import sys
import time
from datetime import datetime
from typing import Dict
from tqdm import tqdm
import argparse

from cs336_basics.model import TransformerLM, generate
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.utils import (
    lr_cosine_schedule,
    cross_entropy,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)

# -------------------------------------------------------------

# -- Data and Checkpoint Paths --
VOCAB_FILE = "data/TinyStoriesV2-GPT4-train-vocab_size_10000-vocab.json"
MERGES_FILE = "data/TinyStoriesV2-GPT4-train-vocab_size_10000-merges.txt"

# Raw text data
RAW_TRAIN_DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
RAW_VAL_DATA_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

# Paths for tokenized binary files
TOKENIZED_TRAIN_PATH = "data/train.bin"
TOKENIZED_VAL_DATA_PATH = "data/valid.bin"

CHECKPOINT_PATH = "data/checkpoint_model.pt"  # Directory will be created
LOG_FILE = "data/log.md"

# -- Model Architecture --
VOCAB_SIZE = 10163  # TinyStories vocab size
CONTEXT_LENGTH = 256  # Max sequence length
D_MODEL = 512  # Model dimension
NUM_LAYERS = 4  # Number of transformer blocks
NUM_HEADS = 16  # Number of attention heads
D_FF = 1344  # Feed-forward dimension (~8/3 * d_model, multiple of 64)
ROPE_THETA = 10000.0  # RoPE theta parameter

# -- Training Hyperparameters --
BATCH_SIZE = 64  # Number of sequences per batch
NUM_ITERATIONS = 5000  # Total training steps (32*5000*256=40M tokens)
LR = 1e-3  # Maximum learning rate
WARMUP_STEPS = 500  # Steps for linear learning rate warmup
LR_DECAY_STEPS = 5000  # Steps for cosine decay
MIN_LR = 3e-5  # Minimum learning rate after decay
WEIGHT_DECAY = 0.1  # AdamW weight decay
GRAD_CLIP = 1.0  # Gradient clipping value (0 to disable)

# -- Logging and Evaluation --
EVAL_INTERVAL = 250  # Evaluate validation loss every N steps
NOTES = "TinyStories baseline: 4L/16H/512D, 40M tokens"

# -------------------------------------------------------------


def log_experiment(
    log_file: str,
    timestamp: str,
    best_val_loss: float,
    best_step: int,
    config: Dict,
    notes: str,
):
    """Appends the results of an experiment to a markdown log file."""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    header = "| Run Timestamp | Best Val Loss | Best Step | Total Steps | Batch Size | d_model | Layers | Heads | LR | Notes |\n"
    separator = "|---|---|---|---|---|---|---|---|---|---|\n"

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(header)
            f.write(separator)

    with open(log_file, "a") as f:
        log_entry = (
            f"| {timestamp} "
            f"| {best_val_loss:.4f} "
            f"| {best_step} "
            f"| {config.get('num_iterations', 'N/A')} "
            f"| {config.get('batch_size', 'N/A')} "
            f"| {config.get('d_model', 'N/A')} "
            f"| {config.get('num_layers', 'N/A')} "
            f"| {config.get('num_heads', 'N/A')} "
            f"| {config.get('lr', 'N/A'):.1E} "
            f"| {notes} |\n"
        )
        f.write(log_entry)


def training_together(
    device: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    train_data_path: str,
    val_data_path: str,
    batch_size: int,
    num_iterations: int,
    lr: float,
    warmup_steps: int,
    lr_decay_steps: int,
    min_lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_interval: int,
    checkpoint_path: str,
    log_file: str,
    notes: str,
    dtype: torch.dtype = torch.bfloat16,
    enable_compile: bool = False,
):
    """
    Main training and evaluation loop with integrated logging.
    """
    start_run_time = time.time()

    # Setup logging config
    config = {
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "lr": lr,
    }

    print("Loading datasets...")
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode="r")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )
    model.to(device)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters."
    )

    # Optional torch.compile for acceleration
    if enable_compile:
        model = torch.compile(model)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("Starting training...")
    best_val_loss = float("inf")
    best_step = 0
    progress_bar = tqdm(range(num_iterations), desc="Training")

    for i in progress_bar:
        current_lr = lr_cosine_schedule(i, lr, min_lr, warmup_steps, lr_decay_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        if i % eval_interval == 0 or i == num_iterations - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(10):  # Eval batches
                    x, y = get_batch(val_data, batch_size, context_length, device)
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
            val_loss /= 10
            model.train()

            progress_bar.set_postfix(
                {"valid_loss": f"{val_loss:.4f}", "lr": f"{current_lr:.6f}"}
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                if checkpoint_path:
                    save_checkpoint(model, optimizer, i, checkpoint_path)
                    print(
                        f"New best checkpoint at step {i}, valid_loss: {best_val_loss:.4f}"
                    )

        # Training step
        x, y = get_batch(train_data, batch_size, context_length, device)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()

        progress_bar.update(1)

    # End of training
    run_duration = time.time() - start_run_time
    print(f"Finished training in {run_duration:.2f} seconds.")

    # Log the final results
    print("Logging experiment results...")
    log_experiment(
        log_file=log_file,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        best_val_loss=best_val_loss,
        best_step=best_step,
        config=config,
        notes=notes,
    )
    print(f"Log saved to {log_file}")

    return model, best_val_loss


def generate_text_mode(args):
    """Generate text using trained model with configurable parameters."""

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    # Load tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=VOCAB_FILE, merges_filepath=MERGES_FILE
    )

    # Create model
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        device=device,
        dtype=dtype,
    )
    model.to(device)

    # Load checkpoint
    optimizer = AdamW(model.parameters(), lr=1e-4)  # dummy optimizer
    load_checkpoint(CHECKPOINT_PATH, model, optimizer)
    model.eval()

    # Generate text
    prompt_tokens = tokenizer.encode(args.prompt)
    stop_token_bytes = "<|endoftext|>".encode("utf-8")
    stop_token = tokenizer.byte_string_to_id.get(stop_token_bytes, None)

    print(f"Generating text with prompt: '{args.prompt}'")
    print(
        f"Parameters: temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}"
    )

    with torch.no_grad():
        generated_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            stop_token_id=stop_token,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    full_text = tokenizer.decode(prompt_tokens + generated_tokens)
    print(f"Generated: {full_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Transformer model or generate text from a checkpoint.",
        epilog="To train, run without arguments. To generate, provide a prompt and optional generation parameters.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        default=None,
        help="The initial text prompt for generation mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for text generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter for text generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    args = parser.parse_args()

    # Check if we are in generation mode
    if args.prompt:
        generate_text_mode(args)
        sys.exit(0)

    # --- Otherwise, run training mode ---
    # -- Data Preparation --
    # This block tokenizes the raw text and saves it to binary files.
    if not (os.path.exists(VOCAB_FILE) and os.path.exists(MERGES_FILE)):
        print("Tokenizer files not found. Please run the tokenizer training first.")
    else:
        tokenizer = BPETokenizer.from_files(
            vocab_filepath=VOCAB_FILE, merges_filepath=MERGES_FILE
        )

        loaded_vocab_size = len(tokenizer.vocab)
        print(f"DEBUG: Vocabulary size loaded from file: {loaded_vocab_size}")

        def tokenize_and_save(text_path, bin_path):
            if not os.path.exists(bin_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    num_lines = sum(1 for _ in f)

                with open(text_path, "r", encoding="utf-8") as f:
                    token_generator = tokenizer.encode_iterable(
                        tqdm(
                            f,
                            total=num_lines,
                            desc=f"Tokenizing {os.path.basename(text_path)}",
                        )
                    )
                    arr = np.fromiter(token_generator, dtype=np.uint16)

                arr.tofile(bin_path)
                print(f"Saved {len(arr)} tokens to {bin_path}.")
            else:
                print(f"Tokenized file already exists: {bin_path}")

        tokenize_and_save(RAW_TRAIN_DATA_PATH, TOKENIZED_TRAIN_PATH)
        tokenize_and_save(RAW_VAL_DATA_PATH, TOKENIZED_VAL_DATA_PATH)

    # -- Device and Precision --
    if torch.cuda.is_available():
        DEVICE = "cuda"
        DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.set_float32_matmul_precision("high")
    else:
        DEVICE = "cpu"
        DTYPE = torch.float32

    # --- Run Training ---
    if CHECKPOINT_PATH:
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    params_per_layer = 4 * D_MODEL**2 + 3 * D_MODEL * D_FF + 2 * D_MODEL
    total_params = NUM_LAYERS * params_per_layer + 2 * VOCAB_SIZE * D_MODEL + D_MODEL
    print(f"Model will have ~{total_params / 1e6:.2f}M parameters")
    print("Learn rate: ", LR)

    enable_compile = DEVICE == "cpu"
    if enable_compile:
        print("Enabling torch.compile for CPU acceleration...")

    model, best_loss = training_together(
        device=DEVICE,
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
        train_data_path=TOKENIZED_TRAIN_PATH,
        val_data_path=TOKENIZED_VAL_DATA_PATH,
        batch_size=BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        lr=LR,
        warmup_steps=WARMUP_STEPS,
        lr_decay_steps=LR_DECAY_STEPS,
        min_lr=MIN_LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        eval_interval=EVAL_INTERVAL,
        checkpoint_path=CHECKPOINT_PATH,
        log_file=LOG_FILE,
        notes=NOTES,
        dtype=DTYPE,
        enable_compile=enable_compile,
    )

    print(f"Training completed. Best loss: {best_loss:.4f}")
    print('Use: uv run train_model.py "your prompt" to generate text')
