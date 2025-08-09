import argparse
from timeit import default_timer as timer
import statistics as stats
import os
from datetime import datetime

import pandas as pd
import torch

try:
    from cs336_basics.model import TransformerLM as _ModelClass
except Exception:  # pragma: no cover
    from cs336_basics.model import BasicsTransformerLM as _ModelClass


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(preferred: str, device: str) -> torch.dtype:
    preferred = preferred.lower()
    if preferred == "fp16":
        return torch.float16 if device == "cuda" else torch.float32
    if preferred == "bf16":
        # bf16 on CUDA if available; CPU bf16 tensor ops are fine for benchmarking
        return torch.bfloat16
    return torch.float32


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmarking of TransformerLM (forward and backward)",
    )
    # Model config
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Benchmark config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run forward+backward if set, else forward-only",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation dtype",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to write markdown table",
    )
    parser.add_argument(
        "--outfile_prefix",
        type=str,
        default="benchmark",
        help="Filename prefix for exported markdown table",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device()
    dtype = resolve_dtype(args.dtype, device)

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    model = _ModelClass(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device)

    # Set compute dtype
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)

    # Random input and labels
    batch = args.batch_size
    seq = args.context_length
    vocab = args.vocab_size
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)

    # Simple loss for backward benchmarking
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, S, V) -> (B*S, V); targets: (B, S) -> (B*S)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), targets.view(-1)
        )

    # Optional optimizer (no weight decay, simple)
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-3) if args.backward else None
    )

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup
    model.train(True)
    for _ in range(args.warmup_steps):
        logits = model(x)
        if args.backward:
            loss = compute_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        synchronize()

    # Measured steps
    times_ms = []
    for _ in range(args.measure_steps):
        t0 = timer()
        logits = model(x)
        if args.backward:
            loss = compute_loss(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        synchronize()
        t1 = timer()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = stats.mean(times_ms)
    std_ms = stats.pstdev(times_ms) if len(times_ms) > 1 else 0.0

    mode = "fwd+bwd" if args.backward else "fwd"
    print(
        f"device={device}, dtype={dtype}, batch={batch}, seq={seq}, d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}, d_ff={args.d_ff}"
    )
    print(
        f"{mode}: {mean_ms:.2f} ms/step ± {std_ms:.2f} (over {args.measure_steps} steps, warmup={args.warmup_steps})"
    )

    # Export single-row results table with all parameters
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {
        "model_name": args.outfile_prefix,
        "timestamp": timestamp,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "mode": mode,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mean_ms_per_step": round(mean_ms, 3),
        "std_ms_per_step": round(std_ms, 3),
    }

    unified_filename = os.path.join(args.output_dir, "benchmark_results.md")

    if os.path.exists(unified_filename):
        try:
            existing_df = pd.read_csv(unified_filename, sep="|", header=0, skiprows=[1])
            existing_df.drop(existing_df.columns[[0, -1]], axis=1, inplace=True)
            existing_df.columns = existing_df.columns.str.strip()
            for col in existing_df.columns:
                if col not in row:
                    existing_df = existing_df.drop(columns=[col])

            new_df = pd.DataFrame([row])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            combined_df = pd.DataFrame([row])
    else:
        combined_df = pd.DataFrame([row])

    with open(unified_filename, "w", encoding="utf-8") as f:
        f.write(combined_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
