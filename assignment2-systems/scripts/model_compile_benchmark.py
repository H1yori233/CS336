import argparse
from contextlib import nullcontext
import os
import statistics as stats
from timeit import default_timer as timer
from datetime import datetime

import torch
import pandas as pd

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
        return torch.bfloat16
    return torch.float32


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "-"
    gb = num_bytes / (1024**3)
    mb = num_bytes / (1024**2)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{mb:.2f} MB"


def build_model(args, device: str, dtype: torch.dtype) -> torch.nn.Module:
    model = _ModelClass(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device)

    # Set compute dtype if not using MP
    if not args.mixed_precision and dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=dtype)
    return model


def run_single_mode(
    args,
    method: str,
    device: str,
    dtype: torch.dtype,
) -> dict:
    # Inputs and labels
    batch = args.batch_size
    seq = args.context_length
    vocab = args.vocab_size

    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)

    def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), targets.view(-1)
        )

    model = build_model(args, device, dtype)

    # Compile model if requested
    compiled = False
    if method == "compiled":
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode=args.compile_mode, fullgraph=False)
            compiled = True
        else:
            return {
                "method": method,
                "status": "UNAVAILABLE",
            }

    optimizer = None
    scaler = None
    if args.backward:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        if args.mixed_precision and device == "cuda":
            scaler = torch.amp.GradScaler("cuda")

    if args.mixed_precision and device == "cuda":
        autocast_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    # Warmup
    model.train(True)
    for _ in range(args.warmup_steps):
        with autocast_context:
            logits = model(x)
            if args.backward:
                loss = compute_loss(logits, y)
        if args.backward:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        synchronize(device)

    # Measure memory after forward (before backward)
    mem_before_bwd_bytes = None
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        with autocast_context:
            logits = model(x)
        synchronize(device)
        if device == "cuda":
            mem_before_bwd_bytes = torch.cuda.memory_allocated()
        if args.backward:
            loss = compute_loss(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            synchronize(device)
    except torch.cuda.OutOfMemoryError:
        return {
            "method": method,
            "status": "OOM",
        }

    # Timed steps
    times_ms = []
    for _ in range(args.measure_steps):
        t0 = timer()
        with autocast_context:
            logits = model(x)
            if args.backward:
                loss = compute_loss(logits, y)
        if args.backward:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        synchronize(device)
        t1 = timer()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = stats.mean(times_ms)
    std_ms = stats.pstdev(times_ms) if len(times_ms) > 1 else 0.0

    return {
        "method": method,
        "status": "OK",
        "compiled": compiled,
        "mean_ms_per_step": round(mean_ms, 3),
        "std_ms_per_step": round(std_ms, 3),
        "mem_before_bwd_bytes": mem_before_bwd_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end benchmarking of TransformerLM with optional torch.compile",
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
        help="Run forward+backward+optimizer if set, else forward-only",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision autocast (BF16 on CUDA)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--outfile",
        type=str,
        default="model_compile_benchmark.md",
        help="Markdown file to write results into (inside output_dir)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Mode for torch.compile",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device()
    dtype = resolve_dtype(args.dtype, device)

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    # Run both methods
    methods = ["eager", "compiled"]
    rows = []
    for method in methods:
        res = run_single_mode(args, method, device, dtype)
        rows.append(res)

    mode = "fwd+bwd+opt" if args.backward else "fwd"

    # Build row dicts
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_dicts = []
    for r in rows:
        row_dicts.append(
            {
                "model_name": "model_compile_benchmark",
                "timestamp": timestamp,
                "method": r.get("method", "-"),
                "status": r.get("status", "-"),
                "mode": mode,
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "mixed_precision": args.mixed_precision,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "mean_ms_per_step": r.get("mean_ms_per_step"),
                "std_ms_per_step": r.get("std_ms_per_step"),
                "mem_before_bwd": format_bytes(r.get("mem_before_bwd_bytes")),
            }
        )

    # Merge into a unified markdown file, do not overwrite previous results
    os.makedirs(args.output_dir, exist_ok=True)
    unified_filename = os.path.join(args.output_dir, args.outfile)

    if os.path.exists(unified_filename):
        try:
            existing_df = pd.read_csv(unified_filename, sep="|", header=0, skiprows=[1])
            existing_df.drop(existing_df.columns[[0, -1]], axis=1, inplace=True)
            existing_df.columns = existing_df.columns.str.strip()
            # Drop columns not in our current schema to keep intersection
            for col in list(existing_df.columns):
                if col not in row_dicts[0]:
                    existing_df = existing_df.drop(columns=[col])
            new_df = pd.DataFrame(row_dicts)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            combined_df = pd.DataFrame(row_dicts)
    else:
        combined_df = pd.DataFrame(row_dicts)

    with open(unified_filename, "w", encoding="utf-8") as f:
        f.write(combined_df.to_markdown(index=False))

    print(f"Results written to: {unified_filename}")


if __name__ == "__main__":
    main()
