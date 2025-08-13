import argparse
import math
import os
import statistics as stats
from timeit import default_timer as timer

import torch
from typing import Callable


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(preferred: str, device: str) -> torch.dtype:
    preferred = preferred.lower()
    if preferred == "fp16":
        return torch.float16 if device == "cuda" else torch.float32
    if preferred == "bf16":
        return torch.bfloat16
    return torch.float32


def attention_naive(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def format_bytes(num_bytes: int) -> str:
    if num_bytes is None:
        return "-"
    gb = num_bytes / (1024**3)
    mb = num_bytes / (1024**2)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{mb:.2f} MB"


def run_case(
    batch_size: int,
    d_model: int,
    seq_len: int,
    iters: int,
    warmup_iters: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    attention_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    Q = torch.randn(
        batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True
    )
    K = torch.randn(
        batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True
    )
    V = torch.randn(
        batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True
    )

    # Warmup (also triggers compilation if attention_fn is compiled)
    for _ in range(warmup_iters):
        out = attention_fn(Q, K, V)
        loss = out.sum()
        loss.backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        synchronize(device)

    # Measure memory just after forward (before backward)
    mem_before_bwd_bytes = None
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        out = attention_fn(Q, K, V)
        synchronize(device)
        if device == "cuda":
            mem_before_bwd_bytes = torch.cuda.memory_allocated()
        # cleanup the forward graph (via backward) before real timing loops
        out.sum().backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        synchronize(device)
    except torch.cuda.OutOfMemoryError:
        raise

    # Timings
    fwd_times_ms = []
    bwd_times_ms = []

    for _ in range(iters):
        try:
            synchronize(device)
            t0 = timer()
            out = attention_fn(Q, K, V)
            synchronize(device)
            t1 = timer()
            fwd_times_ms.append((t1 - t0) * 1000.0)

            loss = out.sum()
            t2 = timer()
            loss.backward()
            synchronize(device)
            t3 = timer()
            bwd_times_ms.append((t3 - t2) * 1000.0)

            Q.grad = None
            K.grad = None
            V.grad = None
        except torch.cuda.OutOfMemoryError:
            raise

    fwd_mean = stats.mean(fwd_times_ms) if fwd_times_ms else float("nan")
    fwd_std = stats.pstdev(fwd_times_ms) if len(fwd_times_ms) > 1 else 0.0
    bwd_mean = stats.mean(bwd_times_ms) if bwd_times_ms else float("nan")
    bwd_std = stats.pstdev(bwd_times_ms) if len(bwd_times_ms) > 1 else 0.0

    return {
        "status": "ok",
        "fwd_mean_ms": round(fwd_mean, 3),
        "fwd_std_ms": round(fwd_std, 3),
        "bwd_mean_ms": round(bwd_mean, 3),
        "bwd_std_ms": round(bwd_std, 3),
        "mem_before_bwd_bytes": mem_before_bwd_bytes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark naive (non-multihead) PyTorch attention for varying d_model and seq_len",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Fixed batch size (must be 8 per spec)",
    )
    parser.add_argument(
        "--dmodels",
        type=str,
        default="16,32,64,128",
        help="Comma-separated list of d_model values",
    )
    parser.add_argument(
        "--seqlens",
        type=str,
        default="256,1024,4096,8192,16384",
        help="Comma-separated list of sequence lengths",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of measured iterations"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Computation dtype",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--outfile",
        type=str,
        default="attention_benchmark.md",
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

    if args.batch_size != 8:
        print("Warning: spec requests batch_size=8; proceeding anyway.")

    device = resolve_device()
    if device != "cuda":
        print(
            "CUDA is required for this benchmark (for synchronization and memory metrics). Exiting."
        )
        return

    dtype = resolve_dtype(args.dtype, device)

    dmodels = [int(x.strip()) for x in args.dmodels.split(",") if x.strip()]
    seqlens = [int(x.strip()) for x in args.seqlens.split(",") if x.strip()]

    results = []

    print(
        f"device={device}, dtype={dtype}, batch={args.batch_size}, iters={args.iters}, warmup={args.warmup}"
    )

    # Prepare compiled attention if available
    compiled_available = hasattr(torch, "compile")
    compiled_attention = None
    if compiled_available:
        try:
            compiled_attention = torch.compile(
                attention_naive, mode=args.compile_mode, fullgraph=False
            )
        except Exception:
            compiled_available = False

    for d_model in dmodels:
        for seq_len in seqlens:
            # Eager
            try:
                res_eager = run_case(
                    batch_size=args.batch_size,
                    d_model=d_model,
                    seq_len=seq_len,
                    iters=args.iters,
                    warmup_iters=args.warmup,
                    device=device,
                    dtype=dtype,
                    seed=args.seed,
                    attention_fn=attention_naive,
                )
                mem_str_eager = (
                    format_bytes(res_eager["mem_before_bwd_bytes"])
                    if device == "cuda"
                    else "-"
                )
                print(
                    f"[eager]    d_model={d_model:4d}, seq_len={seq_len:6d} | fwd={res_eager['fwd_mean_ms']:.2f}±{res_eager['fwd_std_ms']:.2f} ms | "
                    f"bwd={res_eager['bwd_mean_ms']:.2f}±{res_eager['bwd_std_ms']:.2f} ms | mem_before_bwd={mem_str_eager}"
                )
                results.append(
                    {
                        "method": "eager",
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "status": "OK",
                        "fwd_mean_ms": res_eager["fwd_mean_ms"],
                        "fwd_std_ms": res_eager["fwd_std_ms"],
                        "bwd_mean_ms": res_eager["bwd_mean_ms"],
                        "bwd_std_ms": res_eager["bwd_std_ms"],
                        "mem_before_bwd_bytes": res_eager["mem_before_bwd_bytes"],
                    }
                )
            except torch.cuda.OutOfMemoryError:
                print(f"[eager]    d_model={d_model:4d}, seq_len={seq_len:6d} | OOM")
                torch.cuda.empty_cache()
                results.append(
                    {
                        "method": "eager",
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "status": "OOM",
                        "fwd_mean_ms": None,
                        "fwd_std_ms": None,
                        "bwd_mean_ms": None,
                        "bwd_std_ms": None,
                        "mem_before_bwd_bytes": None,
                    }
                )

            # Compiled (if available)
            if compiled_available and compiled_attention is not None:
                try:
                    res_comp = run_case(
                        batch_size=args.batch_size,
                        d_model=d_model,
                        seq_len=seq_len,
                        iters=args.iters,
                        warmup_iters=args.warmup,
                        device=device,
                        dtype=dtype,
                        seed=args.seed,
                        attention_fn=compiled_attention,
                    )
                    mem_str_comp = (
                        format_bytes(res_comp["mem_before_bwd_bytes"])
                        if device == "cuda"
                        else "-"
                    )
                    print(
                        f"[compiled] d_model={d_model:4d}, seq_len={seq_len:6d} | fwd={res_comp['fwd_mean_ms']:.2f}±{res_comp['fwd_std_ms']:.2f} ms | "
                        f"bwd={res_comp['bwd_mean_ms']:.2f}±{res_comp['bwd_std_ms']:.2f} ms | mem_before_bwd={mem_str_comp}"
                    )
                    results.append(
                        {
                            "method": "compiled",
                            "d_model": d_model,
                            "seq_len": seq_len,
                            "status": "OK",
                            "fwd_mean_ms": res_comp["fwd_mean_ms"],
                            "fwd_std_ms": res_comp["fwd_std_ms"],
                            "bwd_mean_ms": res_comp["bwd_mean_ms"],
                            "bwd_std_ms": res_comp["bwd_std_ms"],
                            "mem_before_bwd_bytes": res_comp["mem_before_bwd_bytes"],
                        }
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[compiled] d_model={d_model:4d}, seq_len={seq_len:6d} | OOM"
                    )
                    torch.cuda.empty_cache()
                    results.append(
                        {
                            "method": "compiled",
                            "d_model": d_model,
                            "seq_len": seq_len,
                            "status": "OOM",
                            "fwd_mean_ms": None,
                            "fwd_std_ms": None,
                            "bwd_mean_ms": None,
                            "bwd_std_ms": None,
                            "mem_before_bwd_bytes": None,
                        }
                    )
            else:
                results.append(
                    {
                        "method": "compiled",
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "status": "UNAVAILABLE",
                        "fwd_mean_ms": None,
                        "fwd_std_ms": None,
                        "bwd_mean_ms": None,
                        "bwd_std_ms": None,
                        "mem_before_bwd_bytes": None,
                    }
                )

    # Write a simple markdown table
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.outfile)
    headers = [
        "method",
        "d_model",
        "seq_len",
        "status",
        "fwd_mean_ms",
        "fwd_std_ms",
        "bwd_mean_ms",
        "bwd_std_ms",
        "mem_before_bwd",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([" --- "] * len(headers)) + "|")
    for r in results:
        mem_str = (
            format_bytes(r["mem_before_bwd_bytes"])
            if r["mem_before_bwd_bytes"] is not None
            else "OOM"
        )
        values = [
            r.get("method", "eager"),
            str(r["d_model"]),
            str(r["seq_len"]),
            r["status"],
            "" if r["fwd_mean_ms"] is None else f"{r['fwd_mean_ms']:.3f}",
            "" if r["fwd_std_ms"] is None else f"{r['fwd_std_ms']:.3f}",
            "" if r["bwd_mean_ms"] is None else f"{r['bwd_mean_ms']:.3f}",
            "" if r["bwd_std_ms"] is None else f"{r['bwd_std_ms']:.3f}",
            mem_str,
        ]
        lines.append("| " + " | ".join(values) + " |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
