import math
import torch
import triton.testing as ttesting
from cs336_systems.attention import TritonFlashAttentionAutogradFunction


def pytorch_attention(Q, K, V):
    B, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = (
        torch.arange(S, device=Q.device)[None, :]
        <= torch.arange(S, device=Q.device)[:, None]
    )
    scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def zero_grad(*tensors):
    for t in tensors:
        if t.grad is not None:
            t.grad.zero_()


def bench_case(name, fn, B, S, D, dtype, device):
    Q = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, S, D, device=device, dtype=dtype, requires_grad=True)

    def fwd():
        fn(Q, K, V)

    def bwd():
        out = fn(Q, K, V)
        loss = out.sum()
        loss.backward()
        zero_grad(Q, K, V)

    fwd_ms = ttesting.do_bench(lambda: fwd())
    out = fn(Q, K, V)
    bwd_ms = ttesting.do_bench(
        lambda: out.sum().backward(retain_graph=True) or zero_grad(Q, K, V)
    )
    e2e_ms = ttesting.do_bench(lambda: bwd())
    return fwd_ms, bwd_ms, e2e_ms


def main():
    device = "cuda"
    B = 1
    seqlens = [2**i for i in range(7, 17)]  # 128~65536
    dmodels = [2**i for i in range(4, 8)]  # 16~128
    dtypes = [torch.bfloat16, torch.float32]
    results = []

    for dtype in dtypes:
        for D in dmodels:
            for S in seqlens:
                triton_fn = lambda Q, K, V: TritonFlashAttentionAutogradFunction.apply(
                    Q, K, V, True
                )
                torch_fn = lambda Q, K, V: pytorch_attention(Q, K, V)
                fwd_ms, bwd_ms, e2e_ms = bench_case(
                    "triton", triton_fn, B, S, D, dtype, device
                )
                results.append(("triton", dtype, D, S, fwd_ms, bwd_ms, e2e_ms))
                # Only run PyTorch baseline for smaller sequence lengths to avoid OOM
                if S <= 8192:
                    fwd_ms, bwd_ms, e2e_ms = bench_case(
                        "torch", torch_fn, B, S, D, dtype, device
                    )
                    results.append(("torch", dtype, D, S, fwd_ms, bwd_ms, e2e_ms))

    print("| method | dtype | d_model | seq_len | fwd_ms | bwd_ms | e2e_ms |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r[0]} | {str(r[1]).split('.')[-1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} |"
        )


if __name__ == "__main__":
    main()
