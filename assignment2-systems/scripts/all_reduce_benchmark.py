import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def benchmark_all_reduce(rank, world_size, backend, size_bytes, device):
    setup(rank, world_size, backend)
    num_elements = size_bytes // 4  # float32
    data = torch.randn(num_elements, device=device, dtype=torch.float32)
    
    # Warm-up
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    dist.all_reduce(data, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000  # ms
    
    # Gather times
    times = torch.tensor([elapsed], dtype=torch.float32, device=device)
    gathered_times = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(world_size)]
    dist.all_gather(gathered_times, times)
    
    if rank == 0:
        avg_time = sum(t.item() for t in gathered_times) / world_size
        print(f"backend={backend}, device={device}, size={size_bytes/1024/1024:.0f}MB, "
              f"world_size={world_size}, avg_time={avg_time:.3f}ms")


def main():
    configs = [
        ("gloo", "cpu", 2), ("gloo", "cpu", 4), ("gloo", "cpu", 6),
        ("nccl", "cuda", 2), ("nccl", "cuda", 4), ("nccl", "cuda", 6)
    ]
    sizes = [1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024]  # 1MB, 10MB, 100MB, 1GB

    for backend, device, world_size in configs:
        if backend == "nccl" and not torch.cuda.is_available():
            continue
        for size in sizes:
            mp.spawn(
                benchmark_all_reduce,
                args=(world_size, backend, size, device),
                nprocs=world_size,
                join=True
            )


if __name__ == "__main__":
    main()
