import torch
import numpy as np
import numpy.typing as npt
import os
from typing import BinaryIO, IO


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.
    """

    # random sample the position
    ix = np.random.randint(dataset.size - context_length, size=batch_size)

    # get the input and target
    x = torch.stack(
        [torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in ix]
    )  # (batch_size, context_length)
    y = torch.stack(
        [
            torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64))
            for i in ix
        ]
    )  # (batch_size, context_length)

    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.
    """

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


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
    dtype: torch.dtype = torch.bfloat16,
):
    from cs336_basics.nn_utils import TransformerLM
    from cs336_basics.optimizer import (
        AdamW,
        lr_cosine_schedule,
        cross_entropy,
        gradient_clipping,
    )
    from tqdm import tqdm

    print("loading datasets ...")
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_data_path, dtype=np.uint16, mode="r")

    # get model
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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    print("start train ...")
    best_val_loss = float("inf")
    progress_bar = tqdm(range(num_iterations), desc="training")

    for i in progress_bar:
        current_lr = lr_cosine_schedule(
            t=i,
            alpha_max=lr,
            alpha_min=min_lr,
            T_w=warmup_steps,
            T_c=lr_decay_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        if i % eval_interval == 0 or i == num_iterations - 1:
            model.eval()  # switch to eval mode
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(10):
                    x, y = get_batch(val_data, batch_size, context_length, device)
                    logits = model(x)
                    loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
            val_loss /= 10
            model.train()  # back to train mode

            progress_bar.set_postfix(
                {
                    "valid_loss": f"{val_loss:.4f}",
                    "lr": f"{current_lr:.6f}",
                }
            )

            # save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_path:
                    save_checkpoint(model, optimizer, i, checkpoint_path)
                    print(
                        f"save new best checkpoint at {i} step, valid_loss: {best_val_loss:.4f}"
                    )

        # train one step
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()

        # update progress bar
        progress_bar.update(1)

    print("finish.")
