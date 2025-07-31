from collections.abc import Callable, Iterable
from typing import Optional
import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
import numpy as np


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute the cross entropy loss.
    """

    max_logits = torch.max(inputs, dim=-1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_logits), dim=-1))
    selected = inputs[torch.arange(inputs.shape[0]), targets]
    return torch.mean(log_sum_exp - selected + max_logits)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2 parameter: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p
                t = state.get(
                    "t", 0
                )  # Get iteration number from the state, or initial value.
                m = state.get("exp_avg", torch.zeros_like(p.data))
                v = state.get("exp_avg_sq", torch.zeros_like(p.data))

                t += 1
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data.mul_(1 - lr * weight_decay)  # Apply weight decay.

                m.mul_(beta1).add_(
                    grad, alpha=1 - beta1  # beta1 * m + (1 - beta1) * grad
                )  # Update the first moment estimate.
                v.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2  # beta2 * v + (1 - beta2) * grad^2
                )  # Update the second moment estimate.
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                denom = torch.sqrt(v).add_(eps)
                p.data.sub_(m.div(denom).mul(lr_t))  # Update weight tensor in-place.

                state["t"] = t
                state["exp_avg"] = m
                state["exp_avg_sq"] = v

        return loss


def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Return the learning rate according to the scheduler.
    t:          int current iteration
    alpha_max:  float maximum learning rate
    alpha_min:  float minimum learning rate
    T_w:        int warmup period
    T_c:        int cosine decay period
    """
    if t < T_w:
        return t / T_w * alpha_max
    elif t < T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (
            alpha_max - alpha_min
        )
    else:
        return alpha_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.
    """

    grads = [p.grad for p in parameters if p is not None and p.grad is not None]
    if len(grads) == 0:
        return

    l2_norm = torch.norm(torch.cat([g.reshape(-1) for g in grads]))
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + 1e-6)  # Add epsilon for numerical stability
        for g in grads:
            g.mul_(scale)
