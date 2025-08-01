from collections.abc import Callable, Iterable
from typing import Optional
import torch
from torch import Tensor
from jaxtyping import Float, Int
import math
import numpy as np


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
