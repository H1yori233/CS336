import torch
import math
from einops import rearrange


class FlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        B_r = 16
        B_c = 16
        Q = rearrange(Q, "b (Tr Br) d -> Tr b Br d", Br=B_r)
        K = rearrange(K, "b (Tc Bc) d -> Tc b Bc d", Bc=B_c)
        V = rearrange(V, "b (Tc Bc) d -> Tc b Bc d", Bc=B_c)

        # calculate the number of tiles
        T_r, batch_size, _, d = Q.shape
        T_c = K.shape[0]

        # divide Q, K, V into blocks
        O = torch.zeros_like(Q)
        l = torch.zeros(T_r, batch_size, B_r, device=Q.device, dtype=Q.dtype)

        # compute attention
        for i in range(T_r):
            Q_i = Q[i]  # (b, B_r, d)
            O_i = torch.zeros((batch_size, B_r, d), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros((batch_size, B_r), device=Q.device, dtype=Q.dtype)
            m_i = torch.full(
                (batch_size, B_r), float("-inf"), device=Q.device, dtype=Q.dtype
            )

            for j in range(T_c):
                K_j = K[j]  # (b, B_c, d)
                V_j = V[j]  # (b, B_c, d)
                m_prev, l_prev, O_prev = m_i, l_i, O_i

                S_ij = torch.einsum("b r d, b c d -> b r c", Q_i, K_j) / math.sqrt(d)
                m_ij = S_ij.max(dim=-1).values
                m_i = torch.max(m_prev, m_ij)  # (b, B_r,)
                P_ij = torch.exp(S_ij - m_i.unsqueeze(-1))  # (b, B_r, B_c)

                exp_m_diff = torch.exp(m_prev - m_i)  # (b, B_r,)
                l_i = exp_m_diff * l_prev + P_ij.sum(dim=-1)

                scaling_diag = torch.diag_embed(exp_m_diff)  # (b, B_r, B_r)
                O_prev_scaled = torch.einsum(
                    "b r r, b r d -> b r d", scaling_diag, O_prev
                )
                PV_j = torch.einsum("b r c, b c d -> b r d", P_ij, V_j)
                O_i = O_prev_scaled + PV_j

            inv_l_i_diag = torch.diag_embed(1.0 / l_i)
            O[i] = torch.einsum("b r r, b r d -> b r d", inv_l_i_diag, O_i)
            l[i] = m_i + torch.log(l_i)

        O = rearrange(O, "Tr b Br d -> b (Tr Br) d")
        l = rearrange(l, "Tr b Br -> b (Tr Br)")
        Q = rearrange(Q, "Tr b Br d -> b (Tr Br) d")
        K = rearrange(K, "Tc b Bc d -> b (Tc Bc) d")
        V = rearrange(V, "Tc b Bc d -> b (Tc Bc) d")

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, l)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
