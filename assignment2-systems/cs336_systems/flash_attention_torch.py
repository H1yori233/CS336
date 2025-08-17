import torch
import math
from einops import rearrange


class FlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        ctx.B_r = 16
        ctx.B_c = 16
        Q = rearrange(Q, "b (Tr Br) d -> Tr b Br d", Br=ctx.B_r)
        K = rearrange(K, "b (Tc Bc) d -> Tc b Bc d", Bc=ctx.B_c)
        V = rearrange(V, "b (Tc Bc) d -> Tc b Bc d", Bc=ctx.B_c)

        # calculate the number of tiles
        T_r, batch_size, _, d = Q.shape
        T_c = K.shape[0]

        # initialize O, L
        O = torch.zeros_like(Q)
        L = torch.zeros(T_r, batch_size, ctx.B_r, device=Q.device, dtype=Q.dtype)

        # compute attention
        for i in range(T_r):
            Q_i = Q[i]  # (b, B_r, d)
            O_i = torch.zeros((batch_size, ctx.B_r, d), device=Q.device, dtype=Q.dtype)
            L_i = torch.zeros((batch_size, ctx.B_r), device=Q.device, dtype=Q.dtype)
            m_i = torch.full(
                (batch_size, ctx.B_r), float("-inf"), device=Q.device, dtype=Q.dtype
            )

            for j in range(T_c):
                K_j = K[j]  # (b, B_c, d)
                V_j = V[j]  # (b, B_c, d)
                m_prev, l_prev, O_prev = m_i, L_i, O_i

                S_ij = Q_i @ K_j.transpose(1, 2) / math.sqrt(d)  # (b, B_r, B_c)
                if is_causal:
                    mask = torch.triu(
                        torch.ones(Q_i.size(1), K_j.size(1), device=Q_i.device), 1
                    ).bool()  # (B_r, B_c)
                    S_ij = S_ij.masked_fill(mask, float("-inf"))

                m_ij = S_ij.max(dim=-1).values
                m_i = torch.max(m_prev, m_ij)  # (b, B_r,)
                P_ij = torch.exp(S_ij - m_i.unsqueeze(-1))  # (b, B_r, B_c)

                exp_m_diff = torch.exp(m_prev - m_i)  # (b, B_r,)
                L_i = exp_m_diff * l_prev + P_ij.sum(dim=-1)

                O_prev_scaled = exp_m_diff.unsqueeze(-1) * O_prev  # (b, B_r, d)
                O_i = O_prev_scaled + P_ij @ V_j

            O[i] = (1.0 / L_i).unsqueeze(-1) * O_i
            L[i] = m_i + torch.log(L_i)

        O = rearrange(O, "Tr b Br d -> b (Tr Br) d")
        L = rearrange(L, "Tr b Br -> b (Tr Br)")
        Q = rearrange(Q, "Tr b Br d -> b (Tr Br) d")
        K = rearrange(K, "Tc b Bc d -> b (Tc Bc) d")
        V = rearrange(V, "Tc b Bc d -> b (Tc Bc) d")

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        Q = rearrange(Q, "b (Tr Br) d -> Tr b Br d", Br=ctx.B_r)
        K = rearrange(K, "b (Tc Bc) d -> Tc b Bc d", Bc=ctx.B_c)
        V = rearrange(V, "b (Tc Bc) d -> Tc b Bc d", Bc=ctx.B_c)
        O = rearrange(O, "b (Tr Br) d -> Tr b Br d", Br=ctx.B_r)
        L = rearrange(L, "b (Tr Br) -> Tr b Br", Br=ctx.B_r)
        dO = rearrange(grad_output, "b (Tr Br) d -> Tr b Br d", Br=ctx.B_r)

        # calculate the number of tiles
        T_r, batch_size, _, d = Q.shape
        T_c = K.shape[0]
        is_causal = ctx.is_causal

        # initialize grads
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        scale = 1.0 / math.sqrt(d)
        D = (dO * O).sum(dim=-1)  # (Tr, b, Br,)

        # compute grad
        for j in range(T_c):
            K_j, V_j = K[j], V[j]  # (b, B_c, d)
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)

            for i in range(T_r):
                Q_i, O_i, dO_i, dQ_i = Q[i], O[i], dO[i], dQ[i]  # (b, B_r, d)
                L_i, D_i = L[i], D[i]  # (b, B_r,)

                S_ij = Q_i @ K_j.transpose(1, 2) * scale  # (b, B_r, B_c)
                if is_causal:
                    mask = torch.triu(
                        torch.ones(Q_i.size(1), K_j.size(1), device=Q_i.device), 1
                    ).bool()  # (B_r, B_c)
                    S_ij = S_ij.masked_fill(mask, float("-inf"))

                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (b, B_r, B_c)
                dV_j = dV_j + P_ij.transpose(1, 2) @ dO_i  # (b, B_c, d)
                dP_ij = dO_i @ V_j.transpose(1, 2)  # (b, B_r, B_c)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # (b, B_r, B_c)

                # S_ij = (Q_i @ K_j^T) * scale, so dQ_i, dK_j needs `* scale`
                dQ_i = dQ_i + dS_ij @ K_j * scale  # (b, B_r, d)
                dQ[i] = dQ_i
                dK_j = dK_j + dS_ij.transpose(1, 2) @ Q_i * scale  # (b, B_c, d)

            dK[j], dV[j] = dK_j, dV_j

        dQ = rearrange(dQ, "Tr b Br d -> b (Tr Br) d")
        dK = rearrange(dK, "Tc b Bc d -> b (Tc Bc) d")
        dV = rearrange(dV, "Tc b Bc d -> b (Tc Bc) d")

        return dQ, dK, dV, None
