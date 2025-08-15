import torch
import math
from einops import rearrange
import triton
import triton.language as tl


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

                S_ij = Q_i @ K_j.transpose(1, 2) / math.sqrt(d)
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
                P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))  # (b, B_r, B_c)
                dV_j = dV_j + P_ij.transpose(1, 2) @ dO_i  # (b, B_c, d)
                dP_ij = dO_i @ V_j.transpose(1, 2)  # (b, B_r, B_c)
                dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))  # (b, B_r, B_c)
                
                # S_ij = (Q_i @ K_j^T) * scale, so dQ_i, dK_j needs `* scale`
                dQ_i = dQ_i + dS_ij @ K_j * scale # (b, B_r, d)
                dQ[i] = dQ_i
                dK_j = dK_j + dS_ij.transpose(1, 2) @ Q_i * scale  # (b, B_c, d)

            dK[j], dV[j] = dK_j, dV_j
        
        dQ = rearrange(dQ, "Tr b Br d -> b (Tr Br) d")
        dK = rearrange(dK, "Tc b Bc d -> b (Tc Bc) d")
        dV = rearrange(dV, "Tc b Bc d -> b (Tc Bc) d")
        
        return dQ, dK, dV, None


def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,  # (b, (Tr Br), d)
    stride_kb, stride_kk, stride_kd,  # (b, (Tc Bc), d)
    stride_vb, stride_vk, stride_vd,  # (b, (Tc Bc), d)
    stride_ob, stride_oq, stride_od,  # (b, (Tr Br), d)
    stride_lb, stride_lq,  # (b, (Tr Br),)
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # D dimension is major
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # load Q_i to on-chip SRAM
    # since Q_TILE_SIZE might not divide N_QUERIES, and D is fixed, check only the first dim
    Q_i = tl.load(
        Q_block_ptr, boundary_check=(0,), padding_option="zero"
    ).to(tl.float32)  # (B_r, d)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    T_c = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(T_c):
        # load K_j, V_j to on-chip SRAM
        # since K_TILE_SIZE might not divide N_KEYS, and D is fixed, check only the first dim
        K_j = tl.load(
            K_block_ptr, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)  # (B_c, d)
        V_j = tl.load(
            V_block_ptr, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)  # (B_c, d)
        m_prev, l_prev, O_prev = m_i, L_i, O_i

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (Br, Bc)
        if is_causal:
            query_pos = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) # (B_r,)
            key_pos = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (B_c,)
            mask = query_pos[:, None] >= key_pos[None, :] # (B_r, 1) >= (1, B_c) -> (B_r, B_c)
            S_ij = tl.where(mask, S_ij, -float("inf"))

        m_ij = tl.max(S_ij, axis=1)  # (B_r,)
        m_i = tl.maximum(m_prev, m_ij)  # (B_r,), element-wise maximum

        P_ij = tl.exp(S_ij - m_i[:, None])  # (B_r, B_c)
        exp_m_diff = tl.exp(m_prev - m_i)  # (B_r,)
        L_ij = tl.sum(P_ij, axis=1)  # (B_r,)
        L_i = exp_m_diff * l_prev + L_ij  # (B_r,)
        O_i = tl.dot(
            P_ij.to(V_j.dtype), V_j, acc=exp_m_diff[:, None] * O_prev
        )  # (B_r, d)

        # advance block pointers at the end of the loop.
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = (1.0 / L_i[:, None]) * O_i
    L_i = m_i + tl.log(L_i)

    # Write O_i, L_i to HBM
    #  Since Q_TILE_SIZE might not divide N_QUERIES, and D is fixed, check only the first dim
    tl.store(O_block_ptr, O_i.to(O_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, L_i.to(L_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_backward_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    dO_ptr,
    stride_qb, stride_qq, stride_qd,  # (b, (Tr Br), d)
    stride_kb, stride_kk, stride_kd,  # (b, (Tc Bc), d)
    stride_vb, stride_vk, stride_vd,  # (b, (Tc Bc), d)
    stride_ob, stride_oq, stride_od,  # (b, (Tr Br), d)
    stride_lb, stride_lq,  # (b, (Tr Br),)
    stride_db, stride_dd,  # (b, (Tr Br),)
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # D dimension is major
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),  # shared in all tiles
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dd,),
        offsets=(0,),  # shared in all tiles
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),  # D dimension is major
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),  # (B_c, d)
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),  # shared in all tiles
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # load K_j, V_j to on-chip SRAM
    # since K_TILE_SIZE might not divide N_KEYS, and D is fixed, check only the first dim
    K_j = tl.load(
        K_block_ptr, boundary_check=(0,), padding_option="zero"
    ).to(tl.float32)  # (B_c, d)
    V_j = tl.load(
        V_block_ptr, boundary_check=(0,), padding_option="zero"
    ).to(tl.float32)  # (B_c, d)
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    T_r= tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    for i in range(T_r):
        Q_i = tl.load(
            Q_block_ptr, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)  # (B_r, d)
        dO_i = tl.load(
            dO_block_ptr, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)  # (B_r, d)
        L_i = tl.load(
            L_block_ptr, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)  # (B_r,)
        D_i = tl.load(
            D_block_ptr, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)  # (B_r,)
        
        # on-chip compute
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale  # (B_r, B_c)
        if is_causal:
            query_pos = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # (B_r,)
            key_pos = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (B_c,)
            mask = query_pos[:, None] >= key_pos[None, :]  # (B_r, B_c)
            S_ij = tl.where(mask, S_ij, -float("inf"))
        
        P_ij = tl.exp(S_ij - L_i[:, None])  # (B_r, B_c)
        dV_j = dV_j + tl.dot(tl.trans(P_ij), dO_i)  # (B_c, d)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))  # (B_r, B_c)
        dS_ij = P_ij * (dP_ij - D_i[:, None])  # (B_r, B_c)
        
        # atomic write dQ_i, (b, (Tr Br), d)
        dQ_base_ptr = dQ_ptr + batch_index * stride_qb + i * Q_TILE_SIZE * stride_qq
        q_offsets = tl.arange(0, Q_TILE_SIZE)[:, None] * stride_qq
        d_offsets = tl.arange(0, D)[None, :] * stride_qd
        dQ_ptrs = dQ_base_ptr + q_offsets + d_offsets
        dQ_update = tl.dot(dS_ij, K_j) * scale # (B_r, d)
        tl.atomic_add(dQ_ptrs, dQ_update.to(dQ_ptr.type.element_ty))
        
        dK_j = dK_j + tl.dot(tl.trans(dS_ij), Q_i) * scale  # (B_c, d)
        
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dQ_block_ptr = tl.advance(dQ_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
    
    # Write dK_j, dV_j to HBM
    tl.store(dK_block_ptr, dK_j.to(dK_ptr.type.element_ty), boundary_check=(0,))
    tl.store(dV_block_ptr, dV_j.to(dV_ptr.type.element_ty), boundary_check=(0,))


class TritonFlashAttentionAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert (
            Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous Q, K, V"

        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        ctx.Q_TILE_SIZE = 32  # B_r
        ctx.K_TILE_SIZE = 32  # B_c
        ctx.is_causal = is_causal

        # initialize O, L
        O = torch.zeros_like(Q)
        L = torch.zeros((B, N_QUERIES), device=Q.device, dtype=Q.dtype)

        scale = 1.0 / math.sqrt(D)
        
        # launch kernel
        flash_fwd_kernel[
            (cdiv(N_QUERIES, ctx.Q_TILE_SIZE), B) # (Tq, batch_size)
        ](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal
        )
        
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert (
            Q.is_contiguous()
            and K.is_contiguous()
            and V.is_contiguous()
            and grad_output.is_contiguous()
            and O.is_contiguous()
            and L.is_contiguous()
        ), "Our pointer arithmetic will assume contiguous Q, K, V, dO, O, L"

        B, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        scale = 1.0 / math.sqrt(D)
        D_ = (grad_output * O).sum(dim=-1)
        dQ = torch.zeros_like(Q, dtype=torch.float32)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        
        flash_backward_kernel[
            (cdiv(N_KEYS, ctx.K_TILE_SIZE), B) # (Tq, batch_size)
        ](
            Q, K, V,
            O, L, D_,
            dQ, dK, dV,
            grad_output,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D_.stride(0), D_.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal
        )
        
        return  dQ.to(Q.dtype), dK, dV, None
