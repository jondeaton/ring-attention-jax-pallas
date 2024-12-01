"""Pallas kernels for on-device ring-attention."""

from __future__ import annotations

import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np


def _mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,  # Input arrays
    bias_ref,
    segment_ids_ref: jax.Array | None,  # segment_id arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    num_heads: int,
    sm_scale: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)

    # o is the buffer where we accumulate the output on sram.
    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    # acc is the buffer where we accumulate the output on sram.
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load q: it will stay in L1 throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # q tile has shape [block_q, block_d], block_d == head_dim.
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = q_ref[...]
    q_segment_ids = (
        None if segment_ids_ref is None else pl.load(segment_ids_ref, (curr_q_slice,))
    )

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        qk = pl.dot(q, k.T)  # [block_q, block_k]
        if sm_scale != 1.0:
            qk *= sm_scale  # [block_q, block_k]

        # Avoids Triton crash.
        # if num_heads > 2:
        #   qk = qk.astype(q_ref.dtype)
        #   qk = qk.astype(jnp.float32)

        if bias_ref is not None:
            bias = pl.load(bias_ref, (slice(None), curr_k_slice))
            qk += bias

        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp(m_prev - m_next)

        # correction = jnp.where(
        #     jnp.isneginf(m_curr) & jnp.isneginf(m_prev), 0, correction
        # )

        l_prev_corr = correction * l_prev
        s_curr = jnp.exp(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr

        # s_curr = jnp.where(
        #     jnp.isneginf(m_next)[..., None], 0, s_curr
        # )  # if no data in this block

        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        o_curr = pl.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    # We keep an unscaled version of o during the scan over seq_len. Scaling it
    # by the last l_i gives us the correct final output. See section 3.1.1 in the
    # FlashAttention-2 paper: https://arxiv.org/pdf/2307.08691.
    o /= l_i[:, None]

    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log(l_i)
    # Write output to dram.
    o_ref[...] = o.astype(o_ref.dtype)


def mha_forward_block(
    q,
    k,
    v,
    bias,
    sm_scale: float,
    block_q: int,
    block_k: int,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    # Heuristics.
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    kernel = functools.partial(
        _mha_forward_kernel,
        num_heads=num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
    )
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),  # out
        jax.ShapeDtypeStruct(
            shape=(batch_size, num_heads, q_seq_len),
            dtype=jnp.float32,  # lse
        ),
    ]
    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        (
            pl.BlockSpec(
                (None, None, block_q, kv_seq_len), lambda i, j, k: (j, k, i, 0)
            )
            if bias is not None
            else None
        ),
    ]
    in_specs.append(
        None  # type: ignore[arg-type]
        if segment_ids is None
        else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
    )
    out, lse = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        ],
        compiler_params=dict(triton=dict(num_warps=num_warps_, num_stages=num_stages)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, bias, segment_ids)
    return out, (q, k, v, bias, segment_ids, out, lse)
