"""Pallas kernels for on-device ring-attention.

This code was adapted from the JAX project (https://github.com/google/jax) in
particular:
https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/attention.py

Significant modifications have been made:
    1. support for arbitrary attention bias
    2. output block-wise stats and change interface for use in ring attention.

The original copyright:
"""

# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from jaxtyping import Float, Array


def _mha_forward_kernel(
    q_ref,  # inputs
    k_ref,
    v_ref,
    bias_ref,
    o_ref: Any,
    m_ref: Any,
    l_ref: Any,
    o_out_ref: Any,  # outputs
    m_out_ref: Any,
    l_out_ref: Any,
    num_heads: int,
    sm_scale: float,
    block_q: int,
    block_d: int,
    block_k: int,
):
    q_len = q_ref.shape[0]
    kv_len = k_ref.shape[0]

    start_q = pl.program_id(0)
    curr_q_slice = pl.dslice(start_q * block_q, block_q)

    # initialize o, l, m from incoming stats
    o = o_ref[...]
    m_i = m_ref[...]
    l_i = l_ref[...]
    q = q_ref[...]

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

        correction = jnp.where(
            jnp.isneginf(m_curr) & jnp.isneginf(m_prev), 0, correction
        )

        l_prev_corr = correction * l_prev
        s_curr = jnp.exp(
            qk - m_next[:, None]
        )  # Use m_next instead of m_curr to avoid a correction on l_curr

        s_curr = jnp.where(
            jnp.isneginf(m_next)[..., None], 0, s_curr
        )  # if no data in this block

        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        o_curr = pl.dot(s_curr.astype(v.dtype), v)

        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    num_kv_blocks = pl.cdiv(kv_len, block_k)
    o, m_i, l_i = jax.lax.fori_loop(0, num_kv_blocks, body, (o, m_i, l_i))

    # Write outputs to dram.
    o_out_ref[...] = o.astype(o_ref.dtype)
    m_out_ref[...] = m_i.astype(m_out_ref.dtype)
    l_out_ref[...] = l_i.astype(l_out_ref.dtype)


def fwd_block(
    q: Float[Array, "b h lq dk"],
    k: Float[Array, "b h lk dk"],
    v: Float[Array, "b h lk dv"],
    bias: Float[Array, "b h lq lk"] | None,
    o: Float[Array, "b h lq dv"],
    m: Float[Array, "b h lq"],
    l: Float[Array, "b h lq"],
    sm_scale: float,
    block_q: int = 128,
    block_k: int = 128,
    debug: bool = False,
    interpret: bool = True,
) -> tuple[
    Float[Array, "b h lq dv"],
    Float[Array, "b h lq"],
    Float[Array, "b h lq"],
]:
    batch_size, num_heads, q_len, dim_k = q.shape
    _, _, k_len, dim_v = v.shape

    assert q_len % block_q == 0, (q_len, block_q)
    assert k_len % block_k == 0, (k_len, block_k)

    kernel = functools.partial(
        _mha_forward_kernel,
        num_heads=num_heads,
        sm_scale=sm_scale,
        block_q=block_q,
        block_k=block_k,
        block_d=dim_k,
    )

    return pl.pallas_call(
        kernel,
        grid=(batch_size, num_heads, pl.cdiv(q_len, block_q)),
        in_specs=[
            pl.BlockSpec((None, None, block_q, dim_k), lambda b, h, lq: (b, h, lq, 0)),
            pl.BlockSpec((None, None, k_len, dim_k), lambda b, h, _: (b, h, 0, 0)),
            pl.BlockSpec((None, None, k_len, dim_v), lambda b, h, _: (b, h, 0, 0)),
            (
                pl.BlockSpec(
                    (None, None, block_q, k_len), lambda b, h, lq: (b, h, lq, 0)
                )
                if bias is not None
                else None
            ),
            pl.BlockSpec((None, None, block_q, dim_k), lambda b, h, lq: (b, h, lq, 0)),
            pl.BlockSpec((None, None, block_q), lambda b, h, lq: (b, h, lq)),  # m
            pl.BlockSpec((None, None, block_q), lambda b, h, lq: (b, h, lq)),  # l
        ],
        out_specs=[
            pl.BlockSpec((None, None, block_q, dim_k), lambda b, h, lq: (b, h, lq, 0)),
            pl.BlockSpec((None, None, block_q), lambda b, h, lq: (b, h, lq)),
            pl.BlockSpec((None, None, block_q), lambda b, h, lq: (b, h, lq)),
        ],
        compiler_params=dict(
            triton=dict(
                num_warps=4 if dim_k <= 64 else 8,
                num_stages=2,
            )
        ),
        out_shape=[
            jax.ShapeDtypeStruct(  # out
                shape=(batch_size, num_heads, q_len, dim_v), dtype=q.dtype
            ),
            jax.ShapeDtypeStruct(  # m
                shape=(batch_size, num_heads, q_len), dtype=jnp.float32
            ),
            jax.ShapeDtypeStruct(  # l
                shape=(batch_size, num_heads, q_len), dtype=jnp.float32
            ),
        ],
        debug=debug,
        interpret=interpret,
        name="forward_block",
    )(q, k, v, bias, o, m, l)


def bwd_block(
    q: Float[Array, "b h lq dk"],
    k: Float[Array, "b h lk dk"],
    v: Float[Array, "b h lk dv"],
    bias: Float[Array, "b h lq lk"] | None,
    o: Float[Array, "b h lq dv"],
    L: Float[Array, "b h lq"],
    do: Float[Array, "b h lq dv"],
    sm_scale: float,
    block_q: int = 128,
    block_k: int = 128,
    debug: bool = False,
    interpret: bool = True,
):
    """Computes vjp for single block."""
    batch_size, num_heads, q_len, dim_k = q.shape
    _, _, k_len, dim_v = v.shape

    assert q_len % block_q == 0, (q_len, block_q)
    assert k_len % block_k == 0, (k_len, block_k)

    delta = _precompute_delta(o, do, L, block_q, debug, interpret)

    dq, dk, dv = pl.pallas_call(
        functools.partial(
            _bwd_kernel,
            sm_scale=sm_scale,
            block_q1=block_q,
            block_k1=block_k,
            block_q2=block_q,
            block_k2=block_k,
            block_d=dim_k,
        ),
        # TODO: does this grid order have performance implications?
        grid=(batch_size, num_heads, pl.cdiv(k_len, block_k)),
        in_specs=[
            pl.BlockSpec(  # q
                (None, None, q_len, dim_k), lambda b, h, lk: (b, h, 0, 0)
            ),
            pl.BlockSpec(  # k
                (None, None, k_len, dim_k), lambda b, h, lk: (b, h, 0, 0)
            ),
            pl.BlockSpec(  # v
                (None, None, k_len, dim_k), lambda b, h, lk: (b, h, 0, 0)
            ),
            (  # bias
                pl.BlockSpec((None, None, q_len, k_len), lambda b, h, lk: (b, h, 0, lk))
                if bias is not None
                else None
            ),
            pl.BlockSpec((None, None, q_len, dim_k), lambda b, h, _: (b, h, 0, 0)),  # o
            pl.BlockSpec((None, None, q_len), lambda b, h, _: (b, h, 0)),  # lse
            pl.BlockSpec(  # do
                (None, None, q_len, dim_k), lambda b, h, _: (b, h, 0, 0)
            ),
            pl.BlockSpec((None, None, q_len), lambda i, j, _: (i, j, 0)),  # delta
        ],
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ],
        out_specs=[
            pl.BlockSpec(
                (None, None, block_q, dim_k),
                lambda i, j, k: (i, j, k, 0),  # dq
            ),
            pl.BlockSpec(
                (None, None, block_k, dim_k),
                lambda i, j, k: (i, j, k, 0),  # dk
            ),
            pl.BlockSpec(
                (None, None, block_k, dim_k),
                lambda i, j, k: (i, j, k, 0),  # dv
            ),
        ],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        compiler_params=dict(triton=dict(num_warps=8, num_stages=2)),
    )(q, k, v, bias, o, L, do, delta)

    return dq.astype(q.dtype), dk, dv


# This kernel computes dK_i, dV_i and dQ_i in parallel across the sequence
# length.
# Inspired by the triton tutorial: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
def _bwd_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    bias_ref,
    out_ref,
    lse_ref,
    do_scaled_ref,
    delta_ref,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    block_q1: int,
    block_k1: int,
    block_q2: int,
    block_k2: int,
    block_d: int,
):
    del out_ref  # Not needed
    q_len = q_ref.shape[0]
    k_len = k_ref.shape[0]

    # Scan #1: dK and dV
    #   1. Load a block of K and V of size (block_k1, head_dim) in SMEM.
    #   2. Iterate through Q in chunks of (block_q1, head_dim) to accumulate
    #      dK and dV.
    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_k1, block_k1)

    dv = jnp.zeros([block_k1, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_k1, block_d], dtype=jnp.float32)

    v = pl.load(v_ref, (curr_k_slice, slice(None)))
    k = pl.load(k_ref, (curr_k_slice, slice(None)))

    def inner_loop_dkdv(start_q, carry):
        dv, dk = carry
        curr_q_slice = pl.dslice(start_q * block_q1, block_q1)

        q = pl.load(q_ref, (curr_q_slice, slice(None)))
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale

        if bias_ref is not None:
            bias = pl.load(bias_ref, (curr_q_slice, curr_k_slice))
            qk += bias

        lse = pl.load(lse_ref, (curr_q_slice,))
        di = pl.load(delta_ref, (curr_q_slice,))
        do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))

        p = jnp.exp(qk - lse[:, None])
        dv = dv + pl.dot(p.astype(do.dtype).T, do)
        dp = jnp.zeros((block_q1, block_k1), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)

        return dv, dk

    dv, dk = jax.lax.fori_loop(0, pl.cdiv(q_len, block_q1), inner_loop_dkdv, (dv, dk))
    dv_ref[...] = dv.astype(dv_ref.dtype)
    dk_ref[...] = dk.astype(dk_ref.dtype)

    del dv, dk

    # Scan #2: dQ
    #   1. Load a block of Q of size (block_q2, head_dim) in SMEM.
    #   2. Iterate through K and V in chunks of (block_k2, head_dim) to
    #     accumulate dQ.
    start_q = pl.program_id(2)
    curr_q_slice = pl.ds(start_q * block_q2, block_q2)
    span_q = start_q * block_q2 + jnp.arange(block_q2)
    dq = jnp.zeros([block_q2, block_d], dtype=jnp.float32)

    q = pl.load(q_ref, (curr_q_slice, slice(None)))
    lse = pl.load(lse_ref, (curr_q_slice,))
    do = pl.load(do_scaled_ref, (curr_q_slice, slice(None)))
    di = pl.load(delta_ref, (curr_q_slice,))

    def inner_loop_dq(start_k, dq):
        curr_k_slice = pl.dslice(start_k * block_k2, block_k2)
        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        v = pl.load(v_ref, (curr_k_slice, slice(None)))

        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale

        if bias_ref is not None:
            bias = pl.load(bias_ref, (curr_q_slice, curr_k_slice))
            qk += bias

        p = jnp.exp(qk - lse[:, None])
        dp = jnp.zeros((block_q2, block_k2), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale

        dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)

        return dq

    upper_bound = pl.cdiv(k_len, block_k2)

    dq = jax.lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
    dq_ref[...] = dq.astype(dq_ref.dtype)


def _precompute_delta(out, do, lse, block_q: int, debug: bool, interpret: bool):
    batch_size, num_heads, seq_len, head_dim = out.shape

    def kernel(out_ref, dout_ref, delta_ref):
        o = out_ref[...].astype(jnp.float32)
        do = dout_ref[...].astype(jnp.float32)
        delta = jnp.sum(o * do, axis=1)
        delta_ref[...] = delta.astype(delta_ref.dtype)

    return pl.pallas_call(
        kernel,
        grid=(batch_size, num_heads, pl.cdiv(seq_len, block_q)),
        in_specs=[
            pl.BlockSpec((None, None, block_q, head_dim), lambda b, h, l: (b, h, l, 0)),
            pl.BlockSpec((None, None, block_q, head_dim), lambda b, h, l: (b, h, l, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, block_q), lambda b, h, l: (b, h, l)),
        compiler_params=dict(triton=dict(num_warps=4, num_stages=3)),
        out_shape=jax.ShapeDtypeStruct(lse.shape, lse.dtype),
        debug=debug,
        interpret=interpret,
        name="precompute_delta",
    )(out, do)
