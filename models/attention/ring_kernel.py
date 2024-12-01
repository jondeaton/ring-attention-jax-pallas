"""Pallas kernels for on-device ring-attention."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

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

    # load q: it will stay in L1 throughput
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


def _mha_backward_kernel():
    # TODO:...
    ...


def bwd_block(): ...
