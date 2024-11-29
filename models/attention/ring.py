"""Ring Attention.

https://github.com/haoliuhl/ringattention/blob/main/ringattention/ringattention_jax.py
"""

from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import math

import functools
import einops

from jaxtyping import Float, Int, Array, PRNGKeyArray, PyTree


def rotate_block(x: PyTree, axis_name: str, axis_size: int) -> PyTree:
    """Rotates an array block (ie key/query block) along the axis.

    Args:
        x: array block to rotate
        axis_name: the name of the axis along which the array is sharded.
        axis_size: number of blocks/shards/slices that cut the axis.
    Returns:
        rotated block same shape as input block.
    """
    return jax.lax.ppermute(
        x,
        axis_name,
        perm=[(i, (i + 1) % axis_size) for i in range(axis_size)],
    )


def _ring_attention_fwd(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    sm_scale: float,
    bias_fn: Callable[..., Float[Array, "b lq lk"]] | None = None,
    bias_fn_q_kwargs: dict[str, PyTree] | None = None,
    bias_fn_kv_kwargs: dict[str, PyTree] | None = None,
) -> tuple[Float[Array, "b lq h dv"], tuple]:
    batch, q_len, num_heads, dk = q.shape
    batch, kv_len, _, _ = k.shape
    dv = v.shape[-1]

    q = einops.rearrange(q, "b l h d -> b h l d")
    k = einops.rearrange(k, "b l h d -> b h l d")
    v = einops.rearrange(v, "b l h d -> b h l d")

    # TODO: get this number without doing a collective op?
    axis_size = jax.lax.psum(1, axis_name)
    rotate = functools.partial(rotate_block, axis_name=axis_name, axis_size=axis_size)

    def scan_fn(carry, i: Int):
        o, l, m_prev, k, v, kv_kwargs = carry

        s = einops.einsum(q, k, "b h lq dk, b h lk dk -> b h lq lk") * sm_scale
        if bias_fn is not None:
            assert bias_fn_q_kwargs is not None
            assert kv_kwargs is not None
            bias = bias_fn(**bias_fn_q_kwargs, **kv_kwargs)
            s += bias

        m = jnp.maximum(m_prev, jnp.max(s, axis=-1))

        correction = jnp.exp(m_prev - m)

        p = jnp.exp(s - m[..., None])
        pv = einops.einsum(p, v, "b h lq lk, b h lk dv -> b h lq dv")

        o_scale = jnp.where(i == 0, jnp.zeros_like(correction), correction)
        o = o * o_scale[..., None] + pv

        # computed for the backwards pass.
        l = correction * l + jnp.sum(p, axis=-1)

        (k, v, kv_kwargs) = rotate([k, v, kv_kwargs])
        return (o, l, m, k, v, kv_kwargs), None

    # Loop state initialization.
    o = jnp.zeros(shape=(batch, num_heads, q_len, dv), dtype=v.dtype)
    l = jnp.zeros(shape=(batch, num_heads, q_len), dtype=q.dtype)
    m = jnp.zeros(shape=(batch, num_heads, q_len), dtype=q.dtype) - float("inf")
    (o, l, m, _, _, _), _ = jax.lax.scan(
        scan_fn,
        init=(o, l, m, k, v, bias_fn_kv_kwargs),
        xs=jnp.arange(axis_size),
    )

    o = o / l[..., None]
    L: Float[Array, "b h lq"] = m + jnp.log(l)
    res = q, k, v, o, L  # for backwards pass
    return einops.rearrange(o, "b h l d -> b l h d"), res


def _ring_attention_bwd(
    axis_name: str,
    sm_scale: float,
    bias_fn: Callable[..., Float[Array, "b lq lk"]] | None,
    bias_fn_q_kwargs: dict[str, Array] | None,
    bias_fn_kv_kwargs: dict[str, Array] | None,
    residuals,
    do,
):
    """Backwards pass for ring attention."""

    q, k, v, o, L = residuals
    do = einops.rearrange(do, "b l h d -> b h l d")

    batch, num_heads, q_len, dim_k = q.shape
    _, _, kv_len, dim_v = v.shape
    assert k.shape == (batch, num_heads, kv_len, dim_k)

    # TODO: get this number without doing a collective op?
    axis_size = jax.lax.psum(1, axis_name)
    rotate = functools.partial(rotate_block, axis_name=axis_name, axis_size=axis_size)

    def scan_fn(carry, i: Int):
        q, o, dq, dk, dv, do, L = carry

        s = einops.einsum(q, k, "b h lq dk, b h lk dk -> b h lq lk") * sm_scale
        if bias_fn is not None:
            assert bias_fn_q_kwargs is not None
            assert bias_fn_kv_kwargs is not None
            bias = bias_fn(**bias_fn_q_kwargs, **bias_fn_kv_kwargs)
            s += bias

        p: Float[Array, "b h lq lk"] = jnp.exp(s - L[..., None])

        dv += einops.einsum(p, do, "b h lq lk, b h lq dv -> b h lk dv")

        dp = einops.einsum(do, v, "b h lq dv, b h lk dv -> b h lq lk")
        delta: Float[Array, "b h lq 1"] = jnp.sum(do * o, keepdims=True, axis=-1)
        ds: Float[Array, "b h lq lk"] = p * (dp - delta) * sm_scale

        dq += einops.einsum(ds, k, "b h lq lk, b h lk dk -> b h lq dk")
        dk += einops.einsum(ds, q, "b h lq lk, b h lq dk -> b h lk dk")

        q, o, dq, do, L = rotate([q, o, dq, do, L])
        return (q, o, dq, dk, dv, do, L), None

    dq = jnp.zeros(shape=(batch, num_heads, q_len, dim_k), dtype=q.dtype)
    dk = jnp.zeros(shape=(batch, num_heads, kv_len, dim_k), dtype=k.dtype)
    dv = jnp.zeros(shape=(batch, num_heads, kv_len, dim_v), dtype=v.dtype)

    (q, o, dq, dk, dv, do, _), _ = jax.lax.scan(
        scan_fn,
        init=(q, o, dq, dk, dv, do, L),
        xs=jnp.arange(axis_size),
    )

    dq = einops.rearrange(dq, "b h l d -> b l h d")
    dk = einops.rearrange(dk, "b h l d -> b l h d")
    dv = einops.rearrange(dv, "b h l d -> b l h d")
    return dq, dk, dv


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _ring_attention(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    sm_scale: float,
    bias_fn: Callable[..., Float[Array, "b lq lk"]] | None,
    bias_fn_q_kwargs: dict[str, Array] | None,
    bias_fn_kv_kwargs: dict[str, Array] | None,
) -> Float[Array, "b lq h dv"]:
    o, _ = _ring_attention_fwd(
        q,
        k,
        v,
        axis_name=axis_name,
        sm_scale=sm_scale,
        bias_fn=bias_fn,
        bias_fn_q_kwargs=bias_fn_q_kwargs,
        bias_fn_kv_kwargs=bias_fn_kv_kwargs,
    )
    return o


_ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)


def ring_attention(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    sm_scale: float | None = None,
    bias_fn: Callable[..., Float[Array, "b lq lk"]] | None = None,
    bias_fn_q_kwargs: dict[str, PyTree] | None = None,
    bias_fn_kv_kwargs: dict[str, PyTree] | None = None,
) -> Float[Array, "b lq h dv"]:
    """Ring attention.

    Args:
        q: single block of queries sharded along the length dimension.
        k: single block of keys sharded along the length dimension.
        v: single block of values sharded along the length dimension.
        axis_name: name of device mesh axis along which sequences are sharded.
        sm_scale: optional softmax scale, defaults to 1/sqrt(dk) if unspecified.
        bias_fn: optional function which computes a block-wise attention bias.
        bias_fn_q_kwargs: optional set of kwargs to pass to bias_fn which are aligned
            with the queries.
        bias_fn_kv_kwargs: optional set of kwargs to pass to bias_fn which are aligned
            with the keys/values.
    Returns:
        Attention output (sharded).
    """
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])

    return _ring_attention(
        q,
        k,
        v,
        axis_name=axis_name,
        sm_scale=sm_scale,
        bias_fn=bias_fn,
        bias_fn_q_kwargs=bias_fn_q_kwargs,
        bias_fn_kv_kwargs=bias_fn_kv_kwargs,
    )


# def ring_attention(
#     q: Float[Array, "b lq h dk"],
#     k: Float[Array, "b lk h dk"],
#     v: Float[Array, "b lk h dv"],
#     axis_name: str,
#     sm_scale: float | None = None,
#     q_segment_ids: Int[Array, "b lq"] | None = None,
#     kv_segment_ids: Int[Array, "b lq"] | None = None,
#     causal: bool = False,
#     q_positions: Int[Array, "b lq"] | None = None,
#     kv_positions: Int[Array, "b lq"] | None = None,
# ) -> Float[Array, "b lq h dv"]:
#     """Ring attention with common parameters."""
#
#     def bias_fn(
#         q_segment_ids: Float[Array, "b lq"] | None,
#         kv_segment_ids: Float[Array, "b lk"] | None,
#         q_positions: Int[Array, "b lq"] | None,
#         k_positions: Int[Array, "b lq"] | None,
#     ) -> Float[Array, "b lq lk"]:
#         """Computes attention bias for single block."""
#
#
#         if q_segment_ids is not None:
#             segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
#
#         if causal:
#
#
#
#
#         return jnp.where(mask, 0, -jnp.inf)
#
#     return ring_attention_flex(
#         q,
#         k,
#         v,
#         axis_name=axis_name,
#         sm_scale=sm_scale,
#         bias_fn=bias_fn,
#         bias_fn_q_kwargs={
#             "q_segment_ids": q_segment_ids,
#             "query_positions": q_positions,
#         },
#         bias_fn_kv_kwargs={
#             "kv_segment_ids": kv_segment_ids,
#             "key_positions": kv_positions,
#         },
#     )
