"""Ring Attention.

https://github.com/haoliuhl/ringattention/blob/main/ringattention/ringattention_jax.py
"""

from __future__ import annotations
from typing import Any

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
        o, l, m_prev, k, v = carry

        s = einops.einsum(q, k, "b h lq dk, b h lk dk -> b h lq lk") * sm_scale
        m = jnp.maximum(m_prev, jnp.max(s, axis=-1))

        correction = jnp.exp(m_prev - m)

        p = jnp.exp(s - m[..., None])
        pv = einops.einsum(p, v, "b h lq lk, b h lk dv -> b h lq dv")

        o_scale = jnp.where(i == 0, jnp.zeros_like(correction), correction)
        o = o * o_scale[..., None] + pv

        # computed for the backwards pass.
        l = correction * l + jnp.sum(p, axis=-1)

        k, v = rotate([k, v])
        return (o, l, m, k, v), None

    # Loop state initialization.
    o = jnp.zeros(shape=(batch, num_heads, q_len, dv), dtype=v.dtype)
    l = jnp.zeros(shape=(batch, num_heads, q_len), dtype=q.dtype)
    m = jnp.zeros(shape=(batch, num_heads, q_len), dtype=q.dtype) - float("inf")
    (o, l, m, _, _), _ = jax.lax.scan(
        scan_fn,
        init=(o, l, m, k, v),
        xs=jnp.arange(axis_size),
    )

    o = o / l[..., None]
    L: Float[Array, "b h lq"] = m + jnp.log(l)
    res = q, k, v, o, L  # for backwards pass
    return einops.rearrange(o, "b h l d -> b l h d"), res


def _ring_attention_bwd(axis_name: str, sm_scale: float, res, do):
    """Backwards pass for ring attention."""

    q, k, v, o, L = res
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


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _ring_attention(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    sm_scale: float,
) -> Float[Array, "b lq h dv"]:
    o, _ = _ring_attention_fwd(q, k, v, axis_name=axis_name, sm_scale=sm_scale)
    return o


_ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)


def ring_attention(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    sm_scale: float | None = None,
) -> Float[Array, "b lq h dv"]:
    """Ring attention.

    Args:
        q: queries.
        k: keys
        v: values
        axis_name: name of device mesh axis along which sequences are sharded.
        sm_scale: optional softmax scale, defaults to 1/sqrt(dk) if unspecified.
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
    )
