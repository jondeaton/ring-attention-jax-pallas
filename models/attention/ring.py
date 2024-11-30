"""Ring Attention with flexible attention.

This is a JAX adaptation of the ring-attention algorithm introduced in [1] inspired by
the original JAX implementation in [2]. The original code was adapted significantly
to more closely resemble the Flash Attention 2 algorithm [3], but where the
loads/stores to/from SRAM/HBM are replaced with rotations of query/key/value around the
ring of devices which the sequences are shadred across.

This implementation also supports a general mechamnism for incorporating arbitrary
attention biases from a user-defined function, similar to Flex Attention [3].

1. ring attention paper
2. ring attention code https://github.com/haoliuhl/ringattention/blob/main/ringattention/ringattention_jax.py
3. flash attention 2 paper
4. flex attention
"""

from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental.shard_map import shard_map

import math

import functools
import einops

from jaxtyping import Float, Int, Bool, Array, PyTree


def _rotate_block(x: PyTree, axis_name: str, axis_size: int) -> PyTree:
    """Rotates an array block (ie query/key block) along the sharding axis.

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
    bias_fn: Callable[..., Float[Array, "b lq lk"]] | None,
    bias_q_kwargs: dict[str, PyTree] | None,
    bias_kv_kwargs: dict[str, PyTree] | None,
) -> tuple[Float[Array, "b lq h dv"], tuple]:
    batch, q_len, num_heads, dk = q.shape
    batch, kv_len, _, _ = k.shape
    dv = v.shape[-1]

    q = einops.rearrange(q, "b l h d -> b h l d")
    k = einops.rearrange(k, "b l h d -> b h l d")
    v = einops.rearrange(v, "b l h d -> b h l d")

    # TODO: get this number without doing a collective op?
    axis_size = jax.lax.psum(1, axis_name)
    rotate = functools.partial(_rotate_block, axis_name=axis_name, axis_size=axis_size)

    def scan_fn(carry, i: Int):
        o, l, m_prev, k, v, kv_kwargs = carry

        s = einops.einsum(q, k, "b h lq dk, b h lk dk -> b h lq lk") * sm_scale
        if bias_fn is not None:
            assert bias_q_kwargs is not None
            assert kv_kwargs is not None
            bias = bias_fn(**bias_q_kwargs, **kv_kwargs)
            s += bias

        m = jnp.maximum(m_prev, jnp.max(s, axis=-1))

        correction = jnp.exp(m_prev - m)
        correction = jnp.where(jnp.isneginf(m) & jnp.isneginf(m_prev), 0, correction)

        p = jnp.exp(s - m[..., None])
        p = jnp.where(jnp.isneginf(m)[..., None], 0, p)  # if no data in this block
        pv = einops.einsum(p, v, "b h lq lk, b h lk dv -> b h lq dv")

        o_scale = jnp.where(i == 0, jnp.zeros_like(correction), correction)
        o = o * o_scale[..., None] + pv

        # computed and saved for the backwards pass.
        l = correction * l + jnp.sum(p, axis=-1)

        k, v, kv_kwargs = rotate([k, v, kv_kwargs])
        return (o, l, m, k, v, kv_kwargs), None

    # Loop state initialization.
    o = jnp.zeros(shape=(batch, num_heads, q_len, dv), dtype=v.dtype)
    l = jnp.zeros(shape=(batch, num_heads, q_len), dtype=q.dtype)
    m = jnp.zeros(shape=(batch, num_heads, q_len), dtype=q.dtype) - float("inf")

    (o, l, m, k, v, _), _ = jax.lax.scan(
        scan_fn,
        init=(o, l, m, k, v, bias_kv_kwargs),
        xs=jnp.arange(axis_size),
    )

    o = o / l[..., None]
    L: Float[Array, "b h lq"] = m + jnp.log(l)
    res = q, k, v, o, L, bias_q_kwargs, bias_kv_kwargs  # for backwards pass
    return einops.rearrange(o, "b h l d -> b l h d"), res


def _ring_attention_bwd(
    axis_name: str,
    sm_scale: float,
    bias_fn: Callable[..., Float[Array, "b h lq lk"]] | None,
    residuals,
    do,
):
    """Backwards pass for ring attention."""

    q, k, v, o, L, bias_q_kwargs, bias_kv_kwargs = residuals
    do = einops.rearrange(do, "b l h d -> b h l d")

    batch, num_heads, q_len, dim_k = q.shape
    _, _, kv_len, dim_v = v.shape
    assert k.shape == (batch, num_heads, kv_len, dim_k)

    # TODO: get this number without doing a collective op?
    axis_size = jax.lax.psum(1, axis_name)
    rotate = functools.partial(_rotate_block, axis_name=axis_name, axis_size=axis_size)

    def scan_fn(carry, i: Int):
        q, o, dq, dk, dv, do, L, bias_q_kwargs = carry

        s = einops.einsum(q, k, "b h lq dk, b h lk dk -> b h lq lk") * sm_scale
        if bias_fn is not None:
            assert bias_q_kwargs is not None
            assert bias_kv_kwargs is not None
            bias = bias_fn(**bias_q_kwargs, **bias_kv_kwargs)
            s += bias

        p: Float[Array, "b h lq lk"] = jnp.exp(s - L[..., None])

        dv += einops.einsum(p, do, "b h lq lk, b h lq dv -> b h lk dv")

        dp = einops.einsum(do, v, "b h lq dv, b h lk dv -> b h lq lk")
        delta: Float[Array, "b h lq 1"] = jnp.sum(do * o, keepdims=True, axis=-1)
        ds: Float[Array, "b h lq lk"] = p * (dp - delta) * sm_scale

        dq += einops.einsum(ds, k, "b h lq lk, b h lk dk -> b h lq dk")
        dk += einops.einsum(ds, q, "b h lq lk, b h lq dk -> b h lk dk")

        q, o, dq, do, L, bias_q_kwargs = rotate([q, o, dq, do, L, bias_q_kwargs])
        return (q, o, dq, dk, dv, do, L, bias_q_kwargs), None

    dq = jnp.zeros(shape=(batch, num_heads, q_len, dim_k), dtype=q.dtype)
    dk = jnp.zeros(shape=(batch, num_heads, kv_len, dim_k), dtype=k.dtype)
    dv = jnp.zeros(shape=(batch, num_heads, kv_len, dim_v), dtype=v.dtype)

    (q, o, dq, dk, dv, do, _, _), _ = jax.lax.scan(
        scan_fn,
        init=(q, o, dq, dk, dv, do, L, bias_q_kwargs),
        xs=jnp.arange(axis_size),
    )

    dq = einops.rearrange(dq, "b h l d -> b l h d")
    dk = einops.rearrange(dk, "b h l d -> b l h d")
    dv = einops.rearrange(dv, "b h l d -> b l h d")
    return dq, dk, dv, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _ring_attention(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    sm_scale: float,
    bias_fn: Callable[..., Float[Array, "b h lq lk"]] | None,
    bias_q_kwargs: dict[str, PyTree],
    bias_kv_kwargs: dict[str, PyTree],
) -> Float[Array, "b lq h dv"]:
    """Ring attention implementation."""
    o, _ = _ring_attention_fwd(
        q,
        k,
        v,
        axis_name=axis_name,
        sm_scale=sm_scale,
        bias_fn=bias_fn,
        bias_q_kwargs=bias_q_kwargs,
        bias_kv_kwargs=bias_kv_kwargs,
    )
    return o


_ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)


def ring_attention(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    axis_name: str,
    bias_fn: Callable[..., Float[Array, "b h lq lk"]] | None = None,
    bias_q_kwargs: dict[str, PyTree] | None = None,
    bias_kv_kwargs: dict[str, PyTree] | None = None,
    sm_scale: float | None = None,
) -> Float[Array, "b lq h dv"]:
    """Ring attention - general.

    This generalized ring-attention function enables integration of arbitrary attention
    bias computed block-wise. It presents a similar as PyTorch's Flex Attention
    (https://pytorch.org/blog/flexattention/):

    The user-defined bias_fn computes a single block of attention-bias given sharded
    arrays in bias_(q|kv)_kwargs, for example segment-ids for block-sparse attention.
    The sharded arrays in bias_fn kwargs rotate along the device ring in sync
    with queries or keys, must be given to bias_q_kwargs/bias_kv_kwargs, respectively.

    See the implementaiton of ring_self_attention as an example of block-sparse causal
    or prefixlm attention.

    Args:
        q: single block of queries sharded along the length dimension.
        k: single block of keys sharded along the length dimension.
        v: single block of values sharded along the length dimension.
        axis_name: name of device mesh axis along which sequences are sharded.
        sm_scale: optional softmax scale, defaults to 1/sqrt(dk) if unspecified.
        bias_fn: optional function which computes a block-wise attention bias.
        bias_q_kwargs: kwargs to bias_fn with query-aligned sharded arrays. For example
            segment_ids for the query sequence.
        bias_kv_kwargs: kwargs to bias_fn with key/value-aligned sharded arrays. For
            example segment_ids for the key/value sequence.
    Returns:
        Attention output (sharded).
    """
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(q.shape[-1])

    if bias_q_kwargs is None:
        bias_q_kwargs = {}

    if bias_kv_kwargs is None:
        bias_kv_kwargs = {}

    return _ring_attention(
        q,
        k,
        v,
        axis_name=axis_name,
        sm_scale=sm_scale,
        bias_fn=bias_fn,
        bias_q_kwargs=bias_q_kwargs,
        bias_kv_kwargs=bias_kv_kwargs,
    )


def ring_self_attention(
    q: Float[jax.Array, "b l h dk"],
    k: Float[jax.Array, "b l h dk"],
    v: Float[jax.Array, "b l h dv"],
    mesh: jax.sharding.Mesh,
    pspec: PartitionSpec,
    sm_scale: float | None = None,
    causal: bool = False,
    segment_ids: Int[Array, "b l"] | None = None,
    positions: Int[Array, "b l"] | None = None,
    prefix_mask: Bool[Array, "b l"] | None = None,
) -> Float[Array, "b l h dv"]:
    """Ring attention for self-attention.

    Supports several variants (and combinations):
        - full bidirectional attention (default)
        - block-sparse attention via segment_ids
        - causal attention (requires positions)
        - prefixlm attention via prefix_mask (requires positions)

    This "full-service" implementation also wraps the general ring attention function
    wish shard_map so requires `mesh` and `pspec` arguments. Thus it also serves as an
    exmaple of how to use the general single-shard ring_attention function.

    Args:
        q: full query array sharded along the length axis.
        k: full key array sharded along the length axis.
        v: full values array sharded along the length axis.
        mesh: device mesh across which q/k/v are sharded.
        pspec: partition spec describing the sharding of all arrays.
        sm_scale: optional softmax scale, defaults to 1/sqrt(dk) if unspecified.
        causal: whether to use causal self attention
        segment_ids: for block-sparse self-attention eg. for packed-sample attention.
        positions: for causal attention, indicates the position of each token.
        prefix_mask: for prefixlm, indicates which positions in the prefix.
    Returns:
        single block of output sharded along the length dimension.
    """
    if causal:
        assert positions is not None, "positions are requried for causal attention"
        assert prefix_mask is None, "causal attention does not support prefix_mask"

    if prefix_mask is not None:
        assert positions is not None, "positions are requried for prefixlm attention"
        assert not causal

    def bias_fn(
        q_segment_ids: Int[Array, "b l"] | None,
        kv_segment_ids: Int[Array, "b l"] | None,
        q_positions: Int[Array, "b l"] | None,
        kv_positions: Int[Array, "b l"] | None,
        q_prefix_mask: Bool[Array, "b l"] | None,
        kv_prefix_mask: Bool[Array, "b l"] | None,
    ) -> Float[Array, "b 1 l l"]:
        """block-wise attention bias function."""

        mask = jnp.array(True)

        if q_segment_ids is not None:
            assert kv_segment_ids is not None
            segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
            mask &= segment_mask

        if causal:
            assert q_positions is not None
            assert kv_positions is not None
            causal_mask = kv_positions[:, None, :] <= q_positions[:, :, None]
            mask &= causal_mask

        elif q_prefix_mask is not None:
            assert kv_prefix_mask is not None
            assert q_positions is not None
            assert kv_positions is not None

            prefix_mask = q_prefix_mask[:, :, None] & kv_prefix_mask[:, None, :]
            causal_mask = kv_positions[:, None, :] <= q_positions[:, :, None]
            mask &= prefix_mask | causal_mask

        bias = jnp.where(mask, 0, -jnp.inf)
        return bias[:, None, :, :]  # add singleton heads dimension

    bias_q_kwargs = {
        "q_segment_ids": segment_ids,
        "q_positions": positions,
        "q_prefix_mask": prefix_mask,
    }
    bias_kv_kwargs = {
        "kv_segment_ids": segment_ids,
        "kv_positions": positions,
        "kv_prefix_mask": prefix_mask,
    }

    def _ring_attn_fn(q, k, v, bias_q_kwargs, bias_kv_kwargs):
        """binds static args to ring_attention."""
        # TODO: why can we not just bind the static args with functools.partial?
        return ring_attention(
            q,
            k,
            v,
            axis_name=pspec[1],
            sm_scale=sm_scale,
            bias_fn=bias_fn,
            bias_q_kwargs=bias_q_kwargs,
            bias_kv_kwargs=bias_kv_kwargs,
        )

    return shard_map(
        _ring_attn_fn,
        mesh=mesh,
        in_specs=(
            pspec,  # q
            pspec,  # k
            pspec,  # v
            pspec,  # bias_q_kwargs
            pspec,  # bias_kv_kwargs
        ),
        out_specs=pspec,  # type: ignore
        check_rep=False,
    )(q, k, v, bias_q_kwargs, bias_kv_kwargs)
