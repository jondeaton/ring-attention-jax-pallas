"""Test for ring attention."""

import pytest
import os
import functools
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

import einops
from jaxtyping import Float, Int, Array, PRNGKeyArray

from models.attention.ring import ring_attention

# from models.attention.ringattention_jax import ring_attention

flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = flags + " --xla_force_host_platform_device_count=8"


# Reference implementations.
def mha(
    q: Float[Array, "B Lq H Dk"],
    k: Float[Array, "B Lk H Dk"],
    v: Float[Array, "B Lk H Dv"],
    bias: Float[Array, "B H Lq Lk"] | None = None,
) -> Float[Array, "B Lq H Dv"]:
    """Batched multi-head attention."""

    B, Lq, H, Dk = q.shape
    _, Lk, *_ = k.shape

    qk = einops.einsum(q, k, "b i h d, b j h d -> b h i j")
    z = qk / jnp.sqrt(Dk)
    if bias is not None:
        z += bias
    a = jax.nn.softmax(z, axis=-1)
    return einops.einsum(a, v, "b h i j, b j h d -> b i h d")


def test_ring_attention_forward():
    key = jax.random.PRNGKey(0)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 4
    length = 1024
    h = 4
    d = 64

    q = jax.random.normal(key, shape=(batch_size, length, h, d))
    k = jax.random.normal(key, shape=(batch_size, length, h, d))
    v = jax.random.normal(key, shape=(batch_size, length, h, d))
    attn_bias = jax.random.normal(key, shape=(batch_size, h, length, length))
    # segment_ids = einops.repeat(jnp.arange(length) // 128, "l -> b l", b=batch_size)
    segment_ids = einops.repeat(jnp.ones(length), "l -> b l", b=batch_size)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    attn_bias = jax.device_put(
        attn_bias,
        NamedSharding(mesh, PartitionSpec(None, None, "sp", None)),
    )
    segment_ids = jax.device_put(
        segment_ids,
        NamedSharding(mesh, PartitionSpec(None, "sp")),
    )

    chunk_size = 32

    ring_attention_sharded = shard_map(
        functools.partial(ring_attention, axis_name="sp"),
        mesh=mesh,
        in_specs=(
            PartitionSpec("dp", "sp", None),  # q
            PartitionSpec("dp", "sp", None),  # k
            PartitionSpec("dp", "sp", None),  # v
            # PartitionSpec("dp", None, "sp", None),  # attn attn_bias
            # PartitionSpec("dp", "sp"),  # segment_ids
        ),
        out_specs=PartitionSpec("dp", "sp", None),
        check_rep=False,
    )

    with jax.disable_jit():
        output = ring_attention_sharded(q, k, v)
        # output = ring_attention_sharded(q, k, v, attn_bias, segment_ids)

    assert not jnp.isnan(output).any()

    # seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
    # seg_mask = einops.repeat(seg_mask, "B Lq Lk -> B H Lq Lk", H=h)
    # bias = attn_bias + jnp.where(seg_mask, 0, -jnp.inf)
    bias = None

    reference = mha(q, k, v, bias=bias)

    # TODO: next thing to try to solve this:
    # why can't we just replace the inner attention with a simple attention?
    # completely get rid of attention bias?

    np.testing.assert_allclose(output, reference, rtol=0.01)


def test_ring_attention_backward():
    key = jax.random.PRNGKey(0)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 4
    length = 1024
    h = 4
    d = 64

    q = jax.random.normal(key, shape=(batch_size, length, h, d))
    k = jax.random.normal(key, shape=(batch_size, length, h, d))
    v = jax.random.normal(key, shape=(batch_size, length, h, d))
    attn_bias = jax.random.normal(key, shape=(batch_size, h, length, length))
    # segment_ids = einops.repeat(jnp.arange(length) // 128, "l -> b l", b=batch_size)
    segment_ids = einops.repeat(jnp.ones(length), "l -> b l", b=batch_size)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    attn_bias = jax.device_put(
        attn_bias,
        NamedSharding(mesh, PartitionSpec(None, None, "sp", None)),
    )
    segment_ids = jax.device_put(
        segment_ids,
        NamedSharding(mesh, PartitionSpec(None, "sp")),
    )

    chunk_size = 32

    ring_attention_sharded = shard_map(
        functools.partial(ring_attention, axis_name="sp"),
        mesh=mesh,
        in_specs=(
            PartitionSpec("dp", "sp", None),  # q
            PartitionSpec("dp", "sp", None),  # k
            PartitionSpec("dp", "sp", None),  # v
            # PartitionSpec("dp", None, "sp", None),  # attn attn_bias
            # PartitionSpec("dp", "sp"),  # segment_ids
        ),
        out_specs=PartitionSpec("dp", "sp", None),
        check_rep=False,
    )

    with jax.disable_jit():
        dq, dk, dv = jax.grad(
            lambda q, k, v: ring_attention_sharded(q, k, v).sum(),
            argnums=(0, 1, 2),
        )(q, k, v)

    assert not jnp.isnan(dq).any()
    assert not jnp.isnan(dk).any()
    assert not jnp.isnan(dv).any()

    dq_, dk_, dv_ = jax.grad(
        lambda q, k, v: mha(q, k, v).sum(),
        argnums=(0, 1, 2),
    )(q, k, v)

    np.testing.assert_allclose(dq, dq_, atol=1e-4)
    np.testing.assert_allclose(dk, dk_, atol=1e-4)
    np.testing.assert_allclose(dv, dv_, atol=1e-4)
