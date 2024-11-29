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


@pytest.mark.parametrize(
    "seed,q_len,kv_len,h,d",
    [
        (0, 24, 16, 1, 5),
        (0, 128, 64, 4, 64),
        (1, 512, 512, 8, 16),
        (2, 1024, 512, 8, 128),
    ],
)
def test_ring_attention_forward(seed: int, q_len: int, kv_len: int, h: int, d: int):
    key = jax.random.PRNGKey(seed)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 4

    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], shape=(batch_size, q_len, h, d))
    k = jax.random.normal(keys[1], shape=(batch_size, kv_len, h, d))
    v = jax.random.normal(keys[2], shape=(batch_size, kv_len, h, d))

    expected_output = mha(q, k, v, bias=None)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)

    ring_attention_sharded = shard_map(
        functools.partial(ring_attention, axis_name="sp"),
        mesh=mesh,
        in_specs=(
            PartitionSpec("dp", "sp", None),  # q
            PartitionSpec("dp", "sp", None),  # k
            PartitionSpec("dp", "sp", None),  # v
        ),
        out_specs=PartitionSpec("dp", "sp", None),
        check_rep=False,
    )

    output = ring_attention_sharded(q, k, v)
    assert not jnp.isnan(output).any()
    np.testing.assert_allclose(output, expected_output, rtol=0.01, atol=0.001)


@pytest.mark.parametrize(
    "seed,q_len,kv_len,h,d",
    [
        (0, 24, 16, 1, 5),
        (0, 128, 64, 4, 64),
        (1, 512, 512, 8, 16),
        (2, 1024, 512, 8, 128),
    ],
)
def test_ring_attention_backward(seed: int, q_len: int, kv_len: int, h: int, d: int):
    key = jax.random.PRNGKey(seed)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 4

    q = jax.random.normal(key, shape=(batch_size, q_len, h, d))
    k = jax.random.normal(key, shape=(batch_size, kv_len, h, d))
    v = jax.random.normal(key, shape=(batch_size, kv_len, h, d))

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)

    ring_attention_sharded = shard_map(
        functools.partial(ring_attention, axis_name="sp"),
        mesh=mesh,
        in_specs=(
            PartitionSpec("dp", "sp", None),  # q
            PartitionSpec("dp", "sp", None),  # k
            PartitionSpec("dp", "sp", None),  # v
        ),
        out_specs=PartitionSpec("dp", "sp", None),
        check_rep=False,
    )

    dq, dk, dv = jax.jit(
        jax.grad(
            lambda q, k, v: ring_attention_sharded(q, k, v).sum(),
            argnums=(0, 1, 2),
        )
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
