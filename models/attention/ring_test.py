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
from jaxtyping import Float, Int, Array

from models.attention.ring import ring_attention, ring_self_attention

flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = flags + " --xla_force_host_platform_device_count=8"

jax.config.update("jax_traceback_filtering", "off")


# Reference implementations.
def mha(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    bias: Float[Array, "b h lq lk"] | None = None,
) -> Float[Array, "b lq h d"]:
    """Batched multi-head attention."""

    B, Lq, H, Dk = q.shape
    _, Lk, *_ = k.shape

    qk = einops.einsum(q, k, "b i h d, b j h d -> b h i j")
    z = qk / jnp.sqrt(Dk)
    if bias is not None:
        z += bias
    a = jax.nn.softmax(z, axis=-1)
    return einops.einsum(a, v, "b h i j, b j h d -> b i h d")


@pytest.mark.parametrize("impl", ["jax", "pallas"])
@pytest.mark.parametrize(
    "seed,q_len,kv_len,h,d",
    [
        (0, 24, 16, 1, 2),
        (1, 128, 64, 4, 64),
        (2, 64, 512, 4, 16),
        (3, 2048, 2048, 4, 128),
    ],
)
def test_ring_attention_forward(
    impl: str, seed: int, q_len: int, kv_len: int, h: int, d: int
):
    key = jax.random.PRNGKey(seed)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 2

    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], shape=(batch_size, q_len, h, d))
    k = jax.random.normal(keys[1], shape=(batch_size, kv_len, h, d))
    v = jax.random.normal(keys[2], shape=(batch_size, kv_len, h, d))

    expected_output = mha(q, k, v, bias=None)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)

    ring_attention_sharded = jax.jit(
        shard_map(
            functools.partial(ring_attention, axis_name="sp", block_impl=impl),
            mesh=mesh,
            in_specs=(
                PartitionSpec("dp", "sp", None),  # q
                PartitionSpec("dp", "sp", None),  # k
                PartitionSpec("dp", "sp", None),  # v
            ),
            out_specs=PartitionSpec("dp", "sp", None),
            check_rep=False,
        )
    )

    output = ring_attention_sharded(q, k, v)
    assert not jnp.isnan(output).any()
    np.testing.assert_allclose(output, expected_output, rtol=0.01, atol=0.001)


@pytest.mark.parametrize("impl", ["jax", "pallas"])
@pytest.mark.parametrize(
    "seed,q_len,kv_len,h,d",
    [
        (0, 24, 16, 1, 2),
        (1, 128, 64, 4, 64),
        (2, 64, 512, 4, 16),
        (3, 2048, 2048, 4, 128),
    ],
)
def test_ring_attention_backward(
    impl: str, seed: int, q_len: int, kv_len: int, h: int, d: int
):
    key = jax.random.PRNGKey(seed)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 2

    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], shape=(batch_size, q_len, h, d))
    k = jax.random.normal(keys[1], shape=(batch_size, kv_len, h, d))
    v = jax.random.normal(keys[2], shape=(batch_size, kv_len, h, d))

    do = jax.random.normal(keys[3], shape=(batch_size, q_len, h, d))

    # expected outputs.
    dq_, dk_, dv_ = jax.grad(
        lambda q, k, v: jnp.sum(do * mha(q, k, v)),
        argnums=(0, 1, 2),
    )(q, k, v)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)

    ring_attention_sharded = shard_map(
        functools.partial(ring_attention, axis_name="sp", block_impl=impl),
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
            lambda q, k, v: jnp.sum(do * ring_attention_sharded(q, k, v)),
            argnums=(0, 1, 2),
        )
    )(q, k, v)

    assert not jnp.isnan(dq).any()
    assert not jnp.isnan(dk).any()
    assert not jnp.isnan(dv).any()

    np.testing.assert_allclose(dq, dq_, atol=1e-4)
    np.testing.assert_allclose(dk, dk_, atol=1e-4)
    np.testing.assert_allclose(dv, dv_, atol=1e-4)


@pytest.mark.parametrize("impl", ["jax", "pallas"])
@pytest.mark.parametrize(
    "seed,q_len,kv_len,h,d",
    [
        (0, 24, 16, 1, 2),
        (1, 128, 64, 4, 64),
        (2, 64, 512, 4, 16),
        (3, 2048, 2048, 4, 128),
    ],
)
def test_ring_attention_bias(
    impl: str, seed: int, q_len: int, kv_len: int, h: int, d: int
):
    key = jax.random.PRNGKey(seed)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 2

    keys = jax.random.split(key, 7)
    q = jax.random.normal(keys[0], shape=(batch_size, q_len, h, d))
    k = jax.random.normal(keys[1], shape=(batch_size, kv_len, h, d))
    v = jax.random.normal(keys[2], shape=(batch_size, kv_len, h, d))

    q_segment_ids = jax.random.randint(
        keys[3], shape=(batch_size, q_len), minval=0, maxval=q_len // 5
    )
    kv_segment_ids = jax.random.randint(
        keys[4], shape=(batch_size, kv_len), minval=0, maxval=q_len // 5
    )

    do = jax.random.normal(keys[5], shape=(batch_size, q_len, h, d))

    # bias = jax.random.normal(keys[6], shape=(batch_size, h, q_len, kv_len))
    attn_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
    bias = jnp.where(attn_mask, 0, -jnp.inf)
    bias = einops.repeat(bias, "b lq lk -> b h lq lk", h=h)
    expected_output = mha(q, k, v, bias=bias)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    q_segment_ids = jax.device_put(q_segment_ids, sharding)
    kv_segment_ids = jax.device_put(kv_segment_ids, sharding)

    def bias_fn(
        q_segment_ids: Int[Array, "b lq"],
        kv_segment_ids: Int[Array, "b lk"],
    ) -> Float[Array, "b h lq lk"]:
        """compute attention bias for block based on segment ids."""
        mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
        bias = jnp.where(mask, 0, -jnp.inf)
        return einops.repeat(bias, "b lq lk -> b h lq lk", h=h)

    def attn(q, k, v, q_kwargs, kv_kwargs):
        # TODO: why can't I just bind axis_name using functools.partial
        return ring_attention(
            q,
            k,
            v,
            axis_name="sp",
            bias_fn=bias_fn,
            bias_q_kwargs=q_kwargs,
            bias_kv_kwargs=kv_kwargs,
            block_impl=impl,
        )

    ring_attention_sharded = shard_map(
        attn,
        mesh=mesh,
        in_specs=(
            PartitionSpec("dp", "sp", None),  # q
            PartitionSpec("dp", "sp", None),  # k
            PartitionSpec("dp", "sp", None),  # v
            {  # bias_fn_q_kwargs
                "q_segment_ids": PartitionSpec("dp", "sp"),
            },
            {  # bias_fn_kv_kwargs
                "kv_segment_ids": PartitionSpec("dp", "sp"),
            },
        ),
        out_specs=PartitionSpec("dp", "sp", None),
        check_rep=False,
    )

    output = jax.jit(ring_attention_sharded)(
        q,
        k,
        v,
        {"q_segment_ids": q_segment_ids},
        {"kv_segment_ids": kv_segment_ids},
    )
    np.testing.assert_array_equal(jnp.isnan(output), jnp.isnan(expected_output))
    np.testing.assert_allclose(output, expected_output, rtol=0.01, atol=0.001)

    # Test backwards pass with bias.

    def f(q, k, v):
        o = ring_attention_sharded(
            q,
            k,
            v,
            {"q_segment_ids": q_segment_ids},
            {"kv_segment_ids": kv_segment_ids},
        )
        return jnp.sum(o * do)

    dq, dk, dv = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(q, k, v)

    # expected outputs
    dq_, dk_, dv_ = jax.jit(
        jax.grad(
            lambda q, k, v: jnp.sum(do * mha(q, k, v, bias=bias)),
            argnums=(0, 1, 2),
        )
    )(q, k, v)

    np.testing.assert_allclose(dq, dq_, atol=1e-4)
    np.testing.assert_allclose(dk, dk_, atol=1e-4)
    np.testing.assert_allclose(dv, dv_, atol=1e-4)


def _test_bias_fn(
    segment_ids: Int[Array, "b l"],
    causal: bool = False,
):
    """compute attention bias for block based on segment ids."""
    _, length = segment_ids.shape

    mask = segment_ids[:, :, None] == segment_ids[:, None, :]

    if causal:
        i = jnp.arange(length)
        causal_mask = i[None, :] <= i[:, None]
        mask &= causal_mask

    return jnp.where(mask, 0, -jnp.inf)


@pytest.mark.parametrize("impl", ["jax", "pallas"])
@pytest.mark.parametrize(
    "seed,length,h,d",
    [
        (0, 16, 1, 2),
        (1, 128, 4, 64),
        (2, 512, 4, 16),
        (4, 2048, 4, 64),
    ],
)
def test_ring_self_attention(impl: str, seed: int, length: int, h: int, d: int):
    key = jax.random.PRNGKey(seed)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    batch_size = 2

    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], shape=(batch_size, length, h, d))
    k = jax.random.normal(keys[1], shape=(batch_size, length, h, d))
    v = jax.random.normal(keys[2], shape=(batch_size, length, h, d))

    segment_ids = jax.random.randint(
        keys[3], shape=(batch_size, length), minval=0, maxval=5
    )
    positions = einops.repeat(jnp.arange(length), "l -> b l", b=batch_size)

    bias = _test_bias_fn(segment_ids, causal=True)
    bias = bias[:, None, :, :]
    expected_output = mha(q, k, v, bias=bias)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))
    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    segment_ids = jax.device_put(segment_ids, sharding)
    positions = jax.device_put(positions, sharding)

    output = jax.jit(
        ring_self_attention,
        static_argnames=["mesh", "pspec", "causal", "block_impl"],
    )(
        q,
        k,
        v,
        mesh=mesh,
        pspec=q.sharding.spec,
        segment_ids=segment_ids,
        positions=positions,
        causal=True,
        block_impl=impl,
    )

    np.testing.assert_allclose(output, expected_output, rtol=0.01, atol=0.001)
