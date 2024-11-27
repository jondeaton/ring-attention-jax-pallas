"""Test for ring attention."""

import os


import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import einops

import functools
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

from models.attention.ring import ring_attention

flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = flags + " --xla_force_host_platform_device_count=8"


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
    d = 128

    q = jax.random.normal(key, shape=(batch_size, length, d))
    k = jax.random.normal(key, shape=(batch_size, length, d))
    v = jax.random.normal(key, shape=(batch_size, length, d))
    attn_bias = jax.random.normal(key, shape=(batch_size, length, length))
    segment_ids = einops.repeat(jnp.arange(length) // 128, "l -> b l", b=batch_size)

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp", None))

    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    attn_bias = jax.device_put(attn_bias, sharding)
    segment_ids = jax.device_put(segment_ids, sharding)

    chunk_size = 32

    ring_attention_sharded = shard_map(
        functools.partial(
            ring_attention,
            axis_name="sp",
            float32_logits=True,
            cache_idx=None,
            blockwise_kwargs=dict(
                causal_block_size=1,
                deterministic=True,
                query_chunk_size=chunk_size,
                key_chunk_size=chunk_size,
                dtype=jnp.float32,
                policy=jax.checkpoint_policies.nothing_saveable,
                precision=jnp.float32,
                prevent_cse=False,
            ),
        ),
        mesh=mesh,
        in_specs=(
            PartitionSpec("dp", "sp", None),  # q
            PartitionSpec("dp", "sp", None),  # k
            PartitionSpec("dp", "sp", None),  # v
            PartitionSpec("dp", None, None, None),  #
            PartitionSpec("dp", None),
        ),
        out_specs=PartitionSpec("dp", "sp", None),
        check_rep=False,
    )

    attn_output = ring_attention_sharded(q, k, v, attn_bias, segment_ids)
    assert not jnp.isnan(attn_output).any()
