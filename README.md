# Ring Attention in JAX / Pallas with flexible attention.

This is a JAX adaptation of the ring-attention algorithm introduced in [1] inspired by
the original JAX implementation in [2]. The original code was adapted significantly
to more closely resemble the Flash Attention 2 algorithm [3], but where the
loads/stores to/from SRAM/HBM are replaced with rotations of query/key/value around the
ring of devices which the sequences are shadred across.

This implementation also supports a general mechamnism for incorporating arbitrary
attention biases from a user-defined function, similar to Flex Attention [4]. Finally, 
single-device attention block computaiton is performed with Pallas kernels heavily
adopted from the implementations provided in the JAX repository [5].

References:
1. Ring Attention with Blockwise Transformers for Near-Infinite Context Liu et al. 
    https://arxiv.org/abs/2310.01889
2. Ring Attention JAX code: https://github.com/haoliuhl/ringattention
3. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. Tri
    Dao https://arxiv.org/abs/2307.08691
4. Flex Attention, https://pytorch.org/blog/flexattention/
5. Pallas/JAX Flash attention implementation for Pallas kernels.
    https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/attention.py
