# GPU VRAM accounting

## Why `nvidia-smi` can look misleading

LMCache multiprocess pointer mode imports vLLM's KV tensors through CUDA IPC. NVIDIA per-process accounting then attributes the mapped allocation to both the exporting vLLM worker and the importing LMCache server process.

That does not mean two independent 576 MiB physical KV pools were allocated on each GPU.

Conceptually:

```text
GPU physical memory
├─ model weights / runtime allocations              owned by vLLM worker
├─ active vLLM KV tensor (576 MiB in tested setup)  owned by vLLM worker
│   └─ CUDA IPC mapping                              visible to LMCache process
├─ LMCache temporary staging                         real extra allocation
├─ block-ID / metadata buffers                       real extra allocation
└─ CUDA context / modules                            real extra overhead
```

Adding the vLLM and LMCache process rows double-counts the shared mapped region.

## What the staging patch changes

The tested `cache_context.py` originally provisioned temporary gather/scatter capacity for four chunks. The overlay reads:

```text
LMCACHE_MP_GPU_STAGING_BATCH_SIZE
```

and the launcher sets it to 1. Larger logical transfers are processed as one-chunk windows; the persistent vLLM active KV tensor is unchanged.

`CUDA_MODULE_LOADING=LAZY` also prevents eager loading of CUDA modules that are not yet needed, but the measurable fixed saving is primarily the reduced temporary staging capacity.

## Measured values

Per GPU on the validated TP8/DCP8 deployment:

| LMCache server state | `nvidia-smi` attribution |
|---|---:|
| Staging batch 4 | ~960 MiB |
| Staging batch 1, idle | ~724 MiB |
| Staging batch 1, after request | ~726 MiB |
| Reduction | ~234–236 MiB |

The active KV tensor configured per vLLM rank was:

```text
603,979,776 bytes = 576 MiB
```

Therefore most of the remaining ~724–726 MiB is expected shared mapping plus one staging chunk, metadata and CUDA context—not a second full external GPU cache.

## What LMCache does and does not offload

LMCache extends reusable prefix KV storage to:

- L1: pinned/host RAM
- L2: persistent filesystem storage

It does **not**:

- offload Kimi K3 model weights,
- eliminate the active GPU KV working set needed by vLLM,
- make CUDA IPC pointer mode GPU-invisible,
- make process-level memory rows directly additive.

The GPU-memory benefit is avoiding a large extra persistent GPU cache and reducing staging to the minimum tested amount. The larger capacity benefit comes from keeping reusable prefixes in host RAM and disk rather than only GPU memory.

## Measurement procedure

1. Start the same model and graph configuration.
2. Wait for idle after engine readiness.
3. Identify vLLM and LMCache PIDs.
4. Capture per-process usage:

   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
     --format=csv,noheader,nounits
   ```

5. Send the same long request and capture again.
6. Change only staging batch size, recreate, and repeat.
7. Confirm there are no old LMCache processes.
8. Also record total device memory, not only process rows:

   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.free \
     --format=csv,noheader,nounits
   ```

## Do not use these shortcuts

### Hiding all GPUs from LMCache

```text
CUDA_VISIBLE_DEVICES=""
```

This breaks or destabilizes CUDA IPC registration in pointer mode. It is not a supported CPU-only switch.

### Summing process rows

The sum includes shared mappings and can exceed the physical delta caused by the processes.

### Switching transfer protocols only for a lower process row

A different transfer mode changes correctness and protocol assumptions. The tested installed engine-driven path lacks safe multi-group Kimi K3 support. Lower apparent VRAM is not evidence of equivalent behavior.

### Lowering vLLM KV without capacity math

The 576 MiB active reservation was measured for this model/profile. Reducing it can lower concurrency or fail long-context requests. Treat active vLLM KV and LMCache staging as separate budgets.
