# GPU VRAM accounting

## Two transfer modes, two different accounting models

The tested deployment now uses multi-group `engine_driven` transfer. The older
CUDA IPC/pointer path remains documented as a fallback because its
`nvidia-smi` rows are easy to misinterpret.

### Pointer/CUDA IPC fallback

In pointer mode the standalone LMCache process opens every GPU and imports the
vLLM KV tensors through CUDA IPC. NVIDIA process accounting attributes the
mapped allocation to both the exporting vLLM worker and the importing LMCache
server.

That does not mean two independent physical KV pools were allocated.
Conceptually:

```text
GPU physical memory
├─ model weights / runtime allocations        owned by vLLM worker
├─ active vLLM KV tensor                       owned by vLLM worker
│   └─ CUDA IPC mapping                        visible to LMCache process
├─ LMCache temporary staging                   independent allocation
├─ block-ID / metadata buffers                 independent allocation
└─ LMCache CUDA context / modules               independent overhead
```

Adding the vLLM and LMCache process rows double-counts the shared mapping.

The original pointer profile reserved 576 MiB/rank for active vLLM KV. The
LMCache row measured about 960 MiB/GPU with four staging chunks and 724–726
MiB/GPU after reducing staging to one chunk. The 234–236 MiB change was a real
staging reduction; most of the remaining row was the shared 576 MiB mapping
plus one staging chunk, block IDs, and context overhead.

### CPU-only engine-driven mode

Engine-driven mode moves GPU work into the existing vLLM worker contexts:

```text
vLLM worker
├─ owns model and active KV tensors
├─ gathers GPU KV into bounded CPU transfer buffers
└─ scatters restored CPU objects back into paged GPU KV

standalone LMCache server (CUDA_VISIBLE_DEVICES="")
├─ stores objects in host-RAM L1
├─ stores persistent objects in fs_native/O_DIRECT L2
└─ owns no CUDA context, IPC mapping, GPU staging, or block-ID workspace
```

The launch profile deliberately uses an empty LMCache shared-memory name. This
selects bounded per-transfer pickle buffers instead of registering the entire
48 GB L1 shared-memory pool in every GPU worker.

Validated process accounting:

| Mode | Standalone LMCache process attribution |
|---|---:|
| Pointer fallback, staging batch 1 | ~724–726 MiB/GPU |
| Multi-group engine-driven | 0 MiB/GPU |

In engine-driven mode `nvidia-smi --query-compute-apps` must list only the eight
vLLM worker processes. The LMCache server still exists as a CPU process and
remains visible through its health endpoint, but it must not appear in the GPU
compute-process list.

## Active vLLM KV is a separate capacity decision

Removing LMCache server VRAM does not remove the active GPU KV working set.
The validated 420K profile intentionally reinvests some reclaimed headroom:

```text
K3_MAX_MODEL_LEN=420000
K3_MAX_NUM_SEQS=2
K3_KV_CACHE_MEMORY_BYTES=2147483648  # 2 GiB per rank
```

With a logical LMCache/DCP chunk of 12,288 tokens, one 420K sequence rounds to:

```text
ceil(420000 / 12288) × 12288 = 430080 tokens
```

Two full-length sequences therefore require at least 860,160 aggregate KV
tokens. The validated startup reported 1,034,634 tokens and 2.46× maximum
concurrency at 420K. After a long persistent-L2 restore, total free VRAM was
613–693 MiB/GPU and the strict CUDA/engine error scan was clean. Always confirm
the actual `GPU KV cache size` and `Maximum concurrency` startup lines;
byte-to-token capacity is model/layout specific.

Do not lower or raise this reservation based only on one process row. Validate
startup, the longest intended context, concurrency, request-time headroom, and
CUDA OOM logs together.

## What LMCache does and does not offload

LMCache extends reusable prefix KV storage to:

- L1: host RAM
- L2: persistent filesystem storage

It does **not**:

- offload Kimi K3 model weights,
- eliminate vLLM's active GPU KV working set,
- make the vLLM workers CPU-only,
- make pointer-mode process rows directly additive.

The engine-driven benefit is specifically that the *standalone LMCache server*
uses no GPU memory while persistent L1/L2 restore remains available.

## Measurement procedure

1. Start the exact model, graph, KV, and transfer configuration.
2. Wait for API readiness and idle requests.
3. Confirm the server registered eight non-CUDA engine contexts.
4. Capture GPU processes and total memory:

   ```bash
   nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
     --format=csv,noheader,nounits
   nvidia-smi --query-gpu=index,memory.used,memory.free \
     --format=csv,noheader,nounits
   ```

5. Send a deterministic prompt longer than one 12,288-token chunk.
6. Capture the same values after transfer activity reaches idle.
7. Confirm the LMCache server PID never appears as a GPU process.
8. Recreate the container while preserving L2, resend the exact prompt, and
   prove external tokens were restored on all eight ranks.
9. Scan logs for `CUDA out of memory`, `Traceback`, engine death, and transfer
   errors.

Process-level rows prove attribution. The device-level used/free delta proves
physical headroom. Record both.

## Invalid shortcuts

### Hiding GPUs while still using pointer mode

`CUDA_VISIBLE_DEVICES=""` is valid for the standalone server only when the
connector and server both negotiate `engine_driven`. It breaks CUDA IPC
registration in pointer mode.

### Summing pointer-mode process rows

The sum double-counts imported IPC mappings and can exceed the physical delta.

### Treating a lower process row as correctness proof

A transfer-protocol change must also pass registration, inference, deterministic
persistent-L2 restore, output correctness, and strict error checks. Zero server
VRAM alone is not success.

### Allocating KV without block math

For this TP8/DCP8 profile, use the reported capacity and the 12,288-token
logical chunk. A nominal 420K context consumes 430,080 block-rounded tokens.
Changing DCP, cache layouts, chunk size, or model architecture invalidates this
calculation.
