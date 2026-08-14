# Root-cause analysis

## Summary

Kimi K3 is not a homogeneous attention model. The tested checkpoint combines recurrent Kimi Delta Attention (represented through vLLM's Mamba-like cache specification) with MLA attention. Those groups do not have the same tensor shape, state semantics, block-ID space, or storage format.

Enabling LMCache exposed three independent startup incompatibilities in sequence. Each next failure appeared only after the previous one was fixed. A fourth issue was a correct-but-avoidable GPU staging allocation.

## Cache path involved

```text
vLLM Kimi K3 model
  ├─ KDA/Mamba recurrent-state cache groups
  └─ MLA attention KV cache groups
          │
          ▼
LMCacheMPConnector.register_kv_caches()
          │
          ├─ apply_kv_cache_group_edits(...)
          ├─ create_engine_group_infos_from_vllm(..., dcp_size=8)
          └─ pointer/CUDA IPC transfer context
                  ├─ CPU-RAM L1
                  └─ fs_native/O_DIRECT L2
```

The connector must preserve separate hybrid cache groups while presenting each raw vLLM tensor in a format understood by the installed LMCache native layer.

## Failure 1: forcing the non-hybrid manager

### Error

```text
ValueError: Failed to promote local KV cache specs to one unified type
```

### Cause

`--disable-hybrid-kv-cache-manager` instructs vLLM to promote all local cache specifications into one uniform type. That cannot represent Kimi K3's KDA/Mamba state and MLA attention KV simultaneously.

This flag is sometimes suggested for older connectors that do not implement hybrid model support. It is wrong here: `LMCacheMPConnector` implements vLLM's `SupportsHMA` interface and must receive separate groups.

### Fix

- Remove `--disable-hybrid-kv-cache-manager`.
- Retain vLLM's hybrid KV cache manager.
- Use `--mamba-cache-mode align` so recurrent snapshots and attention KV advance on a common logical prefix.
- Start LMCache with `--separate-object-groups` so differently shaped groups are not serialized into one object.

## Failure 2: vLLM 0.26 unified Mamba representation

### Error

```text
expected a Mamba [conv_state, ssm_state] tensor list, got Tensor
```

### Cause

Older LMCache 0.5.2 integration code expected a Mamba layer cache as a two-item list:

```text
[conv_state_tensor, ssm_state_tensor]
```

vLLM 0.26 registers the states as one unified tensor. The old `_MambaPageViewEdit` rejected it before LMCache could register the engine groups.

Upstream [LMCache PR #4206](https://github.com/LMCache/LMCache/pull/4206) added the vLLM 0.26 unified-Mamba and subpaged-attention view logic that later shipped in the 0.5.3 line. A direct package upgrade was not safe for the tested image because its `0.5.2+glm52dcp.5` package contains custom DCP logic and a native extension with its own `EngineKVFormat` ABI.

### Narrow backport

The overlay performs four coupled changes:

1. `LMCacheMPConnector.register_kv_caches()` obtains `vllm_layout_hints()` and applies cache-group edits before LMCache group metadata is generated.
2. `kv_cache_group_edits.py` recognizes vLLM 0.26 unified Mamba tensors and subpaged MLA tensors.
3. `gpu_connector/utils.py` determines formats per engine group when KDA and MLA tensor shapes are mixed instead of assuming one global format.
4. The local `dcp_size` parameter continues through `create_engine_group_infos_from_vllm()` rather than being lost during the backport.

### Native-format adapter

The newer upstream unified-Mamba view can use `[NB, BS, NH, CS]`. The custom 0.5.2 native extension in the tested image does not expose every newer format enum. Kimi K3's unified KDA cache has `NH == 1`, so the exact same storage can be exposed as:

```python
kv_cache.view(num_blocks, spec.block_size, -1)
```

This maps `[NB, BS, 1, CS]` to `[NB, BS, HS]` without a copy. The safety invariants are:

- `NH` must equal 1,
- `data_ptr()` must not change,
- byte count must not change,
- logical block boundaries must remain intact.

Do not generalize this flattening to `NH != 1`.

## Failure 3: DCP block-unit mismatch

### Error

```text
chunk_size (1536) must be a multiple of engine group 0 tokens_per_block (12288)
```

The complete message can include implementation-specific group labels, but the two numbers are the useful evidence.

### Cause

Two different units had been assigned the same value:

- vLLM scheduler/local recurrent step: 1536 tokens
- LMCache DCP-aware global engine block: local step × DCP KV shards

For the tested TP8/DCP8 configuration:

```text
1536 × 8 = 12288
```

The scheduler still advances at 1536 tokens per local rank, so `max_num_batched_tokens=1536` remains correct. LMCache must store complete global snapshots and therefore needs a chunk that is a multiple of 12,288 tokens.

### Fix

```text
K3_MAX_NUM_BATCHED_TOKENS=1536
K3_LMCACHE_CHUNK_SIZE=12288
```

Do not raise the scheduler value to 12,288 merely to make the numbers identical.

## Issue 4: avoidable staging VRAM

### Observation

After startup became correct, `nvidia-smi` attributed about 960 MiB per GPU to the LMCache server process.

### Cause

LMCache-driven pointer mode does two different things:

1. maps the existing vLLM active KV tensor through CUDA IPC, and
2. allocates reusable temporary gather/scatter staging buffers.

The mapped storage is shared physical memory even though process accounting shows it for both processes. The default temporary staging batch contained four chunks and was a real additional allocation.

### Fix

`cache_context.py` now reads and validates:

```text
LMCACHE_MP_GPU_STAGING_BATCH_SIZE
```

The launcher starts the server with:

```text
CUDA_MODULE_LOADING=LAZY
LMCACHE_MP_GPU_STAGING_BATCH_SIZE=1
```

Transfers larger than one chunk are processed in windows. One is the minimum tested correct batch for this deployment.

Observed attribution:

```text
before: ~960 MiB/GPU
idle after: ~724 MiB/GPU
request after: ~726 MiB/GPU
saved: ~234–236 MiB/GPU
```

See [VRAM_ACCOUNTING.md](VRAM_ACCOUNTING.md) before interpreting per-process sums.

## Issue 5: stock engine-driven transfer could not serve Kimi K3

### Observation

Hiding GPUs from the standalone server removed its CUDA allocation, but the base image's engine-driven implementation could register only one cache layout. Kimi K3 exposes four separate recurrent KDA/Mamba and MLA object groups.

### Cause

The installed protocol had one global layout and did not preserve object-group IDs through registration, key resolution, transfer planning, and worker gather/scatter. Its DCP geometry also confused the 12,288-token logical LMCache object with the 1,536 physical tokens resident on each DCP rank.

### Fix

The overlay backports the reviewed multi-group protocol from LMCache PR #4410 and pins its PR-head commit in `manifest.json`. A local adaptation then:

- registers all four group layouts for each of eight ranks,
- retains group IDs through lookup, store, load, and transfer plans,
- keeps 12,288-token logical objects and keys,
- gathers/scatters 1,536 physical tokens per DCP rank,
- executes CUDA work in the existing vLLM worker contexts, and
- starts the standalone LMCache server with no visible GPU.

The server consequently owns no CUDA context, IPC mapping, GPU staging allocation, or block-ID workspace. Persistent host-RAM L1 and filesystem L2 remain active.

### Verification

The production gate requires 8/8 registered non-CUDA contexts, no LMCache GPU process, successful short inference, deterministic L2 restore after recreate, external-token metrics, and a strict error scan. Zero VRAM attribution alone is not correctness proof.

## Why Compose validation did not catch this

`docker compose config --quiet` validates YAML interpolation and structure. It cannot instantiate vLLM cache specifications, inspect tensor shapes, load the LMCache native extension, or compute DCP-aware tokens per block. All three fatal failures occur after the process enters model/engine initialization.

Use the complete validation gates in [VALIDATION.md](VALIDATION.md), not YAML validation alone.
