#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Modified by myshytf/kimi-k3-qsrt-lmcache on 2026-08-14.
# Adds hybrid-safe TP8/DCP8 LMCache startup and CPU-only multi-group
# engine-driven transfer while preserving persistent L1/L2 caching.
# Launch Kimi K3 QSRT with one full CUDA graph for each decode batch shape.
set -euo pipefail

K3_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K3_PYTHON_BIN="${K3_PYTHON_BIN:-${K3_SCRIPT_DIR}/.venv/bin/python}"
K3_MODEL_DIR="${K3_MODEL_DIR:-/data/models/Kimi-K3-QSRT-SQG-XOR-CHEB-T12-3p08-v2-mm-mxfp8-model}"
K3_ENABLE_DSPARK="${K3_ENABLE_DSPARK:-1}"
K3_DSPARK_MODEL_DIR="${K3_DSPARK_MODEL_DIR:-/data/models/Inferact-Kimi-K3-DSpark}"
K3_NUM_SPECULATIVE_TOKENS="${K3_NUM_SPECULATIVE_TOKENS:-7}"
K3_DSPARK_ATTENTION_BACKEND="${K3_DSPARK_ATTENTION_BACKEND:-B12X_MLA}"

if [[ ! -x "${K3_PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${K3_PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi
if [[ ! -d "${K3_MODEL_DIR}" ]]; then
  echo "QSRT checkpoint directory not found: ${K3_MODEL_DIR}" >&2
  exit 1
fi
if [[ "${K3_ENABLE_DSPARK}" != 0 && "${K3_ENABLE_DSPARK}" != 1 ]]; then
  echo "K3_ENABLE_DSPARK must be 0 or 1." >&2
  exit 2
fi
if [[ "${K3_ENABLE_DSPARK}" == 1 ]]; then
  if [[ ! -f "${K3_DSPARK_MODEL_DIR}/config.json" \
    || ! -f "${K3_DSPARK_MODEL_DIR}/model.safetensors" ]]; then
    echo "Kimi-K3 DSpark checkpoint is incomplete: ${K3_DSPARK_MODEL_DIR}" >&2
    exit 1
  fi
  if [[ "${K3_NUM_SPECULATIVE_TOKENS}" != 7 ]]; then
    echo "Kimi-K3 DSpark requires seven speculative tokens." >&2
    exit 2
  fi
  if [[ "${K3_DSPARK_ATTENTION_BACKEND}" != B12X_MLA ]]; then
    echo "Kimi-K3 DSpark is qualified with B12X_MLA." >&2
    exit 2
  fi
fi

# Kimi K3 otherwise defaults to breakable CUDA graphs. This launch profile
# requires B12X MLA, KDA, MoE, and dense linears in one full decode graph.
case "${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF|"")
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    ;;
  *)
    echo "This launcher requires full, unbroken decode CUDA graphs." >&2
    echo "Do not set VLLM_USE_BREAKABLE_CUDAGRAPH=1 for this run." >&2
    exit 1
    ;;
esac

for arg in "$@"; do
  case "${arg}" in
    --enforce-eager|--enforce-eager=*|\
    --compilation-config|--compilation-config=*|--compilation-config.*|\
    -cc|-cc=*|-cc.*|\
    --attention-backend|--attention-backend=*|\
    --linear-backend|--linear-backend=*|\
    --moe-backend|--moe-backend=*|\
    --speculative-config|--speculative-config=*|\
    --speculative-model|--speculative-model=*)
      echo "Argument ${arg} would override a launcher-owned runtime option." >&2
      exit 1
      ;;
  esac
done

export PYTHONPATH="${K3_SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-134217728}"

export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export KDA_DISABLE_AUTOTUNE="${KDA_DISABLE_AUTOTUNE:-1}"

export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

K3_KV_CACHE_ARGS=()
K3_KV_CACHE_MEMORY_BYTES="${K3_KV_CACHE_MEMORY_BYTES:-2147483648}"
if [[ -n "${K3_KV_CACHE_MEMORY_BYTES}" \
  && "${K3_KV_CACHE_MEMORY_BYTES}" != "0" \
  && "${K3_KV_CACHE_MEMORY_BYTES}" != "auto" ]]; then
  K3_KV_CACHE_ARGS+=(
    --kv-cache-memory-bytes "${K3_KV_CACHE_MEMORY_BYTES}"
  )
fi

# FULL_DECODE_ONLY captures the whole model for uniform single-token decode
# while leaving prefill outside CUDA graphs. Custom ops remain opaque launches
# inside the outer graph; they are not eager regions or capture boundaries.
# Allow compose to constrain capture sizes (for example [1,2] for C2) while
# preserving the image's original full-decode configuration by default.
K3_COMPILATION_CONFIG_DEFAULT='{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}'
K3_COMPILATION_CONFIG="${K3_COMPILATION_CONFIG:-${K3_COMPILATION_CONFIG_DEFAULT}}"

K3_SPECULATIVE_ARGS=()
if [[ "${K3_ENABLE_DSPARK}" == 1 ]]; then
  printf -v K3_SPECULATIVE_CONFIG \
    '{"method":"dspark","model":"%s","num_speculative_tokens":7,"attention_backend":"%s","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    "${K3_DSPARK_MODEL_DIR}" "${K3_DSPARK_ATTENTION_BACKEND}"
  K3_SPECULATIVE_ARGS+=(--speculative-config "${K3_SPECULATIVE_CONFIG}")
fi

# ── LMCache MP server: engine-driven CPU-only server + CPU-RAM L1 + NVMe L2 ──
# GPU KV gather/scatter runs inside the existing vLLM worker CUDA contexts.
# The standalone LMCache server sees no GPUs and therefore owns no per-GPU
# CUDA context, IPC mapping, or staging allocation. An empty SHM name forces
# per-transfer pickle buffers instead of pinning the entire 48 GB L1 pool in
# every GPU worker; the 48 GB L1 capacity and persistent L2 remain unchanged.
K3_LMCACHE_ARGS=()
if [[ "${K3_ENABLE_LMCACHE:-1}" == "1" ]]; then
  # expandable_segments conflicts with the LMCache MP connector (proven in
  # the GLM-5.2 MP setup), so it must not be active.
  unset PYTORCH_CUDA_ALLOC_CONF || true

  K3_LMCACHE_BIN="${K3_LMCACHE_BIN:-$(command -v lmcache || echo /opt/venv/bin/lmcache)}"
  K3_LMCACHE_MP_HOST="${K3_LMCACHE_MP_HOST:-127.0.0.1}"
  K3_LMCACHE_MP_PORT="${K3_LMCACHE_MP_PORT:-5555}"
  K3_LMCACHE_HTTP_PORT="${K3_LMCACHE_HTTP_PORT:-8088}"
  # vLLM's scheduler block is 1536 tokens. The DCP-aware LMCache protocol
  # exposes one global engine block as 1536 x 8 KV shards = 12288 tokens.
  # Hybrid chunks must land on complete global recurrent-state snapshots.
  K3_LMCACHE_CHUNK_SIZE="${K3_LMCACHE_CHUNK_SIZE:-12288}"
  K3_LMCACHE_L1_GB="${K3_LMCACHE_L1_GB:-48}"
  K3_LMCACHE_L1_INIT_GB="${K3_LMCACHE_L1_INIT_GB:-${K3_LMCACHE_L1_GB}}"
  K3_LMCACHE_L2_GB="${K3_LMCACHE_L2_GB:-1200}"
  K3_LMCACHE_DISK_PATH="${K3_LMCACHE_DISK_PATH:-/lmcache/l2}"
  K3_LMCACHE_LOG="${K3_LMCACHE_LOG:-/container-tmp/lmcache_mp_server.log}"
  K3_LMCACHE_TRANSFER_MODE="${K3_LMCACHE_TRANSFER_MODE:-engine_driven}"
  K3_LMCACHE_SHM_NAME="${K3_LMCACHE_SHM_NAME:-}"
  # An empty CUDA_VISIBLE_DEVICES is process-local: vLLM workers still see all
  # GPUs, while the standalone LMCache server remains CPU-only.
  K3_LMCACHE_SERVER_ENV="${K3_LMCACHE_SERVER_ENV:-CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY}"
  read -r -a K3_LMCACHE_SERVER_ENV_ARGS <<< "${K3_LMCACHE_SERVER_ENV}"
  for assignment in "${K3_LMCACHE_SERVER_ENV_ARGS[@]}"; do
    if [[ ! "${assignment}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
      echo "[kimik3] Invalid K3_LMCACHE_SERVER_ENV assignment: ${assignment}" >&2
      exit 2
    fi
  done
  # Stale-lock recovery timeouts for crashed readers/writers, not entry TTLs.
  K3_LMCACHE_L1_WRITE_TTL="${K3_LMCACHE_L1_WRITE_TTL:-600}"
  K3_LMCACHE_L1_READ_TTL="${K3_LMCACHE_L1_READ_TTL:-300}"
  K3_LMCACHE_EVICT_WATERMARK="${K3_LMCACHE_EVICT_WATERMARK:-0.90}"
  K3_LMCACHE_EVICT_RATIO="${K3_LMCACHE_EVICT_RATIO:-0.10}"
  K3_LMCACHE_PREFETCH_POLICY="${K3_LMCACHE_PREFETCH_POLICY:-default}"
  K3_LMCACHE_PREFETCH_MAX_INFLIGHT="${K3_LMCACHE_PREFETCH_MAX_INFLIGHT:-8}"

  mkdir -p "${K3_LMCACHE_DISK_PATH}"
  echo "[kimik3] Starting LMCache MP server: tcp://${K3_LMCACHE_MP_HOST}:${K3_LMCACHE_MP_PORT}, mode=${K3_LMCACHE_TRANSFER_MODE}, CPU-only server, L1=${K3_LMCACHE_L1_GB}GB init=${K3_LMCACHE_L1_INIT_GB}GB CPU RAM, L2=${K3_LMCACHE_L2_GB}GB @ ${K3_LMCACHE_DISK_PATH}, chunk=${K3_LMCACHE_CHUNK_SIZE}"
  rm -f "${K3_LMCACHE_LOG}"
  env "${K3_LMCACHE_SERVER_ENV_ARGS[@]}" "${K3_LMCACHE_BIN}" server \
    --host "${K3_LMCACHE_MP_HOST}" \
    --port "${K3_LMCACHE_MP_PORT}" \
    --supported-transfer-mode "${K3_LMCACHE_TRANSFER_MODE}" \
    --shm-name "${K3_LMCACHE_SHM_NAME}" \
    --chunk-size "${K3_LMCACHE_CHUNK_SIZE}" \
    --separate-object-groups \
    --l1-size-gb "${K3_LMCACHE_L1_GB}" \
    --l1-init-size-gb "${K3_LMCACHE_L1_INIT_GB}" \
    --l1-write-ttl-seconds "${K3_LMCACHE_L1_WRITE_TTL}" \
    --l1-read-ttl-seconds "${K3_LMCACHE_L1_READ_TTL}" \
    --eviction-policy LRU \
    --eviction-trigger-watermark "${K3_LMCACHE_EVICT_WATERMARK}" \
    --eviction-ratio "${K3_LMCACHE_EVICT_RATIO}" \
    --l2-prefetch-policy "${K3_LMCACHE_PREFETCH_POLICY}" \
    --l2-prefetch-max-in-flight "${K3_LMCACHE_PREFETCH_MAX_INFLIGHT}" \
    --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"${K3_LMCACHE_DISK_PATH}\",\"relative_tmp_dir\":\"tmp\",\"max_capacity_gb\":${K3_LMCACHE_L2_GB},\"use_odirect\":true,\"num_workers\":12,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.1}}" \
    --http-port "${K3_LMCACHE_HTTP_PORT}" \
    >"${K3_LMCACHE_LOG}" 2>&1 &
  K3_LMCACHE_MP_PID=$!
  trap 'kill ${K3_LMCACHE_MP_PID:-} 2>/dev/null || true' EXIT

  _lmcache_ready=0
  for _i in $(seq 1 120); do
    if ! kill -0 "${K3_LMCACHE_MP_PID}" 2>/dev/null; then
      echo "[kimik3] LMCache MP server exited during startup; log follows:" >&2
      sed -n '1,220p' "${K3_LMCACHE_LOG}" >&2 || true
      exit 1
    fi
    if [[ -f "${K3_LMCACHE_LOG}" ]] && grep -q "ZMQ cache server is running" "${K3_LMCACHE_LOG}"; then
      _lmcache_ready=1
      break
    fi
    sleep 1
  done
  if [[ "${_lmcache_ready}" != "1" ]]; then
    echo "[kimik3] LMCache MP server did not become ready; log follows:" >&2
    sed -n '1,220p' "${K3_LMCACHE_LOG}" >&2 || true
    exit 1
  fi
  echo "[kimik3] LMCache MP server ready"

  _kv_transfer_config=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://%s","lmcache.mp.port":%s,"lmcache.mp.mq_timeout":30,"lmcache.mp.heartbeat_interval":5,"lmcache.mp.mp_transfer_mode":"%s"}}' "${K3_LMCACHE_MP_HOST}" "${K3_LMCACHE_MP_PORT}" "${K3_LMCACHE_TRANSFER_MODE}")
  # LMCacheMPConnector implements SupportsHMA and must receive Kimi-K3's
  # separate KDA and MLA cache groups. Forcing the legacy unified-cache path
  # makes vLLM fail while trying to promote MambaSpec to MLAAttentionSpec.
  K3_LMCACHE_ARGS+=( --kv-transfer-config "${_kv_transfer_config}" )
fi

exec "${K3_PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${K3_MODEL_DIR}" \
  --served-model-name "${K3_SERVED_MODEL_NAME:-Kimi-K3}" \
  --trust-remote-code \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice \
  --host "${K3_HOST:-0.0.0.0}" \
  --port "${K3_PORT:-8000}" \
  --tensor-parallel-size "${K3_TP_SIZE:-12}" \
  --load-format instanttensor \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X_MLA \
  --compilation-config "${K3_COMPILATION_CONFIG}" \
  "${K3_SPECULATIVE_ARGS[@]}" \
  --additional-config '{"kda_prefill_backend":"triton"}' \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode "${K3_MAMBA_CACHE_MODE:-align}" \
  --max-model-len "${K3_MAX_MODEL_LEN:-262144}" \
  --kv-cache-dtype fp8 \
  --block-size "${K3_BLOCK_SIZE:-128}" \
  --gpu-memory-utilization "${K3_GPU_MEMORY_UTILIZATION:-0.9711}" \
  "${K3_KV_CACHE_ARGS[@]}" \
  --max-num-batched-tokens "${K3_MAX_NUM_BATCHED_TOKENS:-1536}" \
  --max-num-seqs "${K3_MAX_NUM_SEQS:-3}" \
  "${K3_LMCACHE_ARGS[@]}" \
  "$@"
