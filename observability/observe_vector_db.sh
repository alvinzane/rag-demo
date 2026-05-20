#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BPF_DIR="${ROOT_DIR}/observability/bpftrace"
OUT_DIR="${ROOT_DIR}/.rag/observability"
DURATION_SECONDS="${DURATION_SECONDS:-60}"

usage() {
  cat <<'USAGE'
Usage:
  observability/observe_vector_db.sh <profile> [duration_seconds]

Profiles:
  cpu           Sample kernel CPU stacks for milvus/qdrant/weaviate
  offcpu        Show time spent off CPU, useful for lock/I/O waits
  io            Show block I/O latency
  vfs           Show VFS read/write latency and bytes
  tcp           Show TCP send/receive size for Milvus
  retrans       Show TCP retransmits for vector DB processes
  syscalls      Show syscall latency for Milvus
  faults        Show page faults for vector DB processes
  docker-stats  Capture docker stats once per second

Examples:
  sudo observability/observe_vector_db.sh cpu 30
  sudo observability/observe_vector_db.sh offcpu 60
  sudo observability/observe_vector_db.sh docker-stats 60

Environment:
  DURATION_SECONDS=60 overrides the default duration when the second argument is omitted.
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROFILE="$1"
if [[ $# -ge 2 ]]; then
  DURATION_SECONDS="$2"
fi

mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="${OUT_DIR}/${PROFILE}-${STAMP}.log"

run_bpftrace() {
  local script="$1"
  if [[ ! -f "${script}" ]]; then
    echo "Missing bpftrace script: ${script}" >&2
    exit 1
  fi
  echo "Profile: ${PROFILE}"
  echo "Duration: ${DURATION_SECONDS}s"
  echo "Press Ctrl-C to stop early."
  echo "Writing ${OUT_FILE}"
  timeout "${DURATION_SECONDS}" bpftrace "${script}" 2>&1 | tee "${OUT_FILE}"
}

case "${PROFILE}" in
  cpu)
    run_bpftrace "${BPF_DIR}/vector_db_cpu.bt"
    ;;
  offcpu)
    run_bpftrace "${BPF_DIR}/vector_db_offcpu.bt"
    ;;
  io)
    run_bpftrace "${BPF_DIR}/milvus_io.bt"
    ;;
  vfs)
    run_bpftrace "${BPF_DIR}/vector_db_vfs_latency.bt"
    ;;
  tcp)
    run_bpftrace "${BPF_DIR}/milvus_tcp.bt"
    ;;
  retrans)
    run_bpftrace "${BPF_DIR}/vector_db_tcp_retrans.bt"
    ;;
  syscalls)
    run_bpftrace "${BPF_DIR}/milvus_syscalls.bt"
    ;;
  faults)
    run_bpftrace "${BPF_DIR}/vector_db_page_faults.bt"
    ;;
  docker-stats)
    echo "Writing ${OUT_FILE}"
    end_time=$((SECONDS + DURATION_SECONDS))
    {
      while (( SECONDS < end_time )); do
        date --iso-8601=seconds
        docker stats --no-stream --format \
          'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}'
        sleep 1
      done
    } 2>&1 | tee "${OUT_FILE}"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    usage
    exit 2
    ;;
esac
