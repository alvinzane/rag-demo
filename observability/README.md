# Performance Observability

This directory contains lightweight scripts for watching vector database performance during the W4-T2 and W4-T3 demos.

The scripts focus on short classroom investigations rather than permanent monitoring. Run a benchmark in one terminal and an observability profile in another terminal.

## Installed Tools

Ubuntu packages used:

```bash
sudo apt-get install -y bpftrace bpfcc-tools linux-tools-common linux-tools-generic sysstat jq
```

Verify:

```bash
bpftrace --version
perf --version
sar -V
```

## Quick Start

Start vector DB services:

```bash
docker-compose up -d
```

Run a benchmark:

```bash
uv run rag-demo t2 bench --index-type IVF_FLAT --runs 1000 --dim 768 --nprobe 16
```

Observe from another terminal:

```bash
sudo observability/observe_vector_db.sh cpu 30
sudo observability/observe_vector_db.sh offcpu 30
sudo observability/observe_vector_db.sh io 30
sudo observability/observe_vector_db.sh docker-stats 30
```

Logs are written to:

```text
.rag/observability/
```

## Profiles

| Profile | Script | What to Look For |
| --- | --- | --- |
| `cpu` | `bpftrace/vector_db_cpu.bt` | Hot kernel stacks while Milvus/Qdrant/Weaviate consume CPU |
| `offcpu` | `bpftrace/vector_db_offcpu.bt` | Time spent waiting off CPU, often lock, I/O, or scheduling pressure |
| `io` | `bpftrace/milvus_io.bt` | Block I/O latency distribution |
| `vfs` | `bpftrace/vector_db_vfs_latency.bt` | Read/write latency at VFS layer |
| `tcp` | `bpftrace/milvus_tcp.bt` | TCP send/receive size distribution for Milvus |
| `retrans` | `bpftrace/vector_db_tcp_retrans.bt` | TCP retransmit counts |
| `syscalls` | `bpftrace/milvus_syscalls.bt` | Syscall latency distribution for Milvus |
| `faults` | `bpftrace/vector_db_page_faults.bt` | Page fault counts and stacks |
| `docker-stats` | Docker CLI | CPU, memory, network I/O, block I/O, PIDs |

## Suggested Demo Flow

IVF_FLAT `nprobe` sweep:

```bash
sudo observability/observe_vector_db.sh cpu 60
uv run rag-demo t2 sweep --index-type IVF_FLAT --values 1,4,8,16,32,64 --runs 1000 --dim 768
```

HNSW `ef` sweep:

```bash
sudo observability/observe_vector_db.sh offcpu 60
uv run rag-demo t2 sweep --index-type HNSW --values 16,32,64,128,256 --runs 1000 --dim 768
```

Qdrant vs Weaviate comparison:

```bash
sudo observability/observe_vector_db.sh docker-stats 60
uv run rag-demo t3 compare --runs 1000 --dim 768 --filtered
```

## Notes

- `bpftrace` generally requires root privileges.
- Some probes depend on kernel version and enabled tracepoints.
- If a profile fails to attach, run `sudo bpftrace -l '<probe-pattern>'` to confirm probe availability.
- The current process filters target common vector DB process names: `milvus`, `qdrant`, and `weaviate`.
