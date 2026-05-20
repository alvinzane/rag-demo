# RAG Demo 2

一个面向分享和学习的 RAG / 向量数据库 CLI demo。项目按 5 个连续目标推进：

1. W4-T1: LangChain / LlamaIndex RAG Hello World
2. W4-T2: Milvus 单机 + 1M 向量压测
3. W4-T3: Qdrant vs Weaviate 选型 POC
4. W4-T4: 分块策略对比
5. W4-T5: Embedding 选型

当前已实现：

- W4-T1：用 Ollama 对 Markdown 目录建立 RAG 索引，并提供一次性问答和交互式问答。
- W4-T2：用 Milvus Standalone + Attu 做 1M 向量压测，并用 bpftrace 做 eBPF 观察。
- W4-T3：用同一批 1M 向量对比 Qdrant 与 Weaviate 的过滤检索延迟、内存和 API 体验。

## Quick Start

安装依赖：

```bash
uv sync
```

默认使用远端 Ollama `http://192.168.1.18:11434`：

```bash
chat: deepseek-v4-pro:cloud
embedding: qwen3-embedding:latest
```

检查环境：

```bash
uv run rag-demo doctor
```

索引 Markdown 目录：

```bash
uv run rag-demo t1 index --docs ./confluence-export --persist ./.rag/index
```

可以重复 `--docs` 一次索引多个 Markdown 目录：

```bash
uv run rag-demo t1 index \
  --docs /home/ubuntu/workspace/easylearning/ez-cli/docs/beyond-vibe-coding \
  --docs /home/ubuntu/workspace/easylearning/ez-cli/docs/beyond-vibe-coding-cn \
  --persist ./.rag/beyond-vibe
```

问答：

```bash
uv run rag-demo t1 ask "团队的发布流程是什么？" --persist ./.rag/index
```

交互式问答：

```bash
uv run rag-demo t1 chat --persist ./.rag/index
```

## Worktree

规划文档见 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)。

## W4-T2 Milvus 1M Benchmark

启动 Milvus、etcd、MinIO 和 Attu：

```bash
docker-compose up -d
```

检查环境：

```bash
uv run rag-demo t2 doctor
```

准备确定性查询集。默认不落盘 1M 全量向量，导入时会按相同 seed 流式生成：

```bash
uv run rag-demo t2 generate --output .rag/t2 --count 1000000 --dim 768
```

课堂快速验证可以先跑小数据：

```bash
uv run rag-demo t2 load --count 10000 --dim 128 --reset
uv run rag-demo t2 index --index-type IVF_FLAT --nlist 128
uv run rag-demo t2 bench --index-type IVF_FLAT --runs 50 --dim 128 --nprobe 8
```

完整 1M 演示：

```bash
uv run rag-demo t2 load --count 1000000 --dim 768 --batch-size 5000 --reset

uv run rag-demo t2 index --index-type IVF_FLAT --nlist 1024 \
  --report .rag/t2/ivf_index.json
uv run rag-demo t2 sweep --index-type IVF_FLAT --values 1,4,8,16,32,64 \
  --runs 1000 --dim 768 --report .rag/t2/ivf_sweep.json

uv run rag-demo t2 index --index-type HNSW --hnsw-m 16 --ef-construction 200 \
  --report .rag/t2/hnsw_index.json
uv run rag-demo t2 sweep --index-type HNSW --values 16,32,64,128,256 \
  --runs 1000 --dim 768 --report .rag/t2/hnsw_sweep.json
```

Attu 地址：

```text
http://localhost:8000
```

eBPF 观察性演示需要安装 `bpftrace` 并使用 root 权限。一个终端跑压测，另一个终端跑：

```bash
sudo bpftrace observability/bpftrace/milvus_io.bt
sudo bpftrace observability/bpftrace/milvus_tcp.bt
sudo bpftrace observability/bpftrace/milvus_syscalls.bt
```

## W4-T3 Qdrant vs Weaviate POC

启动 Qdrant 和 Weaviate：

```bash
docker-compose up -d qdrant weaviate
```

检查环境：

```bash
uv run rag-demo t3 doctor
```

课堂快速验证：

```bash
uv run rag-demo t3 load --backend both --count 10000 --dim 128 --batch-size 1000 --reset
uv run rag-demo t3 compare --runs 50 --dim 128 --top-k 10
uv run rag-demo t3 notes
```

完整 1M POC：

```bash
uv run rag-demo t3 load \
  --backend both \
  --count 1000000 \
  --dim 768 \
  --batch-size 5000 \
  --reset \
  --report .rag/t3/load_report.json

uv run rag-demo t3 compare \
  --runs 1000 \
  --dim 768 \
  --top-k 10 \
  --filtered \
  --report .rag/t3/compare_report.json
```

单库调试：

```bash
uv run rag-demo t3 bench --backend qdrant --runs 1000 --dim 768
uv run rag-demo t3 bench --backend weaviate --runs 1000 --dim 768
```

演示点：

- Qdrant 使用 Rust 内核，HTTP JSON search API，payload filter 采用 `must` 条件。
- Weaviate 使用 schema-first 数据模型，检索通过 GraphQL `nearVector` + `where`。
- 报告输出 QPS、P50/P95/P99、容器内存和 API 类型，便于课堂选型讨论。
- Weaviate 映射到宿主机 `8081`，避免和常见本地 `8080` 服务冲突。
