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

## Quick Start

安装依赖：

```bash
uv sync
```

默认代码会使用本机 `http://localhost:11434`、`llama3.1` 和 `nomic-embed-text`。课堂环境使用远端 Ollama：

```bash
export OLLAMA_HOST=http://192.168.1.18:11434
export RAG_DEMO_CHAT_MODEL=deepseek-v4-pro:cloud
export RAG_DEMO_EMBED_MODEL=qwen3-embedding:latest
```

检查环境：

```bash
uv run rag-demo doctor
```

如果没有设置 `RAG_DEMO_EMBED_MODEL`，索引会使用默认的 `nomic-embed-text`。远端 Ollama 没有该模型时会报 `model "nomic-embed-text" not found`。

索引 main 分支的一本 ebook：

```bash
uv run rag-demo t1 index \
  --docs ../rag-demo/ebooks/beyond-vibe-coding-cn \
  --persist ./.rag/ebooks \
  --reset
```

问答：

```bash
uv run rag-demo t1 ask "70% 问题指的是什么？最后 30% 为什么仍然需要人类工程判断？" --persist ./.rag/ebooks
```

交互式问答：

```bash
uv run rag-demo t1 chat --persist ./.rag/ebooks
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
