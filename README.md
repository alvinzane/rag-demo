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
- W4-T4：对同一份 RFC 对比固定、语义、父子分块策略，并计算 Recall@5。

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
uv run rag-demo t1 ask "什么是 Vibe 编程？它和 AI 辅助工程有什么区别？" --persist ./.rag/ebooks
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

## W4-T4 Chunking Recall

默认命令会自动生成一份小型 RFC demo 文档，便于课堂快速跑通：

```bash
uv run rag-demo t4 evaluate
```

输出三种策略的 chunk 数、平均检索延迟和 Recall@5，并写入：

```text
.rag/t4/recall_report.json
```

使用 main 分支的一本 ebook 做真实文档评测。`t4 evaluate` 的 `--rfc` 接受单个 Markdown/text 文件，先把 ebook 目录按路径排序后合并：

```bash
mkdir -p .rag/t4
find ../rag-demo/ebooks/beyond-vibe-coding-cn \
  -type f -name '*.md' | sort | while IFS= read -r path; do
    printf '\n\n<!-- source: %s -->\n\n' "$path"
    sed 's/[[:space:]]*$//' "$path"
  done > .rag/t4/ebooks.md
```

使用合并后的 ebook：

```bash
uv run rag-demo t4 evaluate \
  --rfc .rag/t4/ebooks.md \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --top-k 5
```

人工维护一份查询集，`expected_section_id` 来自上面确定性合并后的标题顺序：

```bash
mkdir -p .rag/t4
cat > .rag/t4/ebook_queries.jsonl <<'JSONL'
{"query":"什么是 Vibe 编程？它和 AI 辅助工程有什么区别？","expected_section_id":"s0","expected_section_title":"第1章 简介：什么是Vibe编程？"}
{"query":"70% 问题指的是什么？最后 30% 为什么仍然需要人类工程判断？","expected_section_id":"s13","expected_section_title":"第3章 70%问题：真正有效的AI辅助工作流程"}
{"query":"AI 生成代码常见的安全风险有哪些？应该如何审查？","expected_section_id":"s117","expected_section_title":"第8章 安全性、可维护性和可靠性"}
JSONL

uv run rag-demo t4 evaluate --rfc .rag/t4/ebooks.md --queries .rag/t4/ebook_queries.jsonl
```

查看单条问题在不同策略下的检索结果：

```bash
uv run rag-demo t4 ask "70% 问题指的是什么？最后 30% 为什么仍然需要人类工程判断？" --rfc .rag/t4/ebooks.md --strategy fixed
uv run rag-demo t4 ask "70% 问题指的是什么？最后 30% 为什么仍然需要人类工程判断？" --rfc .rag/t4/ebooks.md --strategy semantic
uv run rag-demo t4 ask "70% 问题指的是什么？最后 30% 为什么仍然需要人类工程判断？" --rfc .rag/t4/ebooks.md --strategy parent-child
```

演示点：

- 固定分块：直接观察 `chunk_size` / `chunk_overlap` 对召回和 chunk 数的影响。
- 语义分块：按段落相邻相似度合并，展示 chunk 边界与语义完整性的关系。
- 父子分块：检索 child chunk，但按 parent section 去重返回，模拟 Parent-Child Retriever。
