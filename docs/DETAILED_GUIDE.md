# RAG Demo 2 完整交付文档

本文档是 `rag-demo` 的完整运行和讲解手册，面向团队分享、课堂演示和后续二次开发。

项目目标是用一个可执行的现代 Python CLI，把 RAG 基础、向量数据库压测、向量库选型、分块策略和 Embedding 选型串成 5 个连续 demo。

## 1. 项目概览

`rag-demo` 覆盖 5 个目标：

| 目标 | 主题 | 交付命令组 | 关键学习点 |
| --- | --- | --- | --- |
| W4-T1 | LangChain RAG Hello World | `rag-demo t1` | Loader、Splitter、Retriever、LCEL、RAG 问答 |
| W4-T2 | Milvus 单机 + 1M 向量压测 | `rag-demo t2` | IVF_FLAT、HNSW、nprobe/ef、Attu、eBPF |
| W4-T3 | Qdrant vs Weaviate POC | `rag-demo t3` | 过滤检索、GraphQL、Rust 内核、内存/延迟对比 |
| W4-T4 | 分块策略对比 | `rag-demo t4` | 固定分块、语义分块、父子分块、Recall@5 |
| W4-T5 | Embedding 选型 | `rag-demo t5` | BGE-M3、OpenAI text-embedding-3、多语言、维度、成本 |

项目强调可交付：

- 每个目标都有可运行 CLI。
- 默认提供 smoke test 路径，课堂上可以快速跑通。
- 重型实验支持扩大到 1M 向量或真实模型。
- 报告以 JSON 保存，便于复盘和横向对比。

## 2. 技术栈

| 分类 | 技术 |
| --- | --- |
| 语言 | Python 3.10+ |
| 包管理 | uv |
| CLI | Typer + Rich |
| RAG 框架 | LangChain |
| 本地向量索引 | Chroma |
| 模型服务 | Ollama、OpenAI Embeddings API |
| 向量数据库 | Milvus、Qdrant、Weaviate |
| 容器编排 | Docker Compose |
| 观察性 | bpftrace/eBPF |
| 测试 | pytest |
| Lint | Ruff |

默认远端 Ollama：

```text
http://192.168.1.18:11434
```

默认模型：

```text
Chat model: deepseek-v4-pro:cloud
Embedding model: qwen3-embedding:latest
```

W4-T5 真实 embedding 对比使用：

```text
BGE-M3: bge-m3:latest via Ollama
OpenAI: text-embedding-3-small / text-embedding-3-large
```

## 3. 仓库结构

```text
rag-demo/
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── DETAILED_GUIDE.md
│   ├── sample_rfc.md
│   └── sample_embedding_eval.jsonl
├── observability/
│   └── bpftrace/
│       ├── milvus_io.bt
│       ├── milvus_syscalls.bt
│       └── milvus_tcp.bt
├── src/
│   └── rag_demo/
│       ├── cli.py
│       ├── config.py
│       ├── documents.py
│       ├── rag.py
│       ├── milvus_bench.py
│       ├── qdrant_weaviate.py
│       ├── chunking_recall.py
│       ├── embedding_selection.py
│       └── ollama.py
└── tests/
    ├── test_documents.py
    ├── test_rag_metadata.py
    ├── test_milvus_bench.py
    ├── test_qdrant_weaviate.py
    ├── test_chunking_recall.py
    └── test_embedding_selection.py
```

核心模块职责：

- `cli.py`: Typer 命令入口，组织 `t1` 到 `t5` 命令组。
- `config.py`: 模型和 Ollama 地址默认配置。
- `documents.py`: Markdown 发现、加载、切分。
- `rag.py`: T1 的 Chroma 索引和 LCEL 问答链。
- `milvus_bench.py`: T2 的 Milvus 数据生成、导入、建索引、压测。
- `qdrant_weaviate.py`: T3 的 Qdrant/Weaviate 加载、检索和对比。
- `chunking_recall.py`: T4 的分块策略、检索、Recall@K 评测。
- `embedding_selection.py`: T5 的 embedding 模型评测、缓存、成本和推荐。

## 4. Git Worktree 组织

项目按目标使用独立分支和 worktree，便于教学时展示每一步演进：

| 目标 | 分支 | Worktree |
| --- | --- | --- |
| W4-T1 | `w4-t1-langchain-rag` | `../rag-demo-w4-t1` |
| W4-T2 | `w4-t2-milvus-1m-benchmark` | `../rag-demo-w4-t2` |
| W4-T3 | `w4-t3-qdrant-weaviate-poc` | `../rag-demo-w4-t3` |
| W4-T4 | `w4-t4-chunking-recall` | `../rag-demo-w4-t4` |
| W4-T5 | `w4-t5-embedding-selection` | `../rag-demo-w4-t5` |

查看 worktree：

```bash
git worktree list
```

进入最终完整版本：

```bash
cd /home/ubuntu/workspace/easylearning/rag-demo-w4-t5
```

查看提交链：

```bash
git log --oneline --decorate -10
```

## 5. 安装与基础检查

安装依赖：

```bash
uv sync
```

检查 CLI：

```bash
uv run rag-demo --help
uv run rag-demo doctor
```

默认配置来自：

```text
src/rag_demo/config.py
```

也可以用环境变量覆盖：

```bash
export OLLAMA_HOST=http://192.168.1.18:11434
export RAG_DEMO_CHAT_MODEL=deepseek-v4-pro:cloud
export RAG_DEMO_EMBED_MODEL=qwen3-embedding:latest
```

开发验证：

```bash
uv run ruff check .
uv run pytest
```

## 6. 推荐演示路线

建议分享时按下面顺序执行，时间约 60 到 90 分钟。

| 阶段 | 内容 | 建议耗时 |
| --- | --- | --- |
| 1 | 项目目标、CLI 总览、worktree 组织 | 5 分钟 |
| 2 | T1 Markdown RAG 问答 | 15 分钟 |
| 3 | T2 Milvus 单机压测和 eBPF 观察 | 15 分钟 |
| 4 | T3 Qdrant vs Weaviate 横向对比 | 15 分钟 |
| 5 | T4 分块策略 Recall@5 | 15 分钟 |
| 6 | T5 Embedding 选型和成本讨论 | 15 分钟 |

最小 smoke 路线：

```bash
uv run rag-demo doctor
uv run rag-demo t4 evaluate
uv run rag-demo t5 evaluate
```

完整环境路线：

```bash
docker-compose up -d
uv run rag-demo t1 index --docs <markdown-dir> --persist .rag/index --reset
uv run rag-demo t1 ask "你的问题" --persist .rag/index
uv run rag-demo t2 load --count 1000000 --dim 768 --batch-size 5000 --reset
uv run rag-demo t2 sweep --index-type IVF_FLAT --values 1,4,8,16,32,64 --runs 1000
uv run rag-demo t3 compare --runs 1000 --dim 768 --filtered
uv run rag-demo t4 evaluate
uv run rag-demo t5 evaluate --real --models bge-m3
```

## 7. W4-T1: LangChain RAG Hello World

### 7.1 目标

把团队 Confluence Markdown 导出数据做成问答机器人。

关键链路：

```text
Markdown files -> Loader -> Splitter -> Embedding -> Chroma -> Retriever -> LCEL QA chain
```

### 7.2 测试数据

用户提供的测试目录：

```text
/home/ubuntu/workspace/easylearning/ez-cli/docs/beyond-vibe-coding
/home/ubuntu/workspace/easylearning/ez-cli/docs/beyond-vibe-coding-cn
```

### 7.3 建索引

```bash
uv run rag-demo t1 index \
  --docs /home/ubuntu/workspace/easylearning/ez-cli/docs/beyond-vibe-coding \
  --docs /home/ubuntu/workspace/easylearning/ez-cli/docs/beyond-vibe-coding-cn \
  --persist .rag/beyond-vibe \
  --reset
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--docs` | 必填 | Markdown 目录，可重复传入 |
| `--persist` | `.rag/index` | Chroma 持久化目录 |
| `--chunk-size` | `800` | chunk 大小 |
| `--chunk-overlap` | `120` | chunk 重叠 |
| `--collection` | `rag_demo_t1` | Chroma collection 名称 |
| `--reset` | `False` | 删除旧索引后重建 |

### 7.4 一次性问答

```bash
uv run rag-demo t1 ask \
  "Beyond Vibe Coding 主要解决什么问题？" \
  --persist .rag/beyond-vibe \
  --top-k 4
```

显示上下文：

```bash
uv run rag-demo t1 ask \
  "项目中如何判断 AI 生成代码是否可交付？" \
  --persist .rag/beyond-vibe \
  --show-context
```

### 7.5 交互式问答

```bash
uv run rag-demo t1 chat --persist .rag/beyond-vibe
```

退出：

```text
/exit
```

### 7.6 查看索引元数据

```bash
uv run rag-demo t1 inspect --persist .rag/beyond-vibe
```

输出关注：

- `document_count`: 加载文档数。
- `chunk_count`: 切分后 chunk 数。
- `embed_model`: 索引用的 embedding 模型。
- `collection_name`: Chroma collection。

### 7.7 讲解点

- Loader 只负责把文件变成 `Document`，不解决语义粒度。
- Splitter 的 chunk 边界会直接影响检索上下文质量。
- Retriever 的 `top_k` 是召回和噪音之间的权衡。
- LCEL 让 prompt、model、parser 形成清晰流水线。
- 生产环境需要记录索引元数据，否则换 embedding 模型后容易误用旧索引。

## 8. W4-T2: Milvus 单机 + 1M 向量压测

### 8.1 目标

把 Milvus Standalone 跑起来，导入最多 1M 条向量，对 IVF_FLAT 和 HNSW 做压测，并用 eBPF 观察系统行为。

### 8.2 启动服务

```bash
docker-compose up -d
```

包含服务：

- Milvus Standalone
- etcd
- MinIO
- Attu
- Qdrant
- Weaviate

T2 主要使用 Milvus、etcd、MinIO、Attu。

检查：

```bash
uv run rag-demo t2 doctor
```

Attu：

```text
http://localhost:8000
```

### 8.3 生成数据

默认生成查询集和元数据，不落盘 1M 全量向量：

```bash
uv run rag-demo t2 generate \
  --output .rag/t2 \
  --count 1000000 \
  --dim 768 \
  --query-count 1000
```

如果需要生成完整 `.npy`：

```bash
uv run rag-demo t2 generate \
  --output .rag/t2 \
  --count 1000000 \
  --dim 768 \
  --materialize
```

说明：1M x 768 float32 约 3GB，课堂演示通常不需要 materialize。

### 8.4 快速 smoke test

```bash
uv run rag-demo t2 load --count 10000 --dim 128 --batch-size 1000 --reset
uv run rag-demo t2 index --index-type IVF_FLAT --nlist 128
uv run rag-demo t2 bench --index-type IVF_FLAT --runs 50 --dim 128 --nprobe 8
```

### 8.5 1M 导入

```bash
uv run rag-demo t2 load \
  --count 1000000 \
  --dim 768 \
  --batch-size 5000 \
  --reset \
  --report .rag/t2/load_report.json
```

导入策略：

- 使用确定性 seed 流式生成向量。
- 按 batch 插入，避免 Python 进程一次性持有全部向量。
- 每条向量带有简单 payload 字段，便于后续过滤/分组讨论。

### 8.6 IVF_FLAT 压测

建索引：

```bash
uv run rag-demo t2 index \
  --index-type IVF_FLAT \
  --nlist 1024 \
  --report .rag/t2/ivf_index.json
```

单次参数压测：

```bash
uv run rag-demo t2 bench \
  --index-type IVF_FLAT \
  --runs 1000 \
  --dim 768 \
  --nprobe 16 \
  --report .rag/t2/ivf_bench.json
```

扫描 `nprobe`：

```bash
uv run rag-demo t2 sweep \
  --index-type IVF_FLAT \
  --values 1,4,8,16,32,64 \
  --runs 1000 \
  --dim 768 \
  --report .rag/t2/ivf_sweep.json
```

讲解点：

- `nlist`: 建索引时划分的倒排分区数量。
- `nprobe`: 查询时扫描多少个分区。
- `nprobe` 越大，通常召回更高但延迟更高。
- 对 1M 数据，`nlist` 和 `nprobe` 不存在通用最优值，需要用业务数据压测。

### 8.7 HNSW 压测

建索引：

```bash
uv run rag-demo t2 index \
  --index-type HNSW \
  --hnsw-m 16 \
  --ef-construction 200 \
  --report .rag/t2/hnsw_index.json
```

扫描 `ef`：

```bash
uv run rag-demo t2 sweep \
  --index-type HNSW \
  --values 16,32,64,128,256 \
  --runs 1000 \
  --dim 768 \
  --report .rag/t2/hnsw_sweep.json
```

讲解点：

- HNSW 是图索引，通常低延迟表现好。
- `M` 影响图连接密度和内存。
- `efConstruction` 影响构建质量和构建耗时。
- 查询时的 `ef` 影响搜索范围、召回和延迟。

### 8.8 eBPF 观察性

一个终端跑压测，另一个终端运行：

```bash
sudo observability/observe_vector_db.sh cpu 30
sudo observability/observe_vector_db.sh offcpu 30
sudo observability/observe_vector_db.sh io 30
sudo observability/observe_vector_db.sh syscalls 30
```

脚本用途：

| 脚本 | 观察内容 |
| --- | --- |
| `milvus_io.bt` | block I/O 延迟分布 |
| `milvus_tcp.bt` | Milvus TCP retransmit/reset |
| `milvus_syscalls.bt` | Milvus raw syscall 延迟分布 |
| `vector_db_cpu.bt` | Milvus/Qdrant/Weaviate CPU kernel stack 采样 |
| `vector_db_offcpu.bt` | Milvus/Qdrant/Weaviate off-CPU 等待时间 |
| `vector_db_page_faults.bt` | Milvus/Qdrant/Weaviate page fault |
| `vector_db_tcp_retrans.bt` | Milvus/Qdrant/Weaviate TCP retransmit |
| `vector_db_vfs_latency.bt` | page cache/writeback 活动 |

统一入口：

```bash
observability/observe_vector_db.sh --help
```

详细说明见 `observability/README.md`。

讲解点：

- 压测不只看 P95，也要看系统瓶颈来自 CPU、I/O、网络还是内存。
- eBPF 可以低侵入地观察内核层行为。
- 课堂中不需要把 eBPF 做成完整监控系统，重点是展示“压测指标背后的系统现象”。

## 9. W4-T3: Qdrant vs Weaviate 选型 POC

### 9.1 目标

用同一批数据对比 Qdrant 和 Weaviate 的检索延迟、内存占用和开发体验。

### 9.2 启动服务

```bash
docker-compose up -d qdrant weaviate
```

检查：

```bash
uv run rag-demo t3 doctor
```

默认端口：

| 服务 | 地址 |
| --- | --- |
| Qdrant | `http://localhost:6333` |
| Weaviate | `http://localhost:8081` |

Weaviate 使用 `8081` 是为了避免和常见本地 `8080` 服务冲突。

### 9.3 快速 smoke test

```bash
uv run rag-demo t3 load \
  --backend both \
  --count 10000 \
  --dim 128 \
  --batch-size 1000 \
  --reset

uv run rag-demo t3 compare \
  --runs 50 \
  --dim 128 \
  --top-k 10
```

### 9.4 1M POC

导入：

```bash
uv run rag-demo t3 load \
  --backend both \
  --count 1000000 \
  --dim 768 \
  --batch-size 5000 \
  --reset \
  --report .rag/t3/load_report.json
```

对比：

```bash
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

查看开发体验说明：

```bash
uv run rag-demo t3 notes
```

### 9.5 对比维度

| 维度 | Qdrant | Weaviate |
| --- | --- | --- |
| 核心实现 | Rust | Go |
| API 风格 | HTTP JSON search/query | GraphQL + REST |
| Payload 过滤 | `filter.must` | GraphQL `where` |
| Schema | collection + payload | schema-first class |
| 向量化 | 本 demo 手动传 vector | `vectorizer: none` |
| 讨论重点 | 低延迟、过滤检索直接 | GraphQL 表达力、schema 建模 |

### 9.6 讲解点

- 选型不能只看单次延迟，还要看导入、过滤、schema、运维和团队熟悉度。
- 同一批 seed 生成的数据保证实验可复现。
- Payload filter 是 RAG 生产检索中的高频能力，例如按租户、业务线、时间、权限过滤。
- GraphQL API 在复杂查询表达上有优势，但也会带来学习和调试成本。

## 10. W4-T4: 分块策略 Recall@5

### 10.1 目标

对同一份 RFC 文档使用不同分块策略，观察对检索召回率的影响。

策略：

- Fixed: 固定长度滑动窗口。
- Semantic: 按段落相邻语义相似度合并。
- Parent-Child: 检索 child chunk，返回 parent section。

### 10.2 默认评测

```bash
uv run rag-demo t4 evaluate
```

默认使用：

```text
docs/sample_rfc.md
```

输出：

```text
.rag/t4/recall_report.json
```

### 10.3 生成查询集

```bash
uv run rag-demo t4 queries \
  --rfc docs/sample_rfc.md \
  --output .rag/t4/queries.jsonl
```

JSONL 每行代表一个查询样本，包含问题和预期章节。可以人工修改问题，让它更接近真实用户问法。

### 10.4 使用真实 RFC

```bash
uv run rag-demo t4 queries \
  --rfc ./docs/rfc9000.md \
  --output .rag/t4/rfc9000_queries.jsonl

uv run rag-demo t4 evaluate \
  --rfc ./docs/rfc9000.md \
  --queries .rag/t4/rfc9000_queries.jsonl \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --top-k 5
```

### 10.5 对单条问题看检索结果

```bash
uv run rag-demo t4 ask \
  "What does the RFC say about flow control?" \
  --strategy fixed \
  --top-k 5

uv run rag-demo t4 ask \
  "What does the RFC say about flow control?" \
  --strategy semantic \
  --top-k 5

uv run rag-demo t4 ask \
  "What does the RFC say about flow control?" \
  --strategy parent-child \
  --top-k 5
```

### 10.6 评测指标

Recall@5 的含义：

```text
如果预期章节出现在 Top 5 检索结果中，则记为 hit。
Recall@5 = hit 数 / query 数
```

MRR 没有放在 T4 默认输出中，因为 T4 更关注不同 chunking 策略是否能召回正确章节；排序质量可以作为后续扩展。

### 10.7 讲解点

- chunk 太小：容易丢上下文，LLM 回答需要额外拼接。
- chunk 太大：检索噪音变多，embedding 表达被多个主题稀释。
- overlap 可以缓解边界切断，但会增加索引体积。
- 语义分块适合结构明显的文档，但实现和评测成本更高。
- 父子分块很适合 RAG：用小 chunk 做精确匹配，用大 parent 给 LLM 上下文。

## 11. W4-T5: Embedding 选型

### 11.1 目标

选出团队性价比最高的 Embedding 模型，重点比较：

- 多语言能力。
- 向量维度。
- API 成本。
- 向量存储成本。
- 是否可本地部署。
- 团队真实 QA 上的检索效果。

### 11.2 默认 smoke test

默认不访问外部服务，使用 deterministic hashing embedding 跑完整流程：

```bash
uv run rag-demo t5 evaluate
```

输出：

```text
.rag/t5/embedding_report.json
```

这个模式适合：

- 课堂现场先确认 CLI 和评测链路可用。
- CI 或离线环境测试。
- 讲解指标字段和报告结构。

注意：offline hashing 不是模型质量评测，只是流程 smoke test。

### 11.3 生成团队 QA 数据集

```bash
uv run rag-demo t5 sample-data --output .rag/t5/team_qa.jsonl
```

数据格式：

```json
{
  "id": "team-milvus",
  "query": "Milvus 压测里 nprobe 影响什么？",
  "positive": "在 IVF_FLAT 索引中，nprobe 控制查询时扫描的倒排分区数量，通常会影响召回、延迟和 QPS。",
  "negatives": [
    "GraphQL API 是 Weaviate 在 POC 中的主要开发体验差异点。",
    "BGE-M3 可以通过 Ollama 在本地或内网机器上部署。"
  ],
  "language": "zh",
  "source": "team-qa"
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `id` | 样本唯一 ID |
| `query` | 用户问题 |
| `positive` | 正确应召回文本 |
| `negatives` | 干扰文本 |
| `language` | 语言标记，如 `zh`、`en` |
| `source` | 数据来源，如 `mteb-subset`、`team-qa` |

### 11.4 真实 BGE-M3 评测

```bash
uv run rag-demo t5 evaluate \
  --real \
  --models bge-m3 \
  --ollama-base-url http://192.168.1.18:11434 \
  --report .rag/t5/bge_m3_report.json
```

### 11.5 真实 OpenAI 对比

需要环境变量：

```bash
export OPENAI_API_KEY=...
```

执行：

```bash
uv run rag-demo t5 evaluate \
  --real \
  --dataset .rag/t5/team_qa.jsonl \
  --models bge-m3,text-embedding-3-small,text-embedding-3-large \
  --ollama-base-url http://192.168.1.18:11434 \
  --report .rag/t5/embedding_report.json
```

### 11.6 查看报告

```bash
uv run rag-demo t5 inspect --report .rag/t5/embedding_report.json
```

报告字段：

| 字段 | 说明 |
| --- | --- |
| `recall_at_1` | 正样本是否排第 1 |
| `recall_at_3` | 正样本是否进 Top 3 |
| `recall_at_5` | 正样本是否进 Top 5 |
| `mrr` | Mean Reciprocal Rank |
| `avg_latency_ms` | 平均 embedding 延迟 |
| `estimated_tokens` | 估算输入 token 数 |
| `estimated_cost_usd` | 估算 API 成本 |
| `vector_memory_mb` | 当前 corpus 的向量内存估算 |
| `local_deployable` | 是否可本地/内网部署 |
| `unavailable` | 模型或服务是否不可用 |

### 11.7 当前模型元信息

| 模型 | Provider | 默认维度 | API token 成本 | 可本地部署 | 多语言 |
| --- | --- | ---: | ---: | --- | --- |
| `bge-m3` | Ollama | 1024 | $0 | 是 | 是 |
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 / 1M tokens | 否 | 是 |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 / 1M tokens | 否 | 是 |

说明：OpenAI 价格和模型维度可能变化，生产选型前应以 OpenAI 官方文档和价格页为准。

### 11.8 推荐决策方式

建议不要只按 Recall 排名选型，而是按下面顺序做决策：

1. 先用团队真实 QA 跑 Recall@K 和 MRR。
2. 如果 BGE-M3 与最佳 API 模型差距在 3% 以内，优先考虑 BGE-M3 作为团队默认模型。
3. 如果 OpenAI large 明显领先，再评估数据出域、延迟、成本和预算。
4. 如果中文/英文表现差异明显，按业务语言拆分模型策略。
5. 最终在真实 RAG 链路中做 end-to-end answer quality 评测，而不是只看 embedding retrieval。

## 12. Docker Compose 服务说明

启动全部服务：

```bash
docker-compose up -d
```

查看：

```bash
docker-compose ps
```

停止：

```bash
docker-compose down
```

清理 volume：

```bash
docker-compose down -v
```

服务用途：

| 服务 | 用途 |
| --- | --- |
| `milvus` | T2 Milvus 压测 |
| `etcd` | Milvus metadata |
| `minio` | Milvus object storage |
| `attu` | Milvus UI |
| `qdrant` | T3 Qdrant POC |
| `weaviate` | T3 Weaviate POC |

## 13. 输出文件和缓存

默认输出目录：

```text
.rag/
```

常见文件：

| 路径 | 来源 | 说明 |
| --- | --- | --- |
| `.rag/index/` | T1 | Chroma RAG 索引 |
| `.rag/t2/*.json` | T2 | Milvus 压测报告 |
| `.rag/t2/queries.npy` | T2 | 查询向量 |
| `.rag/t3/*.json` | T3 | Qdrant/Weaviate POC 报告 |
| `.rag/t4/recall_report.json` | T4 | 分块 Recall 报告 |
| `.rag/t5/embedding_report.json` | T5 | Embedding 选型报告 |
| `.rag/t5/cache/` | T5 | Embedding 结果缓存 |

清理本地实验输出：

```bash
rm -rf .rag
```

## 14. 测试与质量门禁

运行全部测试：

```bash
uv run pytest
```

运行 lint：

```bash
uv run ruff check .
```

建议提交前执行：

```bash
uv run ruff check .
uv run pytest
git status --short
```

当前测试覆盖：

- Markdown discovery 和 split 参数校验。
- RAG 索引元数据兼容。
- Milvus 数据生成和参数逻辑。
- Qdrant/Weaviate POC 配置和报告逻辑。
- T4 分块策略和 Recall 评测。
- T5 embedding dataset、评测和报告 roundtrip。

## 15. 常见故障排查

### 15.1 `rag-demo doctor` 显示 Ollama server unavailable

检查地址：

```bash
echo $OLLAMA_HOST
curl http://192.168.1.18:11434/api/tags
```

如果使用本地 Ollama：

```bash
export OLLAMA_HOST=http://localhost:11434
```

### 15.2 T1 问答提示 embedding 模型缺失

检查 Ollama 模型：

```bash
ollama list
```

需要：

```text
qwen3-embedding:latest
deepseek-v4-pro:cloud
```

### 15.3 OpenAI embedding 评测不可用

确认：

```bash
echo $OPENAI_API_KEY
```

只跑 BGE-M3：

```bash
uv run rag-demo t5 evaluate --real --models bge-m3
```

或者跑离线 smoke：

```bash
uv run rag-demo t5 evaluate
```

### 15.4 Docker 服务端口冲突

检查端口：

```bash
docker-compose ps
ss -ltnp | rg '6333|8081|19530|8000'
```

Weaviate 默认映射到 `8081`，不是 `8080`。

### 15.5 Milvus 导入慢或内存高

降低 batch size：

```bash
uv run rag-demo t2 load --count 1000000 --dim 768 --batch-size 1000 --reset
```

先跑小数据：

```bash
uv run rag-demo t2 load --count 10000 --dim 128 --reset
```

### 15.6 eBPF 脚本无法运行

确认权限和安装：

```bash
which bpftrace
sudo bpftrace --version
```

eBPF 需要 root 权限和支持的 Linux kernel。

## 16. 生产化改造建议

本项目是教学 demo，不是完整生产系统。生产化时建议补齐：

- 权限过滤：检索时按用户、团队、文档 ACL 做 payload filter。
- 增量索引：按文件 hash 或更新时间增量更新。
- 索引版本：记录 embedding 模型、维度、chunk 策略、代码版本。
- 离线评测集：维护团队真实 QA golden set。
- Answer 评测：除了 Recall@K，还要评估最终回答正确性、引用质量和拒答能力。
- 监控：记录检索延迟、LLM 延迟、token 成本、空回答率、用户反馈。
- 数据治理：处理 PII、密级、数据出域和模型供应商合规。
- 部署：将 CLI 中的核心逻辑拆为服务 API 或任务队列。

## 17. 讲师速查命令

```bash
# 基础
uv sync
uv run rag-demo doctor

# T1
uv run rag-demo t1 index --docs ./confluence-export --persist .rag/index --reset
uv run rag-demo t1 ask "团队文档里最重要的约定是什么？" --persist .rag/index
uv run rag-demo t1 chat --persist .rag/index

# T2
docker-compose up -d
uv run rag-demo t2 doctor
uv run rag-demo t2 load --count 10000 --dim 128 --reset
uv run rag-demo t2 index --index-type IVF_FLAT --nlist 128
uv run rag-demo t2 bench --index-type IVF_FLAT --runs 50 --dim 128 --nprobe 8

# T3
uv run rag-demo t3 load --backend both --count 10000 --dim 128 --batch-size 1000 --reset
uv run rag-demo t3 compare --runs 50 --dim 128 --top-k 10
uv run rag-demo t3 notes

# T4
uv run rag-demo t4 evaluate
uv run rag-demo t4 ask "What does the RFC say about flow control?" --strategy parent-child

# T5
uv run rag-demo t5 evaluate
uv run rag-demo t5 evaluate --real --models bge-m3 --ollama-base-url http://192.168.1.18:11434
```

## 18. 当前验收状态

最近一次 W4-T5 交付验证：

```bash
uv run ruff check .
uv run pytest
uv run rag-demo t5 evaluate --report .rag/t5/smoke_embedding_report.json
uv run rag-demo t5 evaluate --real --models bge-m3 --ollama-base-url http://192.168.1.18:11434
```

验证结果：

- Ruff: passed
- Pytest: 23 passed
- T5 offline smoke: passed
- T5 real BGE-M3 smoke: passed
