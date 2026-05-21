# RAG 基础 & 向量数据库 Demo 开发计划

## 项目目标

构建一个 Python + uv 管理的现代化 CLI demo，用 5 个连续目标帮助团队学习 RAG 基础、向量数据库压测、选型、分块策略和 Embedding 评测。

默认模型后端为远端 Ollama `http://192.168.1.18:11434`；演示数据由用户导入 Markdown 目录。

## 分支与 worktree

主仓库使用 `main` 保存项目骨架、文档和公共约定。每个目标使用独立分支与 sibling worktree 开发：

| 目标 | 分支 | Worktree |
| --- | --- | --- |
| W4-T1 | `w4-t1-langchain-rag` | `../rag-demo-w4-t1` |
| W4-T2 | `w4-t2-milvus-1m-benchmark` | `../rag-demo-w4-t2` |
| W4-T3 | `w4-t3-qdrant-weaviate-poc` | `../rag-demo-w4-t3` |
| W4-T4 | `w4-t4-chunking-recall` | `../rag-demo-w4-t4` |
| W4-T5 | `w4-t5-embedding-selection` | `../rag-demo-w4-t5` |

## W4-T1: LangChain RAG Hello World

交付一个可运行 CLI：

- `rag-demo doctor`: 检查 Ollama、模型和运行环境。
- `rag-demo t1 index`: 加载一个或多个 Markdown 目录，切分 chunk，写入本地 Chroma 索引。
- `rag-demo t1 ask`: 基于索引执行一次性 RAG 问答。
- `rag-demo t1 chat`: 交互式问答。
- `rag-demo t1 inspect`: 查看索引元数据。

关键点：

- Loader: 使用 LangChain `TextLoader` 递归读取 `.md` / `.markdown`。
- Splitter: 使用 `RecursiveCharacterTextSplitter`，默认 `chunk_size=800`、`chunk_overlap=120`。
- Retriever: 使用 Chroma 本地持久化向量库。
- LCEL: 使用 `ChatPromptTemplate | ChatOllama | StrOutputParser` 组合问答链。
- 默认模型: `deepseek-v4-pro:cloud` 用于问答，`qwen3-embedding:latest` 用于 embedding。

验收：

- 可以索引约 100 篇 Markdown。
- 可以返回答案和来源文件。
- 空目录、索引不存在、Ollama 不可用时给出清晰提示。

## W4-T2: Milvus 单机 + 1M 向量压测

已交付：

- `docker-compose up -d` 启动 Milvus Standalone + Attu。
- `rag-demo t2 generate`: 生成确定性查询集和可选全量 `.npy` 向量文件。
- `rag-demo t2 load`: 按 batch 流式生成并导入 1M 向量，避免一次性占用内存。
- `rag-demo t2 index`: 对 IVF_FLAT 和 HNSW 建索引。
- `rag-demo t2 bench`: 执行固定次数检索，输出 QPS、P50/P95/P99、RSS 和行数。
- `rag-demo t2 sweep`: 扫描 IVF_FLAT `nprobe` 或 HNSW `ef`。
- `rag-demo t2 observe`: 输出 bpftrace 观察脚本运行方式。

观察性：

- `observability/bpftrace/milvus_io.bt`: block I/O 延迟分布。
- `observability/bpftrace/milvus_tcp.bt`: Milvus TCP 收发。
- `observability/bpftrace/milvus_syscalls.bt`: syscall 延迟分布。

说明：当前环境使用 `docker-compose` 命令；若本地 Docker 支持 Compose plugin，也可使用 `docker compose`。

## W4-T3: Qdrant vs Weaviate 选型 POC

已交付：

- `docker-compose up -d qdrant weaviate` 启动两个候选向量库。
- `rag-demo t3 doctor` 检查 Docker、Qdrant、Weaviate 和 GraphQL 端点。
- `rag-demo t3 load` 用同一 seed 流式生成 1M 向量，并写入 Qdrant 与 Weaviate。
- `rag-demo t3 bench` 对单个库执行检索压测。
- `rag-demo t3 compare` 对两个库各跑 1000 次同等过滤检索，输出 QPS、P50/P95/P99、内存。
- `rag-demo t3 notes` 展示 Rust 内核、GraphQL API、Payload/where 过滤和开发体验差异。

关键点：

- Qdrant：Rust 内核，JSON search API，payload filter 使用 `must` 条件。
- Weaviate：schema-first，GraphQL `nearVector` + `where`，显式 `vectorizer: none`。
- 数据集：确定性合成向量，按 batch 生成，避免一次性在 Python 进程中持有 1M x 768。

验收：

- 可用小数据快速 smoke test。
- 可扩展到 1M 条向量、1000 次检索。
- 产出 JSON 报告，用于课堂横向选型讨论。

## W4-T4: 分块策略对比

计划交付：

- 固定长度、语义切块、父子切块三种策略。
- 对同一份 RFC 建索引并执行固定查询集。
- 计算 Recall@5。
- 输出策略对比表和错误样例。

## W4-T5: Embedding 选型

计划交付：

- 对 BGE-M3 与 OpenAI `text-embedding-3` 系列进行对比。
- 覆盖多语言、维度、成本、部署方式。
- 使用 MTEB 子集和团队真实 QA 对评测。
- 输出推荐结论和适用场景。
