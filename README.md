# RAG Demo 2

一个面向分享和学习的 RAG / 向量数据库 CLI demo。项目按 5 个连续目标推进：

1. W4-T1: LangChain / LlamaIndex RAG Hello World
2. W4-T2: Milvus 单机 + 1M 向量压测
3. W4-T3: Qdrant vs Weaviate 选型 POC
4. W4-T4: 分块策略对比
5. W4-T5: Embedding 选型

当前已实现 W4-T1：用本地 Ollama 对 Markdown 目录建立 RAG 索引，并提供一次性问答和交互式问答。

## Quick Start

安装依赖：

```bash
uv sync
```

准备 Ollama 模型：

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

检查环境：

```bash
uv run rag-demo doctor
```

索引 Markdown 目录：

```bash
uv run rag-demo t1 index --docs ./confluence-export --persist ./.rag/index
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
