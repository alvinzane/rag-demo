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

配置 Ollama。默认会使用本机 `http://localhost:11434`、`llama3.1` 和 `nomic-embed-text`；如果使用课堂远端 Ollama，先设置：

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
