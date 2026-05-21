# 开源代码库 RAG / MCP / CLI 项目调研

调研日期：2026-05-21

本文汇总一批可以把代码库作为 RAG 上下文，并通过 MCP 工具或 CLI 工具提供给 AI 编程工具使用的开源项目。这里的“代码库 RAG”按宽口径理解：包括向量检索、混合检索、AST/符号索引、调用图、代码图谱和面向 agent 的上下文工具。

## 快速结论

优先试用顺序：

1. **cocoindex-code**：轻量、CLI + MCP 接入清晰，适合快速验证。
2. **ogrep**：本地优先、SQLite 索引、AST chunking，适合小团队和个人工作流。
3. **Octocode**：功能较完整，包含 Tree-sitter、LanceDB、语义搜索和 GraphRAG 思路。
4. **codebase-memory-mcp / Code Pathfinder / project-rag**：更偏符号、调用图或结构化代码理解。
5. **Qdrant / pgvector 方案**：适合已经有向量数据库基础设施的团队。

## 候选项目

| 项目 | 形态 | 主要能力 | 适合场景 |
| --- | --- | --- | --- |
| [Octocode](https://octomind.run/product/octocode) | MCP + CLI | Rust、Tree-sitter、LanceDB、语义搜索、代码签名、GraphRAG | 想要较完整的本地代码库索引和 MCP 接入 |
| [cocoindex-code](https://cocoindex.io/cocoindex-code/) | CLI + MCP | AST 代码搜索、`ccc mcp`、可接 Claude Code / Codex CLI / OpenCode / Cursor | 快速把代码搜索能力接给 AI 编程工具 |
| [ogrep](https://ogrep.be/) | CLI + MCP | 本地优先、SQLite 索引、AST chunking、自动增量刷新 | 轻量本地代码检索 |
| [Semble](https://mcp.so/server/semble/Minish) | MCP | CPU 本地运行、语义 + BM25 混合检索、无 API key | 不想依赖外部模型服务或云 API |
| [flupkede/codesearch](https://github.com/flupkede/codesearch) | CLI + MCP | Rust、多 repo、向量 + BM25、Tree-sitter chunking、offline-first | 多仓库语义搜索和本地 agent 工作流 |
| [DeusData/codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp) | MCP + CLI | Tree-sitter、函数/类/调用链/路由/跨服务链接 | 需要代码知识图谱或结构化上下文 |
| [Brainwires/project-rag](https://github.com/Brainwires/project-rag) | MCP | Rust、FastEmbed、LanceDB、混合检索、git history、definition/reference | 需要结合代码和 Git 历史的项目 RAG |
| [aanno/cocoindex-code-mcp-server](https://github.com/aanno/cocoindex-code-mcp-server) | MCP | PostgreSQL + pgvector、CocoIndex 增量索引、Tree-sitter chunking | 已经使用 PostgreSQL / pgvector 的团队 |
| [Code Pathfinder](https://codepathfinder.dev/mcp) | MCP / CLI | AST 分析、调用图、符号搜索、import/dataflow | 偏程序分析、调用路径和符号定位 |
| [er77/code-graph-rag-mcp](https://github.com/er77/code-graph-rag-mcp) | MCP | 代码图 + 语义搜索，可接 Claude Code / Gemini CLI / Codex CLI | 需要代码图谱和 agent 工具集成 |
| [GitNexus](https://github.com/nxpatterns/gitnexus) | CLI + MCP | `gitnexus analyze` 建索引，生成 agent 上下文配置 | 希望围绕 Git 仓库生成上下文和 agent 配置 |
| [KIRI](https://github.com/CAPHTECH/kiri) | MCP | Git repo 上下文抽取、DuckDB 索引、语义搜索 | 轻量 repo 上下文索引 |
| [opencode-codebase-index](https://github.com/Helweg/opencode-codebase-index) | CLI + MCP | 语义搜索、peek、similar、call_graph | OpenCode 用户，也可通过 MCP 给其他工具使用 |
| [ancoleman/qdrant-rag-mcp](https://github.com/ancoleman/qdrant-rag-mcp) | MCP | Qdrant 向量库、代码库语义搜索、GitHub issue/项目工具 | 已有 Qdrant 或希望集中管理向量索引 |
| [mhalder/qdrant-mcp-server](https://github.com/mhalder/qdrant-mcp-server) | MCP | Qdrant、embedding、AST-aware chunking、多语言、增量索引 | 多语言代码库 + Qdrant |
| [autodev-codebase](https://github.com/anrgct/autodev-codebase) | CLI + MCP | Qdrant、Tree-sitter、支持 Ollama/OpenAI/Jina/Gemini embedding | 想灵活替换 embedding provider |
| [semcode](https://mcpservers.org/servers/goodbyeplanet/semcode) | MCP | GitHub repo、Tree-sitter symbol、dense embedding + BM25、commit history | 需要索引 GitHub 仓库和提交历史 |

## 按接入方式分类

### MCP 优先

适合 Cursor、Claude Code、Codex CLI、OpenCode、Windsurf 等支持 MCP 的 AI 编程工具。

- Octocode
- cocoindex-code
- ogrep
- Semble
- codebase-memory-mcp
- project-rag
- Code Pathfinder
- code-graph-rag-mcp
- KIRI
- Qdrant 相关 MCP server

### CLI 优先

适合放进脚本、CI、pre-commit、agent shell tool 或自定义编程助手。

- cocoindex-code
- ogrep
- flupkede/codesearch
- GitNexus
- opencode-codebase-index
- autodev-codebase

## 按技术路线分类

### 向量 / 混合检索

这类项目更接近传统 RAG：对代码分块、生成 embedding、写入本地或外部向量库，再通过语义搜索返回相关代码片段。

- Octocode：LanceDB
- project-rag：FastEmbed + LanceDB
- Qdrant 相关项目：Qdrant
- cocoindex-code-mcp-server：PostgreSQL + pgvector
- Semble / codesearch / semcode：语义 + BM25 混合检索

### AST / 符号 / 代码图

这类项目更适合 AI 编程，因为“找函数定义、引用、调用路径、类关系”通常比单纯 chunk 相似度更可靠。

- codebase-memory-mcp
- Code Pathfinder
- code-graph-rag-mcp
- opencode-codebase-index
- GitNexus
- ogrep

### 本地优先

适合隐私要求高、不能上传源码、或希望离线运行的团队。

- ogrep
- Semble
- flupkede/codesearch
- Octocode
- project-rag

## 选型建议

如果目标是“给 AI 编程工具增加代码库上下文”，不要只看 embedding 检索效果。建议同时评估：

- **增量索引**：代码改动后是否能快速更新索引。
- **符号能力**：是否能找 definition、references、call graph、imports。
- **混合检索**：是否同时支持关键词、BM25、向量和结构化过滤。
- **MCP 工具设计**：工具返回内容是否短、准、可控，避免一次塞太多上下文。
- **本地运行成本**：是否需要 Docker、数据库、GPU、外部 API。
- **多语言支持**：Tree-sitter grammar 覆盖是否满足团队代码栈。
- **与现有工具集成**：是否能直接接 Claude Code、Codex CLI、Cursor、OpenCode、Windsurf。

## 推荐验证路径

可以用同一个中等规模代码库做 3 组 smoke test：

1. **语义问题**：例如“用户登录后的权限检查在哪里做？”
2. **符号问题**：例如“这个函数有哪些调用方？”
3. **变更问题**：修改一个文件后，看索引是否能增量更新并返回新内容。

验证指标：

- Top 5 是否命中正确文件或函数。
- 返回片段是否足够完整，能让 AI 继续修改代码。
- 索引耗时、更新耗时和本地资源占用。
- MCP 工具返回是否稳定，是否容易超过上下文预算。

## 与本项目的关系

`rag-demo` 当前主要演示文档 RAG、向量数据库压测、分块策略和 embedding 选型。上述项目可以作为后续扩展方向：

- 在 W4-T4 的分块策略基础上加入代码 AST chunking。
- 在 W4-T5 的 embedding 评测中增加代码检索数据集。
- 新增一个目标：对同一代码库比较“关键词检索、向量检索、AST/符号检索、混合检索”的效果。
- 增加 MCP server demo，把本项目的检索能力暴露给 AI 编程工具。
