# 本地优先代码库 RAG / MCP 工具详细对比

调研日期：2026-05-21

对比对象：

- [ogrep](https://ogrep.be/)
- [Semble](https://github.com/MinishLab/semble)
- [flupkede/codesearch](https://github.com/flupkede/codesearch)
- [Octocode](https://github.com/bgauryy/octocode-mcp)
- [project-rag](https://github.com/Brainwires/project-rag)

本文重点比较“隐私优先、尽量本地运行、给 AI 编程工具提供代码库上下文”的使用场景。需要注意：这 5 个项目都能用于本地代码上下文增强，但“本地优先”的程度不同，尤其 Octocode 同时覆盖 GitHub/GitLab 远程代码研究，不是单纯离线 RAG 工具。

## 一句话定位

| 项目 | 定位 |
| --- | --- |
| ogrep | Python 生态的本地语义 grep：SQLite 单文件索引、AST chunking、MCP + CLI，适合快速落地。 |
| Semble | 面向 coding agent 的轻量代码搜索层：强调 CPU 本地搜索、少读文件、少耗 token。 |
| flupkede/codesearch | Rust 实现的多仓库离线混合检索 MCP：向量 + BM25 + Tree-sitter + 多 repo 路由。 |
| Octocode | 综合型代码研究 MCP：本地代码、GitHub/GitLab、LSP 导航、PR 考古、包发现和 agent skills。 |
| project-rag | 较完整的 Rust RAG / 代码导航 MCP：FastEmbed + LanceDB、混合检索、定义/引用/调用图。 |

## 总览对比

| 维度 | ogrep | Semble | flupkede/codesearch | Octocode | project-rag |
| --- | --- | --- | --- | --- | --- |
| 主要形态 | CLI + MCP | CLI + MCP + Python 库 | MCP + CLI/TUI 倾向 | MCP + CLI 安装器 + skills | MCP server |
| 本地优先程度 | 高 | 高 | 很高 | 中等 | 高 |
| 完全离线潜力 | 取决于 embedding 配置，本地 embedding 可离线 | 高，强调 CPU 本地和无 API key | 很高，README 明确 no GPU / no Docker / offline | 较弱，GitHub/GitLab 能力通常需要认证和网络；本地工具可本地用 | 高，默认 LanceDB 嵌入式；可选 Qdrant 需要服务 |
| 索引存储 | `.ogrep/index.sqlite` | 本地缓存/索引，支持本地路径和 Git URL | LMDB + arroy + Tantivy | 项目更偏工具集合，含本地搜索和 LSP | 默认 `.lancedb`；可选 Qdrant |
| 检索方式 | embedding 检索、fulltext、可 rerank | semantic search、find-related，强调 agent chunk 返回 | 向量 + BM25 混合检索 | 代码搜索、文件搜索、LSP semantic navigation、PR/包搜索 | 向量 + BM25 混合检索、过滤检索 |
| AST / 结构能力 | Tree-sitter AST chunking，默认启用条件是安装 AST extras | 返回代码 chunk，公开资料更强调搜索体验 | Tree-sitter AST-aware chunking，多语言 fallback | LSP 导航、AST + dependency graph skills | AST-based chunking、definition/reference/call graph |
| 多仓库 | 单 repo 使用最自然 | 可搜本地 repo 和 Git URL | 强项：multi-repo routing / groups | 强项：本地 + GitHub/GitLab 多代码源 | 支持 multi-project |
| AI 工具接入 | MCP tools、Claude Code skill、CLI | MCP、Bash/AGENTS.md、子 agent 初始化 | MCP client，如 OpenCode、Claude Code、Cursor | Cursor、Claude、VS Code 和 MCP-compatible assistant | Claude Code / Claude Desktop MCP 配置 |
| 安装复杂度 | 低，`pip install "ogrep[ast]"` | 低，`uvx --from "semble[mcp]" semble` 或安装 CLI | 中，Rust binary / release | 低到中，`npx octocode-cli install`，但认证配置更多 | 中，需要 Rust 1.88+ 和 protobuf compiler |
| 适合规模 | 小到中型 repo；轻量、单文件索引方便 | 小到中大型 repo 的 agent 检索；强调快速搜索 | 多 repo / 中大型 repo | 跨本地和远程代码研究，复杂研发流程 | 中大型 repo，愿意使用较完整 MCP RAG 服务 |
| 许可证 | MIT | MIT | Apache-2.0 | MIT | MIT |

## 项目细节

### ogrep

核心特点：

- 把仓库切 chunk、生成 embedding，存入单个本地 SQLite 索引。
- MCP server 提供 `ogrep_query`、`ogrep_chunk`、`ogrep_index`、`ogrep_status`、`ogrep_health` 等工具。
- AST chunking 已作为推荐路径，安装 `ogrep[ast]` 后可按代码结构切分。
- 支持 fulltext 模式，适合查精确 identifier。
- 本地索引默认在 `.ogrep/index.sqlite`，迁移、清理和按 repo 隔离都比较简单。

优势：

- 上手成本低，Python 团队容易维护。
- SQLite 单文件索引非常适合本地优先和 per-repo 工作流。
- 同时支持 CLI 和 MCP，既能人工搜索，也能给 agent 调工具。
- MCP 常驻进程能避免每次 shell 调用都冷启动 Python、加载模型和打开 SQLite。

局限：

- 如果使用 OpenAI、Voyage 等云 embedding，源码片段或查询会离开本地；要离线需要配置本地 embedding。
- 更偏“语义 grep + chunk 读取”，不是完整代码图谱或编译级分析工具。
- 多仓库、跨服务关系和调用图不是它的主要卖点。

适合：

- 希望一天内把现有 repo 接给 Claude Code / Cursor / Codex 类工具。
- 团队能接受 Python 工具链。
- 对本地索引简单性、可解释性和低运维要求高。

不适合：

- 需要跨几十个仓库做统一路由。
- 需要严肃的 definition/reference/call graph 级分析。

### Semble

核心特点：

- 面向 AI coding agent 的代码搜索工具，支持 MCP、CLI 和 Python library。
- 典型命令包括 `semble search` 和 `semble find-related`。
- 支持搜索本地路径，也支持 Git URL 按需使用。
- README 明确建议 agent 先用 Semble 搜 chunk，只有 chunk 不够时再读完整文件。
- 提供 Bash / AGENTS.md 集成，适合 MCP 不可用的 sub-agent 场景。
- 有 token savings 统计，用文件全文读取与返回 snippet 的差额估算节省量。

优势：

- 很贴合 agent 工作流：先检索相关 chunk，再按需读文件。
- 同时覆盖 MCP 和 Bash 调用，对 Codex/Claude/Gemini/OpenCode/Cursor 这类工具比较友好。
- 对“子 agent 不能直接调用 MCP”的情况给了 Bash 集成路径。
- 支持 `find-related`，便于从一个命中点继续扩展上下文。

局限：

- 公开文档更强调搜索体验和 benchmark，底层索引结构、语言级符号能力没有 codesearch/project-rag 那么直观。
- 更像“高质量代码搜索层”，不是完整代码知识图谱。
- 对精确符号引用、调用图、跨语言静态分析的能力需要实测验证。

适合：

- 想减少 agent 到处 `grep + read` 带来的 token 浪费。
- 需要同时支持 MCP 和 shell/AGENTS.md 的团队。
- 想在本地和远程 Git repo 上快速做自然语言代码搜索。

不适合：

- 主要需求是编译级导航、调用图、跨服务依赖分析。
- 需要自己强控制底层向量库和索引格式。

### flupkede/codesearch

核心特点：

- Rust 实现，多仓库语义代码搜索 MCP server。
- 强调 fully local、fully offline、no GPU、no Docker。
- 使用 fastembed + ONNX Runtime 在 CPU 上生成 embedding。
- 向量索引用 arroy + LMDB，全文检索用 Tantivy BM25。
- Tree-sitter AST-aware chunking，支持 Rust、Python、JavaScript、TypeScript、C/C++、C#、Go、Java；其他文本文件走 line-based fallback。
- 增量同步使用 SHA-256 内容 hash。
- 支持多 repo routing / groups、outline、similar、get_chunk、status 等工具。
- 对 C# 有更深的 semantic search / impact 功能，但需要对应 helper。

优势：

- 这 5 个里“离线优先”表达最明确的之一。
- Rust + LMDB + Tantivy + arroy 的组合适合本地长期运行。
- 多仓库是核心设计，不是附加能力。
- 混合检索比单纯向量搜索更适合代码场景，精确 identifier 和自然语言查询都能覆盖。
- Tree-sitter chunking 能避免简单按行切分破坏函数/类边界。

局限：

- 项目相对年轻，生态和文档成熟度需要观察。
- 深层符号/影响分析目前看对 C# 最突出，其他语言更多依赖 AST chunking 和通用检索。
- Rust 工具链和二进制安装对部分团队比 Python/npm 稍重。

适合：

- 多仓库、本地离线、无 GPU、无 Docker 的代码搜索 MCP。
- 希望代码检索工具长期作为本地 agent 基础设施运行。
- 希望同时要 BM25、向量、Tree-sitter 和 multi-repo routing。

不适合：

- 只想要最轻量、最少概念的单仓库语义 grep。
- 需要非常成熟的大社区生态或企业支持。

### Octocode

核心特点：

- 定位是“code research MCP”，不是单纯本地 RAG。
- 覆盖 GitHub、GitLab、本地代码、LSP semantic navigation、PR archaeology、package discovery。
- 提供 13 个 MCP tools、内置 slash commands 和 skills。
- 支持 Cursor、Claude、VS Code 以及任意 MCP-compatible assistant。
- 本地侧能力包括 search code、browse directories、find files，以及 LSP 的 go-to-definition、find-references、call hierarchy。
- 安装器 `npx octocode-cli install` 会处理 MCP server 和 skills 配置，但通常涉及 GitHub authentication。

优势：

- 能力面最广：本地代码 + 远程仓库 + PR 历史 + 包生态。
- LSP 导航对 AI 编程很重要，很多 RAG 工具只能返回 chunk，不能可靠做 definition/reference。
- skills 和 prompts 做得比较完整，适合把“研究、计划、生成、review”变成统一工作流。
- 对已有 GitHub/GitLab 研发流程的团队，价值不只在本地检索。

局限：

- “本地优先”程度弱于 ogrep / Semble / codesearch / project-rag；它明显包含远程代码研究能力。
- 如果目标是源码绝不离开机器，需要仔细关闭或避免 GitHub/GitLab 相关能力，并审计配置。
- 功能面广也意味着心智负担更高，不是一个小而专的代码 RAG。

适合：

- 团队不只想搜本地 repo，还想让 agent 查 GitHub/GitLab、PR、包、外部实现。
- 需要 LSP 级导航和更完整的 agent research workflow。
- 可以接受配置认证和较复杂的工具集。

不适合：

- 严格离线环境。
- 只想给一个本地 repo 加一个简单、可控、可审计的语义索引。

### project-rag

核心特点：

- Rust MCP server，目标是给 AI assistant 理解大型代码库。
- 默认使用 LanceDB 嵌入式向量数据库，索引落在 `./.lancedb`；可选 Qdrant 后端。
- FastEmbed 做本地 embedding。
- 支持 hybrid search：LanceDB vector + Tantivy BM25 + Reciprocal Rank Fusion。
- 支持 `index_codebase`、`query_codebase`、`get_statistics`、`clear_index`、`search_by_filters`，以及 definition、references、call graph 等代码导航工具。
- 支持自动全量/增量索引、`.gitignore`、SHA256 变更检测、多项目。
- 支持 40+ 文件类型识别，还包含 PDF 到 Markdown 的处理能力。

优势：

- 比 ogrep/Semble 更接近完整 RAG 服务，索引、过滤、统计、代码导航能力都比较完整。
- 默认 LanceDB 嵌入式，不需要外部向量数据库服务。
- 本地 FastEmbed + LanceDB 对离线场景友好。
- definition/reference/call graph 对 AI 修改大型代码库很有价值。
- 多项目和并发访问保护说明它考虑了长期运行场景。

局限：

- 安装门槛比 pip/uvx/npx 工具更高：需要 Rust 1.88+ 和 protobuf compiler。
- 如果启用 Qdrant 后端，就不再是零服务依赖。
- 功能较多，调试和理解成本比轻量工具高。

适合：

- 想搭一个偏正式的本地代码 RAG MCP server。
- 需要混合检索、过滤、多项目、definition/reference/call graph。
- 团队能接受 Rust 构建和较重的本地服务。

不适合：

- 只是想临时搜索一个 repo。
- 不想引入 LanceDB/向量库概念或额外构建依赖。

## 选型建议

### 只想快速给 AI 编程工具加“语义搜代码”

优先：

1. Semble
2. ogrep

理由：安装简单，CLI/MCP 都自然，agent 使用方式清晰。Semble 更强调 agent 工作流和 token 节省；ogrep 更强调 SQLite 单文件索引和语义 grep 的透明性。

### 严格本地、离线、多仓库

优先：

1. flupkede/codesearch
2. project-rag

理由：codesearch 明确强调 fully offline、no GPU、no Docker，并且多仓库是核心能力。project-rag 功能更重，适合需要完整 MCP RAG 服务和代码导航的团队。

### 需要 definition / references / call graph

优先：

1. project-rag
2. Octocode
3. flupkede/codesearch

理由：project-rag 明确提供 definition、references、call graph 工具。Octocode 有 LSP semantic navigation，但它不是纯本地离线方案。codesearch 有 outline、similar、get_chunk，以及 C# 方向的 impact/reference 能力。

### 想要“代码研究工作流”，不局限于本地 repo

优先：

1. Octocode

理由：Octocode 的重点是综合 code research：本地代码、GitHub/GitLab、PR、包、LSP、skills 和 slash commands。它更像给 AI 助手加一套高级研究工具，而不是只加一个向量索引。

### 想最小运维和最容易回滚

优先：

1. ogrep
2. Semble

理由：ogrep 的 SQLite 单文件索引容易理解和删除；Semble 的 CLI/uvx 集成也很轻。两者都比搭建较完整的 Rust MCP RAG 服务更省心。

## 推荐实测方法

建议选同一个真实代码库，分别测试 4 类问题：

| 测试类型 | 示例问题 | 观察点 |
| --- | --- | --- |
| 概念定位 | “登录后的权限检查在哪里？” | 能否命中正确模块和关键函数 |
| 精确符号 | “`validateToken` 的调用方有哪些？” | 是否支持 exact search、reference 或 call graph |
| 相关实现 | “和这个缓存写入逻辑相似的代码还有哪里？” | 是否支持 similar / find-related |
| 变更后更新 | 修改一个文件后立即搜索新逻辑 | 增量索引是否稳定、MCP 是否能看到新内容 |

建议记录：

- 首次索引耗时
- 增量索引耗时
- 查询延迟
- Top 5 命中率
- 返回内容是否刚好够 agent 修改代码
- 是否需要外部 API
- 是否会把源码或查询发送到外部服务
- MCP 工具输出是否结构化、可控、不会过度占用上下文

## 简短结论

- **个人/小团队最快落地**：Semble 或 ogrep。
- **最强调离线和多仓库**：flupkede/codesearch。
- **最像完整本地代码 RAG 服务**：project-rag。
- **最适合代码研究和复杂 agent workflow**：Octocode，但它不是最纯粹的本地离线方案。
- **严格隐私环境**：优先 codesearch / project-rag / ogrep，并确认 embedding 使用本地模型；谨慎使用 Octocode 的远程仓库能力。

## 资料来源

- ogrep 官网：https://ogrep.be/
- Semble GitHub：https://github.com/MinishLab/semble
- flupkede/codesearch GitHub：https://github.com/flupkede/codesearch
- Octocode GitHub：https://github.com/bgauryy/octocode-mcp
- Octocode 官网：https://octocode.ai/
- project-rag GitHub：https://github.com/Brainwires/project-rag
