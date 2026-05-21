# 从 Markdown 到 RAG 索引

这篇教程面向第一次把文档接入 RAG 的读者。读完后，你应该能回答三个问题：

1. `rag-demo t1 index` 到底做了哪些事情？
2. 为什么 Markdown 文件要先切成 chunk，再写入向量库？
3. 索引目录里的元数据有什么用？

本文结合本项目的 `t1` demo，解释从 Markdown 文档到本地 Chroma 索引的完整链路。

## 1. 先看整体流程

本项目的基础 RAG 索引流程可以理解为：

```text
Markdown 目录
  -> 找到 .md / .markdown 文件
  -> 读取为 Document
  -> 切分成 chunk
  -> 调用 embedding 模型生成向量
  -> 写入 Chroma 向量库
  -> 保存索引元数据
```

对应命令是：

```bash
uv run rag-demo t1 index --docs ./confluence-export --persist ./.rag/index
```

如果有多个 Markdown 目录，可以重复传入 `--docs`：

```bash
uv run rag-demo t1 index \
  --docs ./team-docs \
  --docs ./product-docs \
  --persist ./.rag/team-index
```

## 2. 加载 Markdown 文件

项目只会读取两类文件：

```text
*.md
*.markdown
```

对应代码在 `src/rag_demo2/documents.py`：

```text
discover_markdown_files()
load_markdown_documents()
load_markdown_documents_from_roots()
```

每个文件会被加载成一个 `Document`。新手可以把 `Document` 理解成：

| 字段 | 含义 |
| --- | --- |
| `page_content` | 文件正文 |
| `metadata` | 文件来源、chunk 编号等附加信息 |

项目会把文件路径写入 metadata：

```text
source = /path/to/some-doc.md
```

这个信息很重要。RAG 回答时，用户不仅需要答案，还需要知道答案来自哪里。

## 3. 为什么不能直接索引整篇文档

很多新手会问：为什么不把一整篇 Markdown 直接转成一个向量？

原因主要有三个：

- 文档太长时，embedding 模型和 LLM 都不适合一次处理全部内容。
- 一篇文档里可能有多个主题，整篇生成一个向量会让语义变得很混杂。
- 用户提问通常只需要命中文档中的一小段，而不是整篇文档。

例如一篇员工手册同时包含：

```text
入职流程
请假制度
报销制度
信息安全规范
```

如果用户问“报销发票有什么要求”，检索系统应该优先找到“报销制度”那一段，而不是只知道“员工手册整体相关”。

## 4. chunk 是什么

chunk 就是从原始文档中切出来的小文本片段。

项目默认参数是：

```text
chunk_size = 800
chunk_overlap = 120
```

含义是：

| 参数 | 新手理解 |
| --- | --- |
| `chunk_size` | 每个 chunk 尽量不要超过多少字符 |
| `chunk_overlap` | 相邻 chunk 之间保留多少重复内容 |

为什么需要 overlap？

因为重要信息经常刚好跨过切分边界。例如：

```text
请假超过 2 天需要直属经理审批。
如果是试用期员工，还需要 HRBP 备案。
```

如果两句话被切到不同 chunk，检索或回答时就可能丢失条件。适当 overlap 可以降低这种风险。

## 5. 本项目怎么切分

项目使用 `RecursiveCharacterTextSplitter`，并设置了一组分隔符：

```text
\n## 
\n### 
\n\n
\n
。
.
空格
最后才按字符切
```

这代表它会尽量先按 Markdown 标题、段落、句子切分。只有前面的方式都不合适时，才退回到更细的字符级切分。

这样比简单地每 800 个字符切一刀更适合文档场景，因为标题和段落通常带有语义边界。

## 6. embedding 做了什么

切分完成后，每个 chunk 会交给 embedding 模型，转换成一组数字向量。

```text
chunk 文本 -> embedding 模型 -> [0.12, -0.08, 0.34, ...]
```

本项目默认从配置中读取 Ollama embedding 模型，例如：

```text
qwen3-embedding:latest
```

embedding 的作用不是生成答案，而是让系统可以比较“问题”和“文档片段”的语义距离。

用户提问时，也会先把问题转成向量：

```text
用户问题 -> query vector
```

然后向量库查找最相似的 chunk：

```text
query vector -> Top-K 相似 chunk
```

## 7. Chroma 在这里负责什么

项目的 `t1` demo 使用 Chroma 作为本地向量存储。

它负责保存：

| 内容 | 用途 |
| --- | --- |
| chunk 文本 | 后续作为上下文交给 LLM |
| chunk 向量 | 用于相似度检索 |
| metadata | 用于展示来源和追踪问题 |
| collection | 区分不同索引集合 |

索引持久化目录由 `--persist` 指定：

```bash
uv run rag-demo t1 index --docs ./docs --persist ./.rag/index
```

后续问答时，应用会从同一个目录读取索引：

```bash
uv run rag-demo t1 ask "项目怎么启动？" --persist ./.rag/index
```

## 8. 索引元数据有什么用

项目会在索引目录中保存 `index_meta.json`。它记录：

| 字段 | 含义 |
| --- | --- |
| `docs_dirs` | 索引来自哪些文档目录 |
| `document_count` | 原始文档数量 |
| `chunk_count` | 切分后的 chunk 数量 |
| `chunk_size` | 建索引时使用的 chunk 大小 |
| `chunk_overlap` | 建索引时使用的 overlap |
| `embed_model` | 使用哪个 embedding 模型 |
| `collection_name` | Chroma collection 名称 |

可以用命令查看：

```bash
uv run rag-demo t1 inspect --persist ./.rag/index
```

这些信息可以帮助你回答：

- 这个索引是不是用最新文档构建的？
- 当时使用了哪个 embedding 模型？
- 为什么这次检索结果和上次不一样？
- chunk 数量是否异常偏多或偏少？

## 9. 一次完整练习

准备一个小目录：

```bash
mkdir -p .rag/tutorial-docs
printf '# 请假制度\n\n年假申请需要直属经理审批。试用期员工请假超过 2 天需要 HRBP 备案。\n' > .rag/tutorial-docs/hr.md
```

检查运行环境：

```bash
uv run rag-demo doctor
```

建立索引：

```bash
uv run rag-demo t1 index \
  --docs .rag/tutorial-docs \
  --persist .rag/tutorial-index \
  --reset
```

查看索引信息：

```bash
uv run rag-demo t1 inspect --persist .rag/tutorial-index
```

提问：

```bash
uv run rag-demo t1 ask "试用期请假超过两天需要什么流程？" \
  --persist .rag/tutorial-index \
  --show-context
```

观察输出时，重点看三件事：

- 回答是否只基于文档内容。
- Sources 里是否能看到来源文件和 chunk 编号。
- `--show-context` 展示的上下文是否真的和问题相关。

## 10. 常见问题

### 10.1 文档更新后，索引会自动更新吗

不会。文档变化后，需要重新运行 `t1 index`。

如果想清理旧索引后重建，可以加：

```bash
--reset
```

### 10.2 chunk 越大越好吗

不是。chunk 越大，单个结果包含的信息越多，但也更容易混入无关内容，并占用更多上下文窗口。

chunk 太小也不好，因为它可能丢失完整语义。

实际项目里通常要通过评测找到合适范围，而不是只凭感觉设置。

### 10.3 为什么答案不准确

优先检查检索结果，而不是只改 prompt。

可以使用：

```bash
uv run rag-demo t1 ask "你的问题" --persist ./.rag/index --show-context
```

如果上下文本身就没找到正确资料，LLM 很难生成正确答案。RAG 的质量通常先取决于检索，再取决于生成。

## 11. 小结

从 Markdown 到 RAG 索引，本质上是把“人能读的文档”转换成“应用能按语义检索的知识库”：

```text
文件 -> Document -> Chunk -> Embedding -> Vector Store
```

新手最需要记住的是：

- RAG 索引不是把文档训练进模型。
- chunk 是检索的基本单位。
- embedding 用于语义匹配，不负责回答。
- metadata 决定答案能否追踪来源。
- 文档、切分、embedding、向量库任何一步出问题，都会影响最终回答。
