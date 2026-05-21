# 检索质量评测入门：Recall@K

这篇教程面向已经能跑通 RAG，但不知道如何判断效果好坏的读者。读完后，你应该能回答四个问题：

1. 为什么 RAG 评测要先看检索，而不是只看最终回答？
2. Recall@K 是什么？
3. 查询集和标准答案应该怎么准备？
4. 如何用本项目分析 miss case？

## 1. 为什么先评测检索

RAG 的回答质量通常由两段链路共同决定：

```text
检索阶段：有没有找到正确上下文
生成阶段：LLM 有没有基于上下文正确回答
```

如果第一步没有找回正确资料，第二步很难补救。

例如用户问：

```text
试用期请假超过两天需要谁审批？
```

如果检索结果只包含：

```text
年假余额可以在 HR 系统中查询。
```

即使 LLM 很强，也没有足够依据回答审批流程。

所以调 RAG 时，应该先问：

```text
正确上下文有没有被检索出来？
```

而不是一开始就问：

```text
最终答案写得像不像？
```

## 2. Recall@K 是什么

Recall@K 可以理解成：

```text
在返回的前 K 个结果里，有没有包含应该命中的资料。
```

例如有 10 个测试问题，每个问题都标注了一个正确章节。

如果系统每次返回前 5 个结果，也就是 Top-5：

```text
8 个问题的正确章节出现在 Top-5
2 个问题没有出现
```

那么：

```text
Recall@5 = 8 / 10 = 0.8
```

Recall@K 关注的是“找没找到”，不是“答案写得好不好”。

## 3. K 应该怎么理解

K 是返回结果数量。

| 指标 | 含义 |
| --- | --- |
| Recall@1 | 只看第 1 个结果是否正确 |
| Recall@3 | 看前 3 个结果里是否有正确资料 |
| Recall@5 | 看前 5 个结果里是否有正确资料 |
| Recall@10 | 看前 10 个结果里是否有正确资料 |

K 越大，Recall 通常越高，因为系统有更多机会把正确资料包含进去。

但 K 不能无限大：

- 返回太多 chunk 会占用 LLM 上下文窗口。
- 无关内容变多，可能干扰回答。
- 延迟和成本可能上升。

RAG 应用里常见做法是先看 Recall@5 或 Recall@10，再结合上下文窗口和回答质量决定实际 `top_k`。

## 4. 查询集是什么

查询集是一组用于评测的问题。

每条评测数据至少应该包含：

| 字段 | 含义 |
| --- | --- |
| `query` | 用户可能会问的问题 |
| `expected_section_id` | 应该命中的章节或文档片段 |
| `expected_section_title` | 方便人阅读的正确章节标题 |

本项目 `t4` 的查询集使用 JSONL 格式，每行一条 JSON。

示例：

```json
{"query":"What does the RFC say about flow control?","expected_section_id":"s5","expected_section_title":"5. Flow Control"}
```

JSONL 的好处是：

- 方便追加和编辑。
- 适合版本管理。
- 可以逐行读取，适合大评测集。

## 5. 如何准备查询集

本项目可以从 RFC 或 Markdown 标题自动生成初始查询集：

```bash
uv run rag-demo t4 queries \
  --rfc docs/sample_rfc.md \
  --output .rag/t4/queries.jsonl
```

自动生成只是起点。真实项目里，建议人工补充和修正：

- 加入用户真实问法。
- 加入同义表达，例如“流控”和“flow control”。
- 加入不完整问题，例如“超过两天呢？”。
- 加入容易混淆的问题。
- 标注正确章节或正确 chunk。

好的查询集应该覆盖常见问题，而不是只覆盖文档标题。

## 6. 用本项目跑 Recall@K

最简单的命令：

```bash
uv run rag-demo t4 evaluate
```

它会自动准备示例文档，并比较三种分块策略：

```text
fixed
semantic
parent_child
```

也可以显式传入参数：

```bash
uv run rag-demo t4 evaluate \
  --rfc docs/sample_rfc.md \
  --queries .rag/t4/queries.jsonl \
  --top-k 5 \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --report .rag/t4/recall_report.json
```

评测完成后查看报告：

```bash
uv run rag-demo t4 inspect --report .rag/t4/recall_report.json
```

## 7. 报告怎么看

报告中每个策略会有一组结果：

| 字段 | 含义 |
| --- | --- |
| `strategy` | 分块策略 |
| `chunks` | 生成的 chunk 数量 |
| `indexed_items` | 参与检索的条目数量 |
| `queries` | 评测问题数量 |
| `top_k` | 使用的 K |
| `hits` | 命中正确章节的问题数量 |
| `recall_at_k` | Recall@K 分数 |
| `avg_latency_ms` | 平均检索延迟 |

一个简化例子：

```text
strategy       chunks  queries  hits  recall_at_k
fixed          20      7        5     0.714
semantic       12      7        6     0.857
parent_child   20      7        7     1.000
```

这说明在这组问题上，`parent_child` 的正确章节覆盖率最高。

但不要只看分数。还要看 chunk 数、延迟和 miss case。

## 8. miss case 是什么

miss case 就是“没命中的问题”。

例如：

```text
query: What does the RFC say about packet numbers?
expected: 3. Packet Numbers
retrieved: 2. Streams, 6. Loss Detection, 1. Introduction
```

它比总分更有价值，因为它告诉你系统具体错在哪里。

常见 miss 原因包括：

| 原因 | 现象 |
| --- | --- |
| chunk 切得太碎 | 正确信息被拆散 |
| chunk 太大 | 无关主题干扰相似度 |
| 问法和文档措辞差异大 | 同义词没有匹配上 |
| embedding 模型不适合 | 语义相近内容距离不够近 |
| top_k 太小 | 正确结果排在更后面 |
| 文档结构不清晰 | 标题、段落、列表没有被正确利用 |

调 RAG 时，miss case 通常比成功案例更值得看。

## 9. 单条问题调试

可以用 `t4 ask` 查看某个问题在不同策略下的检索结果：

```bash
uv run rag-demo t4 ask "What does the RFC say about loss detection?" --strategy fixed
uv run rag-demo t4 ask "What does the RFC say about loss detection?" --strategy semantic
uv run rag-demo t4 ask "What does the RFC say about loss detection?" --strategy parent-child
```

重点看：

- 正确章节是否出现在结果里。
- 正确结果排第几。
- 分数是否和错误结果接近。
- preview 是否包含足够回答问题的信息。

如果正确结果在第 6 名，而当前 `top_k=5`，说明可以尝试提高 `top_k`，或者优化 chunking、embedding、query rewrite。

## 10. Recall@K 的局限

Recall@K 很有用，但它不是完整的 RAG 评测。

它不能直接判断：

- LLM 最终答案是否准确。
- 答案是否表达清楚。
- 是否引用了正确来源。
- 是否拒绝回答上下文没有提供的信息。
- 多个 chunk 合起来才能回答的问题是否处理得好。

所以完整评测通常会分层：

```text
检索评测：Recall@K、MRR、nDCG、miss case
生成评测：事实一致性、引用准确性、格式、拒答能力
用户评测：真实任务完成率、人工满意度、反馈数据
```

对新手来说，先把 Recall@K 和 miss case 做扎实，就已经能解决很多 RAG 效果问题。

## 11. 一个实用调试顺序

当 RAG 回答不好时，可以按这个顺序排查：

1. 用 `--show-context` 或 `t4 ask` 看检索上下文。
2. 如果没找到正确资料，先看 Recall@K 和 miss case。
3. 调整 `chunk_size`、`chunk_overlap`、分块策略。
4. 检查 embedding 模型是否适合当前语言和领域。
5. 尝试提高 `top_k`，但注意上下文变长带来的干扰。
6. 检索稳定后，再优化 prompt 和最终回答格式。

这样可以避免把检索问题误判成 prompt 问题。

## 12. 小结

Recall@K 是 RAG 入门阶段最重要的评测指标之一。

它回答的问题很明确：

```text
正确资料有没有进入前 K 个检索结果？
```

如果答案是“没有”，就应该先优化文档、分块、embedding、检索参数或查询改写。

如果答案是“有”，但最终回答仍然不好，再去优化 prompt、上下文组织和 LLM 输出控制。
