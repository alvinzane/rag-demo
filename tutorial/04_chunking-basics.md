# Chunking 分块策略基础

这篇教程面向刚开始调 RAG 效果的读者。读完后，你应该能回答四个问题：

1. 为什么 RAG 需要把文档切成 chunk？
2. `chunk_size` 和 `chunk_overlap` 分别影响什么？
3. 固定分块、语义分块、父子分块有什么区别？
4. 如何用本项目观察不同分块策略的检索效果？

## 1. chunk 是 RAG 的检索单位

在 RAG 中，用户提问后，系统通常不会把整篇文档都交给 LLM，而是先检索出几个最相关的文本片段。

这些文本片段就是 chunk。

```text
原始文档
  -> chunk 1
  -> chunk 2
  -> chunk 3
  -> ...
```

向量数据库里存的通常不是“整篇文档的一个向量”，而是“很多 chunk 的向量”。

用户问题也会变成向量，然后和 chunk 向量做相似度检索：

```text
问题向量 -> 找到最相似的 Top-K chunk -> 交给 LLM 生成回答
```

所以 chunk 切得好不好，会直接影响 RAG 能不能找到正确上下文。

## 2. chunk 太小会怎样

chunk 太小，容易丢失上下文。

例如原文是：

```text
试用期员工请假超过 2 天，需要直属经理审批，并由 HRBP 备案。
```

如果切得太碎，可能变成：

```text
chunk 1: 试用期员工请假超过 2 天
chunk 2: 需要直属经理审批，并由 HRBP 备案
```

用户问：

```text
试用期请假超过两天需要谁处理？
```

单独命中 `chunk 1` 时，没有审批信息；单独命中 `chunk 2` 时，又缺少“试用期”和“超过 2 天”的条件。

这会让 LLM 拿到不完整上下文。

## 3. chunk 太大会怎样

chunk 太大，也会带来问题。

例如一个 chunk 同时包含：

```text
入职流程
请假制度
报销制度
信息安全规范
```

用户问“报销发票要求”时，这个大 chunk 可能相关，但里面有大量无关内容。

后果是：

- 检索排序可能不稳定。
- LLM 需要阅读更多无关上下文。
- prompt 占用更多 token。
- 多个大 chunk 很快塞满上下文窗口。

好的 chunk 应该尽量保持“语义完整，但不过度混杂”。

## 4. chunk_size 和 chunk_overlap

最常见的两个参数是：

| 参数 | 含义 | 调大后的影响 |
| --- | --- | --- |
| `chunk_size` | 每个 chunk 的目标大小 | 上下文更完整，但更容易混入无关内容 |
| `chunk_overlap` | 相邻 chunk 的重复部分 | 减少边界丢失，但会增加 chunk 数和存储量 |

本项目 `t1 index` 默认使用：

```text
chunk_size = 800
chunk_overlap = 120
```

这不是所有场景的最优值，只是一个适合入门 demo 的起点。

更稳妥的做法是：用评测数据比较不同配置，而不是只看单个问题的回答感觉。

## 5. 固定分块

固定分块是最容易理解的策略：

```text
每隔一段长度切一次，相邻片段保留一点 overlap。
```

优点：

- 简单稳定。
- 速度快。
- 参数容易解释。
- 适合先跑通 RAG 流程。

缺点：

- 可能切断句子、段落或表格。
- 不理解标题和语义边界。
- 对结构复杂文档不够友好。

本项目中的 `fixed` 策略就是这个思路。

## 6. 语义分块

语义分块不是只按长度切，而是尽量把语义接近的段落放在一起。

可以先这样理解：

```text
相邻段落意思接近 -> 放进同一个 chunk
相邻段落意思变化明显 -> 切开
```

本项目 `semantic` 策略会计算相邻段落的相似度，并使用 `semantic_threshold` 判断是否合并。

优点：

- 更尊重段落和主题边界。
- 对说明文档、规范文档更自然。
- 有机会减少“一个 chunk 里混太多主题”的问题。

缺点：

- 实现和调参更复杂。
- 对 embedding 质量更敏感。
- 文档段落本身很乱时，效果也会受影响。

## 7. 父子分块

父子分块适合理解成“两层文本”：

```text
parent: 一个较大的章节或段落
child: 从 parent 中切出来的小片段
```

检索时用 child，因为小片段更容易精确命中问题。

返回时按 parent 去重或补充上下文，因为 parent 能提供更完整语义。

```text
用户问题
  -> 检索 child chunk
  -> 找到命中的 parent section
  -> 返回 parent 级别结果
```

优点：

- 兼顾精确检索和完整上下文。
- 适合章节结构明显的文档。
- 可以减少同一章节多个 child 重复占据 Top-K 的情况。

缺点：

- 数据结构更复杂。
- parent 太大时仍然会浪费上下文。
- 需要保存 child 和 parent 的映射关系。

本项目的 `parent_child` 策略会检索 child，但返回时按 section 去重。

## 8. 用本项目跑一次对比

默认命令会自动生成一份小型 RFC 示例文档：

```bash
uv run rag-demo t4 evaluate
```

输出会包含三种策略：

```text
fixed
semantic
parent_child
```

每种策略会展示：

| 指标 | 含义 |
| --- | --- |
| chunks | 生成了多少 chunk |
| queries | 用多少个问题做评测 |
| hits | 有多少问题命中了正确章节 |
| recall_at_k | Recall@K 分数 |
| avg_latency_ms | 平均检索耗时 |

报告会写入：

```text
.rag/t4/recall_report.json
```

可以再次查看：

```bash
uv run rag-demo t4 inspect --report .rag/t4/recall_report.json
```

## 9. 查看单个问题的检索结果

只看总分还不够。调 RAG 时，经常需要看某个问题到底检索到了什么。

例如：

```bash
uv run rag-demo t4 ask "What does the RFC say about flow control?" --strategy fixed
uv run rag-demo t4 ask "What does the RFC say about flow control?" --strategy semantic
uv run rag-demo t4 ask "What does the RFC say about flow control?" --strategy parent-child
```

重点观察：

- 正确章节有没有出现在 Top-K 里。
- 排名是否靠前。
- preview 是否包含回答问题需要的信息。
- 错误命中的章节和问题是否有相似词干扰。

## 10. 调参时看什么

可以尝试修改：

```bash
uv run rag-demo t4 evaluate --chunk-size 400 --chunk-overlap 80
uv run rag-demo t4 evaluate --chunk-size 1200 --chunk-overlap 160
uv run rag-demo t4 evaluate --semantic-threshold 0.35
uv run rag-demo t4 evaluate --semantic-threshold 0.55
```

调参时不要只追求一个指标最高。建议同时看：

| 观察项 | 为什么重要 |
| --- | --- |
| Recall@K | 正确上下文有没有被找回来 |
| chunk 数量 | 影响存储、索引和成本 |
| 平均延迟 | 影响用户体验 |
| miss case | 帮你理解为什么没命中 |
| preview 内容 | 判断 chunk 是否语义完整 |

## 11. 常见误区

### 11.1 chunk_size 有通用最佳值吗

没有。它取决于文档类型、问题类型、embedding 模型、top_k、LLM 上下文窗口和业务容错要求。

制度文档、API 文档、会议纪要、代码文档适合的切分方式可能完全不同。

### 11.2 overlap 越大越安全吗

不是。overlap 太大会造成大量重复 chunk。

重复 chunk 可能让 Top-K 被相似内容占满，反而挤掉其他有用信息。

### 11.3 分块策略能解决所有 RAG 问题吗

不能。分块只解决“文档如何进入检索系统”的一部分问题。

RAG 质量还会受到这些因素影响：

- 原始文档是否准确。
- embedding 模型是否适合当前语言和领域。
- 查询是否需要改写。
- 是否需要 metadata filter。
- prompt 是否明确要求基于上下文回答。
- LLM 是否有足够能力理解上下文。

## 12. 小结

chunking 是 RAG 中最容易被低估的一步。

可以先记住这个判断标准：

```text
好的 chunk 应该让检索器容易命中，也让 LLM 容易读懂。
```

固定分块适合入门和基线评测；语义分块更关注主题边界；父子分块尝试兼顾精确检索和完整上下文。

实际项目里，最好用 Recall@K 和 miss case 来评估分块策略，而不是只凭几次聊天结果做判断。
