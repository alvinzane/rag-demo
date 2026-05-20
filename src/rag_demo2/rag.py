from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_demo2.config import DEFAULT_COLLECTION, ModelConfig
from rag_demo2.documents import load_markdown_documents_from_roots, split_documents

META_FILE = "index_meta.json"


@dataclass(frozen=True)
class IndexMetadata:
    docs_dir: str
    docs_dirs: list[str]
    document_count: int
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    embed_model: str
    collection_name: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[dict[str, object]]
    contexts: list[str]


def embedding_function(config: ModelConfig) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.embed_model, base_url=config.base_url)


def vectorstore(
    persist_dir: Path,
    config: ModelConfig,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embedding_function(config),
    )


def build_index(
    docs_dirs: list[Path],
    persist_dir: Path,
    config: ModelConfig,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> IndexMetadata:
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = load_markdown_documents_from_roots(docs_dirs)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function(config),
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    metadata = IndexMetadata(
        docs_dir=str(docs_dirs[0]),
        docs_dirs=[str(path) for path in docs_dirs],
        document_count=len(docs),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=config.embed_model,
        collection_name=collection_name,
    )
    write_metadata(persist_dir, metadata)
    return metadata


def read_metadata(persist_dir: Path) -> IndexMetadata:
    meta_path = persist_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"Index metadata not found: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if "docs_dirs" not in payload:
        payload["docs_dirs"] = [payload["docs_dir"]]
    return IndexMetadata(**payload)


def write_metadata(persist_dir: Path, metadata: IndexMetadata) -> None:
    (persist_dir / META_FILE).write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_context(docs: list[Document], max_chars: int = 1200) -> str:
    blocks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "?")
        content = doc.page_content[:max_chars]
        blocks.append(f"[source={source} chunk={chunk_id}]\n{content}")
    return "\n\n".join(blocks)


def ask_question(
    question: str,
    persist_dir: Path,
    config: ModelConfig,
    top_k: int = 4,
    collection_name: str | None = None,
) -> Answer:
    metadata = read_metadata(persist_dir)
    store = vectorstore(persist_dir, config, collection_name or metadata.collection_name)
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    prompt = ChatPromptTemplate.from_template(
        """你是团队知识库问答助手。请只根据给定上下文回答问题。
如果上下文不足以回答，请直接说“不确定，当前索引中没有足够信息”。
回答要简洁、可交付，并尽量列出关键步骤或结论。

上下文:
{context}

问题:
{question}
"""
    )
    llm = ChatOllama(model=config.chat_model, base_url=config.base_url, temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": format_context(docs), "question": question})

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "?"),
        }
        for doc in docs
    ]
    return Answer(text=answer, sources=sources, contexts=[doc.page_content for doc in docs])
