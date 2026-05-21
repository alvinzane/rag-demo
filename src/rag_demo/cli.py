from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_demo import __version__
from rag_demo.chunking_recall import (
    DEFAULT_RFC_PATH,
    DEFAULT_T4_QUERIES,
    DEFAULT_T4_REPORT,
    MissCase,
    RecallReport,
    RetrievedItem,
    StrategyEval,
    ensure_sample_rfc,
    evaluate_strategies,
    load_queries,
    load_rfc,
    parse_sections,
    parse_strategies,
    parse_strategy,
    read_report,
    recommended_chunk_sizes,
    strategy_retrieve,
    write_queries,
)
from rag_demo.chunking_recall import (
    write_report as write_t4_report,
)
from rag_demo.config import DEFAULT_COLLECTION, model_config
from rag_demo.embedding_selection import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_T5_CACHE,
    DEFAULT_T5_DATASET,
    DEFAULT_T5_REPORT,
    CandidateResult,
    EmbeddingReport,
    ensure_sample_dataset,
    evaluate_embeddings,
    parse_candidates,
)
from rag_demo.embedding_selection import (
    read_report as read_t5_report,
)
from rag_demo.embedding_selection import (
    write_report as write_t5_report,
)
from rag_demo.milvus_bench import (
    DEFAULT_ATTU_HOST,
    DEFAULT_ATTU_PORT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COUNT,
    DEFAULT_DIM,
    DEFAULT_MILVUS_HOST,
    DEFAULT_MILVUS_PORT,
    DEFAULT_SEED,
    DEFAULT_T2_COLLECTION,
    BenchResult,
    connect_milvus,
    create_index,
    load_vectors,
    prepare_dataset,
    run_benchmark,
    runtime_checks,
    sweep_benchmark,
    write_report,
)
from rag_demo.ollama import check_ollama
from rag_demo.qdrant_weaviate import (
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
    DEFAULT_T3_COLLECTION,
    DEFAULT_WEAVIATE_HOST,
    DEFAULT_WEAVIATE_PORT,
    Backend,
    LoadResult,
    PocBenchResult,
    benchmark_backend,
    experience_notes,
    load_backend,
    weaviate_graphql_query,
)
from rag_demo.qdrant_weaviate import (
    runtime_checks as t3_runtime_checks,
)
from rag_demo.qdrant_weaviate import (
    write_report as write_t3_report,
)
from rag_demo.rag import ask_question, build_index, read_metadata, stream_question

console = Console()
app = typer.Typer(help="RAG and vector database learning demos.", no_args_is_help=True)
t1_app = typer.Typer(help="W4-T1 LangChain RAG Hello World.", no_args_is_help=True)
t2_app = typer.Typer(help="W4-T2 Milvus standalone 1M vector benchmark.", no_args_is_help=True)
t3_app = typer.Typer(help="W4-T3 Qdrant vs Weaviate selection POC.", no_args_is_help=True)
t4_app = typer.Typer(help="W4-T4 chunking strategy Recall@5 comparison.", no_args_is_help=True)
t5_app = typer.Typer(help="W4-T5 embedding model selection evaluation.", no_args_is_help=True)
app.add_typer(t1_app, name="t1")
app.add_typer(t2_app, name="t2")
app.add_typer(t3_app, name="t3")
app.add_typer(t4_app, name="t4")
app.add_typer(t5_app, name="t5")


def version_callback(value: bool) -> None:
    if value:
        console.print(f"rag-demo {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, help="Show version and exit."),
    ] = False,
) -> None:
    _ = version


@app.command()
def doctor(
    chat_model: Annotated[str | None, typer.Option(help="Ollama chat model name.")] = None,
    embed_model: Annotated[str | None, typer.Option(help="Ollama embedding model name.")] = None,
    base_url: Annotated[
        str | None,
        typer.Option(help="Ollama base URL, defaults to OLLAMA_HOST or localhost."),
    ] = None,
) -> None:
    """Check local runtime and Ollama models."""
    config = model_config(chat_model, embed_model, base_url)
    result = check_ollama(config.base_url)

    table = Table(title="RAG Demo Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_row("Ollama CLI", "OK" if result.ollama_cli else "Missing", "ollama command")
    table.add_row("Ollama server", "OK" if result.server else "Unavailable", config.base_url)
    table.add_row(
        "Chat model",
        "OK" if config.chat_model in result.models else "Missing",
        config.chat_model,
    )
    table.add_row(
        "Embedding model",
        "OK" if config.embed_model in result.models else "Missing",
        config.embed_model,
    )
    console.print(table)

    if result.error:
        console.print(f"[yellow]Detail:[/yellow] {result.error}")
    if config.chat_model not in result.models:
        console.print(f"[cyan]Next:[/cyan] ollama pull {config.chat_model}")
    if config.embed_model not in result.models:
        console.print(f"[cyan]Next:[/cyan] ollama pull {config.embed_model}")


@t1_app.command("index")
def t1_index(
    docs: Annotated[
        list[Path],
        typer.Option("--docs", help="Markdown export directory. Repeat for multiple dirs."),
    ],
    persist: Annotated[
        Path,
        typer.Option("--persist", help="Local vector index directory."),
    ] = Path(".rag/index"),
    chunk_size: Annotated[int, typer.Option(help="Splitter chunk size.")] = 800,
    chunk_overlap: Annotated[int, typer.Option(help="Splitter chunk overlap.")] = 120,
    chat_model: Annotated[str | None, typer.Option(help="Ollama chat model name.")] = None,
    embed_model: Annotated[str | None, typer.Option(help="Ollama embedding model name.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Ollama base URL.")] = None,
    collection: Annotated[str, typer.Option(help="Chroma collection name.")] = DEFAULT_COLLECTION,
    reset: Annotated[bool, typer.Option(help="Delete existing index directory first.")] = False,
) -> None:
    """Index Markdown files into a local Chroma vector store."""
    if not docs:
        console.print("[red]Missing option:[/red] provide at least one --docs directory")
        raise typer.Exit(2)
    config = model_config(chat_model, embed_model, base_url)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(
                "Loading, splitting, embedding, and persisting Markdown...",
                total=None,
            )
            metadata = build_index(
                docs_dirs=docs,
                persist_dir=persist,
                config=config,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name=collection,
                reset=reset,
            )
    except Exception as exc:
        console.print(f"[red]Index failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Docs: {metadata.document_count}",
                    f"Chunks: {metadata.chunk_count}",
                    f"Persist: {persist}",
                    f"Embedding: {metadata.embed_model}",
                ]
            ),
            title="Index Ready",
        )
    )


@t1_app.command("ask")
def t1_ask(
    question: Annotated[str, typer.Argument(help="Question to ask against the index.")],
    persist: Annotated[
        Path,
        typer.Option("--persist", help="Local vector index directory."),
    ] = Path(".rag/index"),
    top_k: Annotated[int, typer.Option("--top-k", help="Number of retrieved chunks.")] = 4,
    show_context: Annotated[bool, typer.Option(help="Print retrieved context snippets.")] = False,
    chat_model: Annotated[str | None, typer.Option(help="Ollama chat model name.")] = None,
    embed_model: Annotated[str | None, typer.Option(help="Ollama embedding model name.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Ollama base URL.")] = None,
) -> None:
    """Ask one RAG question."""
    config = model_config(chat_model, embed_model, base_url)
    try:
        answer = ask_question(question, persist, config, top_k=top_k)
    except FileNotFoundError as exc:
        console.print(f"[red]Index not found:[/red] {exc}")
        console.print("[cyan]Next:[/cyan] rag-demo t1 index --docs <markdown-dir>")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Ask failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(Panel(answer.text, title="Answer", expand=False))
    _print_sources(answer.sources)
    if show_context:
        _print_contexts(answer.contexts)


@t1_app.command("chat")
def t1_chat(
    persist: Annotated[
        Path,
        typer.Option("--persist", help="Local vector index directory."),
    ] = Path(".rag/index"),
    top_k: Annotated[int, typer.Option("--top-k", help="Number of retrieved chunks.")] = 4,
    chat_model: Annotated[str | None, typer.Option(help="Ollama chat model name.")] = None,
    embed_model: Annotated[str | None, typer.Option(help="Ollama embedding model name.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Ollama base URL.")] = None,
) -> None:
    """Start an interactive RAG chat."""
    config = model_config(chat_model, embed_model, base_url)
    console.print("[bold]RAG chat[/bold]  输入 /exit 退出")
    while True:
        question = typer.prompt("Question")
        if question.strip().lower() in {"/exit", "exit", "quit", ":q"}:
            break
        try:
            chunks, sources, _contexts = stream_question(question, persist, config, top_k=top_k)
        except FileNotFoundError as exc:
            console.print(f"[red]Index not found:[/red] {exc}")
            console.print("[cyan]Next:[/cyan] rag-demo t1 index --docs <markdown-dir>")
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(f"[red]Ask failed:[/red] {exc}")
            raise typer.Exit(1) from exc

        console.print("[bold]Answer[/bold]")
        try:
            for chunk in chunks:
                console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
            console.print()
        except Exception as exc:
            console.print(f"\n[red]Ask failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        _print_sources(sources)


@t1_app.command("inspect")
def t1_inspect(
    persist: Annotated[
        Path,
        typer.Option("--persist", help="Local vector index directory."),
    ] = Path(".rag/index"),
) -> None:
    """Inspect local index metadata."""
    try:
        metadata = read_metadata(persist)
    except FileNotFoundError as exc:
        console.print(f"[red]Index not found:[/red] {exc}")
        raise typer.Exit(1) from exc

    table = Table(title="Index Metadata")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in metadata.__dict__.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_sources(sources: list[dict[str, object]]) -> None:
    table = Table(title="Sources")
    table.add_column("#", justify="right")
    table.add_column("File")
    table.add_column("Chunk")
    for index, item in enumerate(sources, start=1):
        table.add_row(str(index), str(item["source"]), str(item["chunk_id"]))
    console.print(table)


def _print_contexts(contexts: list[str]) -> None:
    for index, context in enumerate(contexts, start=1):
        console.print(Panel(context[:1200], title=f"Context {index}", expand=False))


@t2_app.command("doctor")
def t2_doctor(
    milvus_host: Annotated[str, typer.Option(help="Milvus host.")] = DEFAULT_MILVUS_HOST,
    milvus_port: Annotated[int, typer.Option(help="Milvus gRPC port.")] = DEFAULT_MILVUS_PORT,
    attu_host: Annotated[str, typer.Option(help="Attu host.")] = DEFAULT_ATTU_HOST,
    attu_port: Annotated[int, typer.Option(help="Attu HTTP port.")] = DEFAULT_ATTU_PORT,
) -> None:
    """Check Docker, Milvus, Attu, and bpftrace readiness."""
    table = Table(title="W4-T2 Milvus Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in runtime_checks(milvus_host, milvus_port, attu_host, attu_port):
        table.add_row(check.name, "OK" if check.ok else "Missing", check.detail)
    console.print(table)
    console.print("[cyan]Milvus:[/cyan] docker-compose up -d")
    console.print("[cyan]Attu:[/cyan] http://localhost:8000")
    console.print("[cyan]eBPF:[/cyan] sudo bpftrace observability/bpftrace/milvus_io.bt")


@t2_app.command("generate")
def t2_generate(
    output: Annotated[Path, typer.Option("--output", help="Dataset metadata output dir.")] = Path(
        ".rag/t2"
    ),
    count: Annotated[int, typer.Option(help="Synthetic vector count.")] = DEFAULT_COUNT,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    query_count: Annotated[int, typer.Option(help="Query vectors to save.")] = 1_000,
    batch_size: Annotated[int, typer.Option(help="Generation batch size.")] = DEFAULT_BATCH_SIZE,
    materialize: Annotated[
        bool,
        typer.Option(help="Write the full vectors.npy memmap; 1M x 768 is about 3GB."),
    ] = False,
) -> None:
    """Generate deterministic dataset metadata and query vectors."""
    try:
        metadata = prepare_dataset(
            output_dir=output,
            count=count,
            dim=dim,
            seed=seed,
            query_count=query_count,
            materialize=materialize,
            batch_size=batch_size,
        )
    except Exception as exc:
        console.print(f"[red]Generate failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Count: {metadata.count}",
                    f"Dim: {metadata.dim}",
                    f"Queries: {metadata.query_count}",
                    f"Materialized: {metadata.materialized}",
                    f"Output: {metadata.output_dir}",
                ]
            ),
            title="Dataset Ready",
        )
    )


@t2_app.command("load")
def t2_load(
    collection: Annotated[
        str,
        typer.Option(help="Milvus collection name."),
    ] = DEFAULT_T2_COLLECTION,
    host: Annotated[str, typer.Option(help="Milvus host.")] = DEFAULT_MILVUS_HOST,
    port: Annotated[int, typer.Option(help="Milvus gRPC port.")] = DEFAULT_MILVUS_PORT,
    count: Annotated[int, typer.Option(help="Synthetic vector count.")] = DEFAULT_COUNT,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    batch_size: Annotated[int, typer.Option(help="Insert batch size.")] = DEFAULT_BATCH_SIZE,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    reset: Annotated[bool, typer.Option(help="Drop and recreate collection first.")] = False,
) -> None:
    """Stream deterministic vectors into Milvus."""
    try:
        connect_milvus(host, port)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Generating and inserting vectors into Milvus...", total=None)
            inserted = load_vectors(collection, count, dim, batch_size, seed, reset)
    except Exception as exc:
        console.print(f"[red]Load failed:[/red] {exc}")
        console.print("[cyan]Next:[/cyan] docker-compose up -d")
        raise typer.Exit(1) from exc
    console.print(
        Panel.fit(f"Collection: {collection}\nRows inserted: {inserted}", title="Load Done")
    )


@t2_app.command("index")
def t2_index(
    collection: Annotated[
        str,
        typer.Option(help="Milvus collection name."),
    ] = DEFAULT_T2_COLLECTION,
    host: Annotated[str, typer.Option(help="Milvus host.")] = DEFAULT_MILVUS_HOST,
    port: Annotated[int, typer.Option(help="Milvus gRPC port.")] = DEFAULT_MILVUS_PORT,
    index_type: Annotated[str, typer.Option(help="IVF_FLAT or HNSW.")] = "IVF_FLAT",
    nlist: Annotated[int, typer.Option(help="IVF_FLAT nlist.")] = 1024,
    hnsw_m: Annotated[int, typer.Option("--hnsw-m", help="HNSW M.")] = 16,
    ef_construction: Annotated[
        int,
        typer.Option("--ef-construction", help="HNSW efConstruction."),
    ] = 200,
    report: Annotated[Path | None, typer.Option(help="Optional JSON report path.")] = None,
) -> None:
    """Build IVF_FLAT or HNSW index on the vector field."""
    index_name = _index_type(index_type)
    try:
        connect_milvus(host, port)
        result = create_index(
            collection_name=collection,
            index_type=index_name,
            nlist=nlist,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
        )
        if report:
            write_report(report, result)
    except Exception as exc:
        console.print(f"[red]Index failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    table = Table(title="Index Built")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in result.__dict__.items():
        table.add_row(key, str(value))
    console.print(table)


@t2_app.command("bench")
def t2_bench(
    collection: Annotated[
        str,
        typer.Option(help="Milvus collection name."),
    ] = DEFAULT_T2_COLLECTION,
    host: Annotated[str, typer.Option(help="Milvus host.")] = DEFAULT_MILVUS_HOST,
    port: Annotated[int, typer.Option(help="Milvus gRPC port.")] = DEFAULT_MILVUS_PORT,
    index_type: Annotated[str, typer.Option(help="IVF_FLAT or HNSW.")] = "IVF_FLAT",
    runs: Annotated[int, typer.Option(help="Number of searches.")] = 1_000,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    top_k: Annotated[int, typer.Option("--top-k", help="Search top K.")] = 10,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    nprobe: Annotated[int, typer.Option(help="IVF_FLAT nprobe.")] = 16,
    ef: Annotated[int, typer.Option(help="HNSW ef.")] = 64,
    filter_expr: Annotated[
        str | None,
        typer.Option("--filter", help='Optional Milvus filter, e.g. "group_id < 10".'),
    ] = None,
    report: Annotated[Path | None, typer.Option(help="Optional JSON report path.")] = None,
) -> None:
    """Run repeated vector searches and print latency/QPS."""
    index_name = _index_type(index_type)
    try:
        connect_milvus(host, port)
        result = run_benchmark(
            collection_name=collection,
            index_type=index_name,
            runs=runs,
            dim=dim,
            top_k=top_k,
            seed=seed,
            nprobe=nprobe,
            ef=ef,
            filter_expr=filter_expr,
        )
        if report:
            write_report(report, result)
    except Exception as exc:
        console.print(f"[red]Bench failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_bench_result(result)


@t2_app.command("sweep")
def t2_sweep(
    collection: Annotated[
        str,
        typer.Option(help="Milvus collection name."),
    ] = DEFAULT_T2_COLLECTION,
    host: Annotated[str, typer.Option(help="Milvus host.")] = DEFAULT_MILVUS_HOST,
    port: Annotated[int, typer.Option(help="Milvus gRPC port.")] = DEFAULT_MILVUS_PORT,
    index_type: Annotated[str, typer.Option(help="IVF_FLAT or HNSW.")] = "IVF_FLAT",
    values: Annotated[
        str,
        typer.Option(help="Comma separated nprobe values for IVF_FLAT or ef values for HNSW."),
    ] = "1,4,8,16,32,64",
    runs: Annotated[int, typer.Option(help="Searches per value.")] = 200,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    top_k: Annotated[int, typer.Option("--top-k", help="Search top K.")] = 10,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    report: Annotated[Path, typer.Option(help="JSON report path.")] = Path(
        ".rag/t2/sweep_report.json"
    ),
) -> None:
    """Sweep nprobe or ef and compare latency."""
    index_name = _index_type(index_type)
    sweep_values = [int(item.strip()) for item in values.split(",") if item.strip()]
    try:
        connect_milvus(host, port)
        results = sweep_benchmark(collection, index_name, sweep_values, runs, dim, top_k, seed)
        write_report(report, results)
    except Exception as exc:
        console.print(f"[red]Sweep failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_sweep_results(results, "nprobe" if index_name == "IVF_FLAT" else "ef")
    console.print(f"[cyan]Report:[/cyan] {report}")


@t2_app.command("observe")
def t2_observe() -> None:
    """Print bpftrace commands for observing Milvus while a benchmark runs."""
    table = Table(title="bpftrace Observability")
    table.add_column("Signal")
    table.add_column("Command")
    table.add_row("Block I/O latency", "sudo bpftrace observability/bpftrace/milvus_io.bt")
    table.add_row("TCP send/receive", "sudo bpftrace observability/bpftrace/milvus_tcp.bt")
    table.add_row("Syscall latency", "sudo bpftrace observability/bpftrace/milvus_syscalls.bt")
    console.print(table)
    console.print("[yellow]Run bpftrace in one terminal and rag-demo t2 bench in another.[/yellow]")


@t3_app.command("doctor")
def t3_doctor(
    qdrant_host: Annotated[str, typer.Option(help="Qdrant host.")] = DEFAULT_QDRANT_HOST,
    qdrant_port: Annotated[int, typer.Option(help="Qdrant HTTP port.")] = DEFAULT_QDRANT_PORT,
    weaviate_host: Annotated[str, typer.Option(help="Weaviate host.")] = DEFAULT_WEAVIATE_HOST,
    weaviate_port: Annotated[int, typer.Option(help="Weaviate HTTP port.")] = DEFAULT_WEAVIATE_PORT,
) -> None:
    """Check Docker, Qdrant, Weaviate, and GraphQL readiness."""
    table = Table(title="W4-T3 Qdrant vs Weaviate Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in t3_runtime_checks(qdrant_host, qdrant_port, weaviate_host, weaviate_port):
        table.add_row(check.name, "OK" if check.ok else "Missing", check.detail)
    console.print(table)
    console.print("[cyan]Start stack:[/cyan] docker-compose up -d qdrant weaviate")
    console.print("[cyan]Qdrant:[/cyan] http://localhost:6333/dashboard")
    console.print("[cyan]Weaviate GraphQL:[/cyan] http://localhost:8081/v1/graphql")


@t3_app.command("load")
def t3_load(
    backend: Annotated[str, typer.Option(help="qdrant, weaviate, or both.")] = "both",
    collection: Annotated[str, typer.Option(help="Collection/class name.")] = DEFAULT_T3_COLLECTION,
    count: Annotated[int, typer.Option(help="Synthetic vector count.")] = DEFAULT_COUNT,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    batch_size: Annotated[int, typer.Option(help="Insert batch size.")] = DEFAULT_BATCH_SIZE,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    reset: Annotated[bool, typer.Option(help="Drop and recreate collection/class first.")] = False,
    qdrant_host: Annotated[str, typer.Option(help="Qdrant host.")] = DEFAULT_QDRANT_HOST,
    qdrant_port: Annotated[int, typer.Option(help="Qdrant HTTP port.")] = DEFAULT_QDRANT_PORT,
    weaviate_host: Annotated[str, typer.Option(help="Weaviate host.")] = DEFAULT_WEAVIATE_HOST,
    weaviate_port: Annotated[int, typer.Option(help="Weaviate HTTP port.")] = DEFAULT_WEAVIATE_PORT,
    report: Annotated[Path | None, typer.Option(help="Optional JSON report path.")] = None,
) -> None:
    """Load the same deterministic vectors and payloads into Qdrant/Weaviate."""
    backends = _t3_backends(backend)
    results = []
    try:
        for item in backends:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[progress.description]Loading {item} vectors..."),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("load", total=None)
                results.append(
                    load_backend(
                        item,
                        count=count,
                        dim=dim,
                        batch_size=batch_size,
                        seed=seed,
                        reset=reset,
                        collection=collection,
                        qdrant_host=qdrant_host,
                        qdrant_port=qdrant_port,
                        weaviate_host=weaviate_host,
                        weaviate_port=weaviate_port,
                    )
                )
        if report:
            write_t3_report(report, results)
    except Exception as exc:
        console.print(f"[red]Load failed:[/red] {exc}")
        console.print("[cyan]Next:[/cyan] docker-compose up -d qdrant weaviate")
        raise typer.Exit(1) from exc
    _print_t3_load_results(results)
    if report:
        console.print(f"[cyan]Report:[/cyan] {report}")


@t3_app.command("bench")
def t3_bench(
    backend: Annotated[str, typer.Option(help="qdrant or weaviate.")] = "qdrant",
    collection: Annotated[str, typer.Option(help="Collection/class name.")] = DEFAULT_T3_COLLECTION,
    runs: Annotated[int, typer.Option(help="Number of searches.")] = 1_000,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    top_k: Annotated[int, typer.Option("--top-k", help="Search top K.")] = 10,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    filtered: Annotated[bool, typer.Option(help="Apply equivalent payload/where filter.")] = True,
    qdrant_host: Annotated[str, typer.Option(help="Qdrant host.")] = DEFAULT_QDRANT_HOST,
    qdrant_port: Annotated[int, typer.Option(help="Qdrant HTTP port.")] = DEFAULT_QDRANT_PORT,
    weaviate_host: Annotated[str, typer.Option(help="Weaviate host.")] = DEFAULT_WEAVIATE_HOST,
    weaviate_port: Annotated[int, typer.Option(help="Weaviate HTTP port.")] = DEFAULT_WEAVIATE_PORT,
    report: Annotated[Path | None, typer.Option(help="Optional JSON report path.")] = None,
) -> None:
    """Run 1000 filtered searches against one backend."""
    selected = _t3_backend(backend)
    try:
        result = benchmark_backend(
            selected,
            runs=runs,
            dim=dim,
            top_k=top_k,
            seed=seed,
            filtered=filtered,
            collection=collection,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            weaviate_host=weaviate_host,
            weaviate_port=weaviate_port,
        )
        if report:
            write_t3_report(report, result)
    except Exception as exc:
        console.print(f"[red]Bench failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t3_bench_results([result])
    if report:
        console.print(f"[cyan]Report:[/cyan] {report}")


@t3_app.command("compare")
def t3_compare(
    collection: Annotated[str, typer.Option(help="Collection/class name.")] = DEFAULT_T3_COLLECTION,
    runs: Annotated[int, typer.Option(help="Number of searches per backend.")] = 1_000,
    dim: Annotated[int, typer.Option(help="Vector dimension.")] = DEFAULT_DIM,
    top_k: Annotated[int, typer.Option("--top-k", help="Search top K.")] = 10,
    seed: Annotated[int, typer.Option(help="Deterministic random seed.")] = DEFAULT_SEED,
    filtered: Annotated[bool, typer.Option(help="Apply equivalent payload/where filter.")] = True,
    qdrant_host: Annotated[str, typer.Option(help="Qdrant host.")] = DEFAULT_QDRANT_HOST,
    qdrant_port: Annotated[int, typer.Option(help="Qdrant HTTP port.")] = DEFAULT_QDRANT_PORT,
    weaviate_host: Annotated[str, typer.Option(help="Weaviate host.")] = DEFAULT_WEAVIATE_HOST,
    weaviate_port: Annotated[int, typer.Option(help="Weaviate HTTP port.")] = DEFAULT_WEAVIATE_PORT,
    report: Annotated[Path, typer.Option(help="JSON report path.")] = Path(
        ".rag/t3/compare_report.json"
    ),
) -> None:
    """Compare Qdrant and Weaviate with the same query workload."""
    results: list[PocBenchResult] = []
    try:
        for backend in ("qdrant", "weaviate"):
            results.append(
                benchmark_backend(
                    backend,
                    runs=runs,
                    dim=dim,
                    top_k=top_k,
                    seed=seed,
                    filtered=filtered,
                    collection=collection,
                    qdrant_host=qdrant_host,
                    qdrant_port=qdrant_port,
                    weaviate_host=weaviate_host,
                    weaviate_port=weaviate_port,
                )
            )
        write_t3_report(report, results)
    except Exception as exc:
        console.print(f"[red]Compare failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t3_bench_results(results)
    console.print(f"[cyan]Report:[/cyan] {report}")


@t3_app.command("notes")
def t3_notes() -> None:
    """Show API and developer-experience notes for the POC."""
    table = Table(title="Developer Experience Notes")
    table.add_column("Backend")
    table.add_column("Engine")
    table.add_column("Query API")
    table.add_column("Filter API")
    table.add_column("Operational Note")
    for note in experience_notes():
        table.add_row(
            note.backend,
            note.engine,
            note.query_api,
            note.filter_api,
            note.operational_note,
        )
    console.print(table)
    sample = weaviate_graphql_query("RagDemoT3Vector", [0.1, 0.2, 0.3], 3, "bucket-1")
    console.print(Panel(sample, title="Weaviate GraphQL Sample", expand=False))


@t4_app.command("queries")
def t4_queries(
    rfc: Annotated[Path, typer.Option("--rfc", help="RFC Markdown/text file.")] = DEFAULT_RFC_PATH,
    output: Annotated[
        Path,
        typer.Option("--output", help="JSONL query set output."),
    ] = DEFAULT_T4_QUERIES,
    limit: Annotated[int | None, typer.Option(help="Optional max query count.")] = None,
) -> None:
    """Generate an editable section-grounded query set from one RFC."""
    try:
        if rfc == DEFAULT_RFC_PATH:
            ensure_sample_rfc(rfc)
        sections = parse_sections(load_rfc(rfc))
        queries = load_queries(None, sections, limit)
        write_queries(output, queries)
    except Exception as exc:
        console.print(f"[red]Query generation failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        Panel.fit(
            f"RFC: {rfc}\nQueries: {len(queries)}\nOutput: {output}",
            title="Query Set Ready",
        )
    )


@t4_app.command("evaluate")
def t4_evaluate(
    rfc: Annotated[Path, typer.Option("--rfc", help="RFC Markdown/text file.")] = DEFAULT_RFC_PATH,
    queries: Annotated[
        Path | None,
        typer.Option("--queries", help="Optional JSONL query set from t4 queries."),
    ] = None,
    strategies: Annotated[
        str,
        typer.Option(help="all or comma-separated: fixed,semantic,parent_child."),
    ] = "all",
    top_k: Annotated[int, typer.Option("--top-k", help="Recall@K value.")] = 5,
    chunk_size: Annotated[int, typer.Option(help="Fixed/child chunk size.")] = 800,
    chunk_overlap: Annotated[int, typer.Option(help="Fixed/child overlap.")] = 120,
    semantic_threshold: Annotated[
        float,
        typer.Option(help="Adjacent paragraph similarity threshold for semantic chunking."),
    ] = 0.42,
    dim: Annotated[int, typer.Option(help="Local hashing embedding dimension.")] = 384,
    query_limit: Annotated[int | None, typer.Option(help="Optional max query count.")] = None,
    report: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_T4_REPORT,
) -> None:
    """Compare fixed, semantic, and parent-child chunking with Recall@5."""
    try:
        if rfc == DEFAULT_RFC_PATH:
            ensure_sample_rfc(rfc)
        selected = parse_strategies(strategies)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Building chunks and evaluating Recall@K...", total=None)
            result = evaluate_strategies(
                rfc_path=rfc,
                query_path=queries,
                strategies=selected,
                top_k=top_k,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                semantic_threshold=semantic_threshold,
                dim=dim,
                query_limit=query_limit,
            )
        write_t4_report(report, result)
    except Exception as exc:
        console.print(f"[red]Evaluate failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t4_report(result)
    _print_t4_misses(result.misses)
    console.print(f"[cyan]Report:[/cyan] {report}")


@t4_app.command("ask")
def t4_ask(
    query: Annotated[str, typer.Argument(help="Question to retrieve against the RFC.")],
    rfc: Annotated[Path, typer.Option("--rfc", help="RFC Markdown/text file.")] = DEFAULT_RFC_PATH,
    strategy: Annotated[str, typer.Option(help="fixed, semantic, or parent_child.")] = "semantic",
    top_k: Annotated[int, typer.Option("--top-k", help="Number of retrieved chunks.")] = 5,
    chunk_size: Annotated[int, typer.Option(help="Fixed/child chunk size.")] = 800,
    chunk_overlap: Annotated[int, typer.Option(help="Fixed/child overlap.")] = 120,
    semantic_threshold: Annotated[float, typer.Option(help="Semantic chunking threshold.")] = 0.42,
    dim: Annotated[int, typer.Option(help="Local hashing embedding dimension.")] = 384,
) -> None:
    """Retrieve top-k chunks for one question under a selected chunking strategy."""
    try:
        if rfc == DEFAULT_RFC_PATH:
            ensure_sample_rfc(rfc)
        selected = parse_strategy(strategy)
        results = strategy_retrieve(
            rfc_path=rfc,
            query=query,
            strategy=selected,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            semantic_threshold=semantic_threshold,
            dim=dim,
        )
    except Exception as exc:
        console.print(f"[red]Retrieve failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t4_retrieved(results, selected)


@t4_app.command("inspect")
def t4_inspect(
    report: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_T4_REPORT,
) -> None:
    """Inspect a saved W4-T4 recall report."""
    try:
        result = read_report(report)
    except Exception as exc:
        console.print(f"[red]Inspect failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t4_report(result)
    _print_t4_misses(result.misses)


@t5_app.command("sample-data")
def t5_sample_data(
    output: Annotated[Path, typer.Option("--output", help="JSONL eval dataset output.")] = (
        DEFAULT_T5_DATASET
    ),
) -> None:
    """Write a small editable MTEB-style + team QA embedding eval dataset."""
    try:
        path = ensure_sample_dataset(output)
    except Exception as exc:
        console.print(f"[red]Sample data failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(Panel.fit(f"Dataset: {path}", title="Embedding Eval Dataset"))


@t5_app.command("evaluate")
def t5_evaluate(
    dataset: Annotated[Path, typer.Option("--dataset", help="JSONL eval dataset.")] = (
        DEFAULT_T5_DATASET
    ),
    models: Annotated[
        str,
        typer.Option(
            help="all or comma-separated: bge-m3,text-embedding-3-small,text-embedding-3-large."
        ),
    ] = "all",
    top_k: Annotated[int, typer.Option("--top-k", help="Recall@K value.")] = 5,
    real: Annotated[
        bool,
        typer.Option(
            "--real",
            help=(
                "Call real Ollama/OpenAI embedding APIs. "
                "Default uses deterministic offline hashing."
            ),
        ),
    ] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option(help="Keep the report even if one provider is unavailable."),
    ] = True,
    cache: Annotated[Path, typer.Option(help="Embedding cache directory.")] = DEFAULT_T5_CACHE,
    report: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_T5_REPORT,
    ollama_base_url: Annotated[
        str,
        typer.Option(help="Ollama base URL for BGE-M3."),
    ] = "http://192.168.1.18:11434",
    openai_base_url: Annotated[
        str,
        typer.Option(help="OpenAI-compatible embeddings base URL."),
    ] = DEFAULT_OPENAI_BASE_URL,
) -> None:
    """Compare BGE-M3 with OpenAI text-embedding-3 models on retrieval QA pairs."""
    try:
        if dataset == DEFAULT_T5_DATASET:
            ensure_sample_dataset(dataset)
        selected = parse_candidates(models)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Embedding dataset and calculating retrieval metrics...", total=None)
            result = evaluate_embeddings(
                dataset_path=dataset,
                candidates=selected,
                top_k=top_k,
                real=real,
                skip_unavailable=skip_unavailable,
                cache_dir=cache,
                ollama_base_url=ollama_base_url,
                openai_base_url=openai_base_url,
            )
        write_t5_report(report, result)
    except Exception as exc:
        console.print(f"[red]Embedding evaluation failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t5_report(result)
    _print_t5_misses(result)
    console.print(f"[cyan]Report:[/cyan] {report}")


@t5_app.command("inspect")
def t5_inspect(
    report: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_T5_REPORT,
) -> None:
    """Inspect a saved W4-T5 embedding selection report."""
    try:
        result = read_t5_report(report)
    except Exception as exc:
        console.print(f"[red]Inspect failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    _print_t5_report(result)
    _print_t5_misses(result)


def _index_type(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"IVF_FLAT", "HNSW"}:
        raise typer.BadParameter("index type must be IVF_FLAT or HNSW")
    return normalized


def _t3_backend(value: str) -> Backend:
    normalized = value.lower()
    if normalized not in {"qdrant", "weaviate"}:
        raise typer.BadParameter("backend must be qdrant or weaviate")
    return normalized  # type: ignore[return-value]


def _t3_backends(value: str) -> list[Backend]:
    normalized = value.lower()
    if normalized == "both":
        return ["qdrant", "weaviate"]
    return [_t3_backend(normalized)]


def _print_bench_result(result: BenchResult) -> None:
    table = Table(title="Benchmark Result")
    table.add_column("Metric")
    table.add_column("Value")
    for key in (
        "collection",
        "index_type",
        "rows",
        "runs",
        "top_k",
        "qps",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "memory_rss_mb",
        "search_params",
    ):
        value = getattr(result, key)
        if isinstance(value, float):
            value = f"{value:.2f}"
        table.add_row(key, str(value))
    console.print(table)


def _print_sweep_results(results: list[BenchResult], param_name: str) -> None:
    table = Table(title="Sweep Result")
    table.add_column(param_name)
    table.add_column("QPS", justify="right")
    table.add_column("P50 ms", justify="right")
    table.add_column("P95 ms", justify="right")
    table.add_column("P99 ms", justify="right")
    for result in results:
        value = result.search_params[param_name]
        table.add_row(
            str(value),
            f"{result.qps:.2f}",
            f"{result.p50_ms:.2f}",
            f"{result.p95_ms:.2f}",
            f"{result.p99_ms:.2f}",
        )
    console.print(table)


def _print_t3_load_results(results: list[LoadResult]) -> None:
    table = Table(title="Load Results")
    table.add_column("Backend")
    table.add_column("Rows", justify="right")
    table.add_column("Dim", justify="right")
    table.add_column("Batch", justify="right")
    table.add_column("Elapsed sec", justify="right")
    table.add_column("Memory MB", justify="right")
    for result in results:
        table.add_row(
            str(result.backend),
            str(result.rows),
            str(result.dim),
            str(result.batch_size),
            f"{result.elapsed_sec:.2f}",
            f"{result.memory_mb:.2f}",
        )
    console.print(table)


def _print_t3_bench_results(results: list[PocBenchResult]) -> None:
    table = Table(title="Qdrant vs Weaviate Benchmark")
    table.add_column("Backend")
    table.add_column("Rows", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Filtered")
    table.add_column("QPS", justify="right")
    table.add_column("P50 ms", justify="right")
    table.add_column("P95 ms", justify="right")
    table.add_column("P99 ms", justify="right")
    table.add_column("Memory MB", justify="right")
    table.add_column("API")
    for result in results:
        table.add_row(
            result.backend,
            str(result.rows),
            str(result.runs),
            str(result.filtered),
            f"{result.qps:.2f}",
            f"{result.p50_ms:.2f}",
            f"{result.p95_ms:.2f}",
            f"{result.p99_ms:.2f}",
            f"{result.memory_mb:.2f}",
            result.api,
        )
    console.print(table)


def _print_t4_report(report: RecallReport) -> None:
    table = Table(title="Chunking Recall Comparison")
    table.add_column("Strategy")
    table.add_column("Chunks", justify="right")
    table.add_column("Queries", justify="right")
    table.add_column(f"Recall@{report.top_k}", justify="right")
    table.add_column("Hits", justify="right")
    table.add_column("Avg ms", justify="right")
    table.add_column("Setting")
    for result in report.results:
        table.add_row(
            result.strategy,
            str(result.chunks),
            str(result.queries),
            f"{result.recall_at_k:.3f}",
            f"{result.hits}/{result.queries}",
            f"{result.avg_latency_ms:.2f}",
            _t4_setting(result),
        )
    console.print(table)
    sizes = ", ".join(str(item) for item in recommended_chunk_sizes(report.results[0].chunk_size))
    console.print(f"[cyan]Tip:[/cyan] try chunk sizes {sizes} and compare recall/latency.")


def _t4_setting(result: StrategyEval) -> str:
    if result.strategy == "semantic":
        return f"threshold={result.semantic_threshold}"
    if result.strategy == "parent_child":
        return f"child_size={result.chunk_size}, overlap={result.chunk_overlap}"
    return f"chunk_size={result.chunk_size}, overlap={result.chunk_overlap}"


def _print_t4_misses(misses: list[MissCase], limit: int = 6) -> None:
    if not misses:
        console.print("[green]No misses in this run.[/green]")
        return
    table = Table(title=f"Miss Cases (first {min(limit, len(misses))})")
    table.add_column("Strategy")
    table.add_column("Query")
    table.add_column("Expected")
    table.add_column("Retrieved Sections")
    for miss in misses[:limit]:
        table.add_row(
            miss.strategy,
            miss.query,
            miss.expected_section,
            " | ".join(miss.retrieved_sections),
        )
    console.print(table)


def _print_t4_retrieved(items: list[RetrievedItem], strategy: str) -> None:
    table = Table(title=f"Top-{len(items)} Retrieval ({strategy})")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Section")
    table.add_column("Chunk")
    table.add_column("Preview")
    for item in items:
        table.add_row(
            str(item.rank),
            f"{item.score:.3f}",
            item.section_title,
            item.chunk_id,
            item.preview,
        )
    console.print(table)


def _print_t5_report(report: EmbeddingReport) -> None:
    table = Table(title=f"Embedding Selection ({report.mode})")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Dim", justify="right")
    table.add_column("Local")
    table.add_column(f"Recall@{report.top_k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("Avg ms", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Vector MB", justify="right")
    table.add_column("Status")
    for result in report.results:
        table.add_row(
            result.name,
            result.provider,
            str(result.dims),
            "yes" if result.local_deployable else "no",
            f"{result.recall_at_5:.3f}",
            f"{result.mrr:.3f}",
            f"{result.avg_latency_ms:.2f}",
            f"${result.estimated_cost_usd:.6f}",
            f"{result.vector_memory_mb:.2f}",
            _t5_status(result),
        )
    console.print(table)
    console.print(Panel(report.recommendation, title="Recommendation", expand=False))


def _t5_status(result: CandidateResult) -> str:
    if not result.unavailable:
        return "OK"
    return f"Unavailable: {result.error}"


def _print_t5_misses(report: EmbeddingReport, limit: int = 6) -> None:
    if not report.misses:
        console.print("[green]No misses in this run.[/green]")
        return
    table = Table(title=f"Embedding Miss Cases (first {min(limit, len(report.misses))})")
    table.add_column("Model")
    table.add_column("Lang")
    table.add_column("Rank", justify="right")
    table.add_column("Query")
    table.add_column("Top IDs")
    for miss in report.misses[:limit]:
        table.add_row(
            miss.model,
            miss.language,
            str(miss.rank or "-"),
            miss.query,
            " | ".join(miss.top_ids),
        )
    console.print(table)


if __name__ == "__main__":
    app()
