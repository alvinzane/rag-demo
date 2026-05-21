from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_demo import __version__
from rag_demo.config import DEFAULT_COLLECTION, model_config
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
from rag_demo.rag import ask_question, build_index, read_metadata, stream_question

console = Console()
app = typer.Typer(help="RAG and vector database learning demos.", no_args_is_help=True)
t1_app = typer.Typer(help="W4-T1 LangChain RAG Hello World.", no_args_is_help=True)
t2_app = typer.Typer(help="W4-T2 Milvus standalone 1M vector benchmark.", no_args_is_help=True)
app.add_typer(t1_app, name="t1")
app.add_typer(t2_app, name="t2")


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
    table.add_row("TCP send/activity", "sudo bpftrace observability/bpftrace/milvus_tcp.bt")
    table.add_row("Syscall latency", "sudo bpftrace observability/bpftrace/milvus_syscalls.bt")
    console.print(table)
    console.print("[yellow]Run bpftrace in one terminal and rag-demo t2 bench in another.[/yellow]")


def _index_type(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"IVF_FLAT", "HNSW"}:
        raise typer.BadParameter("index type must be IVF_FLAT or HNSW")
    return normalized


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


if __name__ == "__main__":
    app()
