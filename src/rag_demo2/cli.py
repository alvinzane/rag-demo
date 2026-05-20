from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_demo2 import __version__
from rag_demo2.config import DEFAULT_COLLECTION, model_config
from rag_demo2.ollama import check_ollama
from rag_demo2.rag import ask_question, build_index, read_metadata

console = Console()
app = typer.Typer(help="RAG and vector database learning demos.", no_args_is_help=True)
t1_app = typer.Typer(help="W4-T1 LangChain RAG Hello World.", no_args_is_help=True)
app.add_typer(t1_app, name="t1")


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
        answer = ask_question(question, persist, config, top_k=top_k)
        console.print(Panel(answer.text, title="Answer", expand=False))
        _print_sources(answer.sources)


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


if __name__ == "__main__":
    app()
