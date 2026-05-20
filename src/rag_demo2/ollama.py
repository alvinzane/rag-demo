from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class DoctorResult:
    ollama_cli: bool
    server: bool
    models: set[str]
    error: str | None = None


def check_ollama(base_url: str, timeout: float = 2.0) -> DoctorResult:
    cli_exists = shutil.which("ollama") is not None
    models: set[str] = set()
    server_ok = False
    error: str | None = None

    try:
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        with urllib.request.urlopen(tags_url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        server_ok = True
        models = {
            item["name"].split(":")[0]
            for item in payload.get("models", [])
            if "name" in item
        }
        models.update(item["name"] for item in payload.get("models", []) if "name" in item)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        error = str(exc)

    if cli_exists and not models:
        try:
            output = subprocess.run(
                ["ollama", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if output.returncode == 0:
                for line in output.stdout.splitlines()[1:]:
                    name = line.split(maxsplit=1)[0]
                    if name:
                        models.add(name)
                        models.add(name.split(":")[0])
        except (OSError, subprocess.SubprocessError) as exc:
            error = str(exc)

    return DoctorResult(ollama_cli=cli_exists, server=server_ok, models=models, error=error)
