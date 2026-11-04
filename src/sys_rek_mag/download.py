"""Pobieranie danych Amazon Reviews 2023 (nigdy nie nadpisuje istniejących plików)."""

from __future__ import annotations

import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

from sys_rek_mag import config
from sys_rek_mag.config import PATHS, Paths

_BLOCK = 1024 * 1024


def _fetch(url: str, target: Path) -> None:
    """Pobiera strumieniowo (bloki 1 MB) z paskiem postępu w MB."""
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        done = 0
        while block := response.read(_BLOCK):
            out.write(block)
            done += len(block)
            print(f"\r  {target.name}: {done / _BLOCK:.0f} MB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)


def download(
    paths: Paths = PATHS,
    files: dict[str, str] = config.AMAZON_FILES,
    fetch: Callable[[str, Path], None] = _fetch,
) -> list[Path]:
    """Pobiera brakujące pliki do `paths.raw`; istniejące pomija. Zwraca wszystkie ścieżki."""
    paths.raw.mkdir(parents=True, exist_ok=True)
    result = []
    for name, url in files.items():
        target = paths.raw / name
        result.append(target)
        if target.exists():
            print(f"Pomijam (istnieje): {name}")
            continue
        part = target.with_name(f".{name}.part")
        try:
            fetch(url, part)
            os.replace(part, target)
        finally:
            part.unlink(missing_ok=True)
        print(f"Pobrano: {name}")
    return result


def main() -> None:
    download()


if __name__ == "__main__":
    main()
