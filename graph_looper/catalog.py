"""Finding graphs by name.

A consuming project keeps its graphs wherever it likes and refers to them by
name. Resolution order is: an actual file path, then each directory on the
search path, then the graphs bundled with this package.

Add your own directories with the `GRAPHLOOPER_PATH` environment variable
(`os.pathsep`-separated, same shape as `PATH`) or by passing `search_paths=`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from graph_looper.spec import GraphError

#: Directories searched after any caller-supplied ones.
BUNDLED_DIR = Path(__file__).parent / "graphs"

#: Environment variable holding extra graph directories.
PATH_ENV = "GRAPHLOOPER_PATH"

SUFFIXES = (".yaml", ".yml", ".json")


def env_paths() -> list[Path]:
    """Directories named by `GRAPHLOOPER_PATH`."""
    raw = os.environ.get(PATH_ENV, "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def search_paths(extra: Iterable[str | Path] | None = None) -> list[Path]:
    """Every directory that will be searched, most specific first."""
    paths = [Path(p).expanduser() for p in (extra or [])]
    paths.extend(env_paths())
    paths.append(BUNDLED_DIR)
    seen: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def available(extra: Iterable[str | Path] | None = None) -> dict[str, Path]:
    """Every graph findable by name. Earlier search paths win on collisions."""
    found: dict[str, Path] = {}
    for directory in search_paths(extra):
        if not directory.is_dir():
            continue
        for suffix in SUFFIXES:
            for path in sorted(directory.glob(f"*{suffix}")):
                found.setdefault(path.stem, path)
    return found


def bundled() -> dict[str, Path]:
    """Only the graphs shipped inside this package."""
    if not BUNDLED_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(BUNDLED_DIR.glob("*.yaml"))}


def resolve(
    reference: str | Path, extra: Sequence[str | Path] | None = None
) -> Path:
    """Turn a path or a bare name into a file, or explain what is available."""
    path = Path(reference)
    if path.exists():
        return path
    names = available(extra)
    key = str(reference)
    if key in names:
        return names[key]
    known = ", ".join(sorted(names)) or "none found"
    raise GraphError(
        f"no graph at {key!r}. Known graphs: {known}. "
        f"Set {PATH_ENV} or pass search_paths= to add your own directories."
    )
