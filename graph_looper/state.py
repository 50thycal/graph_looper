"""Persistent state — what survives between runs.

A run without a state store starts cold every time: whatever a graph learned on
the last pass is gone. A store lets a graph accumulate — a reviewer's recurring
complaint, a house style, a list of things that went wrong before — and read it
back on the next run through `{{ state.<key> }}`.

State is namespaced by graph name, so one store can back many graphs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_STATE_PATH = Path(".graphloop") / "state.json"

#: Keys the engine writes itself. Graph authors should not use this prefix.
RESERVED_PREFIX = "_"


class StateError(RuntimeError):
    """A state store could not be read or written."""


@runtime_checkable
class StateStore(Protocol):
    """Somewhere a graph's state lives between runs."""

    def load(self, namespace: str) -> dict[str, Any]:
        """Return the stored state for *namespace*, or `{}` if there is none."""

    def save(self, namespace: str, state: dict[str, Any]) -> None:
        """Persist *state* for *namespace*, replacing whatever was there."""


class NullStateStore:
    """Remembers nothing. The default, so a library caller never gets
    surprise writes into their working directory."""

    def load(self, namespace: str) -> dict[str, Any]:
        return {}

    def save(self, namespace: str, state: dict[str, Any]) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NullStateStore()"


class MemoryStateStore:
    """Keeps state in this process only. Useful for tests, and for chaining
    several runs together in one script without touching the filesystem."""

    def __init__(self, initial: dict[str, dict[str, Any]] | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in (initial or {}).items()
        }

    def load(self, namespace: str) -> dict[str, Any]:
        return dict(self._data.get(namespace, {}))

    def save(self, namespace: str, state: dict[str, Any]) -> None:
        self._data[namespace] = dict(state)

    @property
    def data(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._data.items()}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MemoryStateStore({sorted(self._data)})"


class FileStateStore:
    """A JSON file holding every namespace. Writes are atomic — a crashed run
    cannot leave a half-written file behind."""

    def __init__(self, path: str | Path = DEFAULT_STATE_PATH) -> None:
        self.path = Path(path)

    def load(self, namespace: str) -> dict[str, Any]:
        return dict(self._read().get(namespace, {}))

    def save(self, namespace: str, state: dict[str, Any]) -> None:
        everything = self._read()
        everything[namespace] = state
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".state-", suffix=".json"
        )
        try:
            with os.fdopen(handle, "w") as file:
                json.dump(everything, file, indent=2, ensure_ascii=False, default=str)
                file.write("\n")
            os.replace(temporary, self.path)
        except OSError as exc:
            Path(temporary).unlink(missing_ok=True)
            raise StateError(f"could not write state to {self.path}: {exc}") from exc

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            loaded = json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            raise StateError(
                f"{self.path} is not valid JSON; move it aside or repair it: {exc}"
            ) from exc
        except OSError as exc:
            raise StateError(f"could not read state from {self.path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise StateError(f"{self.path} should hold a JSON object, not a list")
        return loaded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FileStateStore({str(self.path)!r})"


def resolve_store(state: Any) -> StateStore:
    """Turn the `state=` argument into a store.

    Accepts a `StateStore`, a path, `True` for the default file location, and
    `None`/`False` for no persistence at all.
    """
    if state is None or state is False:
        return NullStateStore()
    if state is True:
        return FileStateStore()
    if isinstance(state, (str, Path)):
        return FileStateStore(state)
    if isinstance(state, StateStore):
        return state
    raise StateError(
        "state must be a StateStore, a path, True for the default file, or None; "
        f"got {type(state).__name__}"
    )


def record(
    state: dict[str, Any],
    key: str,
    value: Any,
    *,
    append: bool = False,
    limit: int | None = None,
) -> None:
    """Write one value into a state dict, in place.

    With `append`, the key holds a list and the newest value goes on the end,
    trimmed to the most recent `limit` entries — so a graph that records a
    lesson every run accumulates a bounded history rather than one overwrite.
    """
    if not append:
        state[key] = value
        return
    existing = state.get(key)
    if not isinstance(existing, list):
        existing = [] if existing is None else [existing]
    existing.append(value)
    if limit is not None and limit > 0:
        existing = existing[-limit:]
    state[key] = existing
