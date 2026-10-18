"""Undo / redo history based on immutable state snapshots."""
from __future__ import annotations

from typing import Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class History(Generic[T]):
    """Linear undo/redo stack.

    States must be immutable (like :class:`~videoeditor.models.EditList`), so a
    snapshot costs nothing more than a reference.
    """

    def __init__(self, initial: T, limit: int = 500) -> None:
        self._current = initial
        self._undo: List[Tuple[str, T]] = []  # (label of the action, state before it)
        self._redo: List[Tuple[str, T]] = []  # (label of the action, state after it)
        self._limit = limit

    @property
    def current(self) -> T:
        return self._current

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> Optional[str]:
        return self._undo[-1][0] if self._undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._redo[-1][0] if self._redo else None

    def push(self, state: T, label: str) -> bool:
        """Record a new state. Returns False (and records nothing) if it is unchanged."""
        if state == self._current:
            return False
        self._undo.append((label, self._current))
        if len(self._undo) > self._limit:
            del self._undo[0]
        self._current = state
        self._redo.clear()
        return True

    def undo(self) -> str:
        """Step back one action and return its label."""
        label, previous = self._undo.pop()
        self._redo.append((label, self._current))
        self._current = previous
        return label

    def redo(self) -> str:
        """Re-apply the most recently undone action and return its label."""
        label, following = self._redo.pop()
        self._undo.append((label, self._current))
        self._current = following
        return label
