"""Process-local reuse for pure validation work in one formal command."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import cast

_VALIDATION_CACHE: ContextVar[dict[tuple[object, ...], object] | None] = ContextVar(
    "fsrl_validation_cache", default=None
)


@contextmanager
def validation_session() -> Generator[None]:
    """Reuse pure validation results until the outer formal command returns."""

    active = _VALIDATION_CACHE.get()
    if active is not None:
        yield
        return
    token = _VALIDATION_CACHE.set({})
    try:
        yield
    finally:
        _VALIDATION_CACHE.reset(token)


def reuse_validation[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Memoize a pure validator only while ``validation_session`` is active."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        cache = _VALIDATION_CACHE.get()
        if cache is None:
            return function(*args, **kwargs)
        key = (function, args, tuple(sorted(kwargs.items())))
        try:
            hash(key)
        except TypeError:
            return function(*args, **kwargs)
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cast(R, cache[key])

    return wrapped
