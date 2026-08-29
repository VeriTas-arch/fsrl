"""Pure metric helpers for frozen causal evaluation."""

from __future__ import annotations

from itertools import combinations


def count_circular_triads(winners: dict[tuple[int, int], int], n_items: int) -> int:
    cycles = 0
    for a, b, c in combinations(range(n_items), 3):
        ab = winners[(a, b)]
        ac = winners[(a, c)]
        bc = winners[(b, c)]
        if (ab == a and bc == b and ac == c) or (ab == b and bc == c and ac == a):
            cycles += 1
    return cycles


def maximum_circular_triads(n_items: int) -> int:
    """Return the maximum number of cyclic triples in an n-item tournament."""

    if n_items % 2:
        return (n_items**3 - n_items) // 24
    return (n_items**3 - 4 * n_items) // 24
