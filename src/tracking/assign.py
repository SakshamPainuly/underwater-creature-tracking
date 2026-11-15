"""Optimal one-to-one assignment (Hungarian / Kuhn-Munkres with potentials), standard library only (D-044).

linear_sum_assignment(cost) minimises the total cost of a rectangular matrix and returns min(n, m) pairs (row, col),
sorted. Costs must be finite: callers give forbidden pairs a large cost and drop them afterwards. O(n^2 m), n <= m.
"""

from __future__ import annotations

INF = float("inf")


def linear_sum_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    n = len(cost)
    if n == 0 or len(cost[0]) == 0:
        return []
    m = len(cost[0])
    if any(len(r) != m for r in cost):
        raise ValueError("cost matrix rows differ in length")
    transposed = n > m
    if transposed:
        cost = [list(col) for col in zip(*cost)]
        n, m = m, n
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)  # p[j] = row (1-based) assigned to column j
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    if transposed:
        pairs = [(j, i) for i, j in pairs]
    return sorted(pairs)
