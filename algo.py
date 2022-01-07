from typing import Optional
from math import dist

import numpy as np

from util import get_vector, get_neighbors, get_targets, get_nearest, get_surface
from surface_types import DROP
from visualize import draw_path


def reverse_path(node: tuple[int, int], start_point: tuple[int, int], adjacent: list) -> np.ndarray:
    path = []

    while node != start_point:
        i, j = node
        path.insert(0, (i, j))
        di, dj = adjacent[i][j]
        node = (i - di, j - dj)

    path.insert(0, start_point)
    return np.array(path)


def hamiltonian_path(start: tuple[int, int], vector: tuple[int, int],
                     group: set[tuple[int, int]], route: list[tuple[int, int]], matrix: np.ndarray):
    if start not in set(route):
        route.append(start)
        if len(set(route)) == len(set(group)):
            return route
        for vertex in set(get_neighbors(start, vector, matrix)).intersection(group):
            candidate = hamiltonian_path(vertex, get_vector(start, vertex),
                                         group, route.copy(), matrix)
            if candidate is not None:
                return candidate


def get_drop_group(node: tuple[int, int], vector: tuple[int, int], matrix: np.ndarray) -> tuple[
    set[tuple[int, int]], list[tuple[int, int]]]:
    group = set()
    queue = {node}
    path = []

    while queue:
        vertex = queue.pop()
        for neighbor in get_neighbors(vertex, (0, 0), matrix, 1):
            if neighbor not in group and get_surface(np.array(neighbor), matrix) == DROP:
                queue.add(neighbor)
        group.add(vertex)

    if len(group) > 1:
        print(hamiltonian_path(node, vector, group.difference(node), [], matrix))
    return group, path


def a_star_algorithm(start_point: tuple[int, int], matrix: np.ndarray, ax, speed_lim=3) -> Optional[
    np.ndarray]:
    queue = {(start_point, (0, 0))}
    visited = set()

    distances = {start_point: 0}
    adjacent = [[None] * matrix.shape[0] for _ in range(matrix.shape[1])]
    i, j = start_point
    adjacent[i][j] = (0, 0)
    targets = get_targets(matrix)
    filtered_targets = set(targets)
    target = get_nearest(start_point, targets)
    vector = (0, 0)
    # print(target)
    result = []

    while queue:
        node = None

        for vertex, vv in queue:
            if node is None or distances[vertex] + dist(vertex, target) < \
                    distances[node] + dist(node, target):
                node = vertex
                vector = vv

        if node is None:
            return

        if node == target:
            drop_group, drop_path = get_drop_group(target, vector, matrix)
            filtered_targets.difference_update(drop_group)

            print('-' * 10)
            path = reverse_path(node, start_point, adjacent)
            result.extend(path)
            result.extend(drop_path)
            result.pop()
            draw_path(ax, path, matrix)
            if len(filtered_targets) == 0:
                return np.array(result)
            print(vector, node, len(filtered_targets))
            print(path)
            start_point = drop_path[-1] if drop_path else node
            queue = {(start_point, vector)}
            # adjacent = {start_point: start_point}
            target = get_nearest(start_point, list(filtered_targets))
            print(start_point)
            print('-' * 10)
            continue

        for vertex in get_neighbors(node, vector, matrix):
            new_vector = get_vector(node, vertex)
            if max(map(abs, new_vector)) < speed_lim and vertex != node:
                if (vertex, new_vector) not in queue and vertex not in set(x[0] for x in queue) and vertex not in visited:
                    queue.add((vertex, new_vector))
                    i, j = vertex
                    adjacent[i][j] = new_vector
                    distances[vertex] = distances[node] + 1
                else:
                    if distances[vertex] > distances[node] + 1:
                        distances[vertex] = distances[node] + 1
                        i, j = vertex
                        adjacent[i][j] = new_vector

                        if vertex in visited:
                            visited.remove(vertex)
                            queue.add((vertex, new_vector))

        queue.remove((node, vector))
        visited.add(node)
