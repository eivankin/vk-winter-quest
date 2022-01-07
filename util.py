from math import dist
from itertools import product

import numpy as np

from data import Submission
from surface_types import DROP, START, WALL, RADIUS, NAMES


def steps_to_values(steps: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
    return np.insert(steps, 0, start_pos, axis=0).cumsum(axis=0)


def get_score(submission: Submission, path: np.ndarray, matrix: np.ndarray) -> float:
    visited = get_count_of_intersections_with(DROP, path, matrix)
    return 3600 * 1.1 ** visited / \
           (len(submission.path) + len(submission.job)) * (
                   visited > 0 and len(submission.path) > 19)


def get_start_position(matrix: np.ndarray) -> np.ndarray:
    return np.stack(np.array(np.where(matrix == START)), axis=1)[0]


def get_count_of_intersections_with(surface_type: int, path: np.ndarray, matrix: np.ndarray):
    filtered = matrix[path[:, 0], path[:, 1]]
    return np.count_nonzero(filtered == surface_type)


def validate(path: np.ndarray, matrix: np.ndarray):
    if get_count_of_intersections_with(WALL, path, matrix):
        raise Exception('intersection with wall')
    for prev_point, current_point, next_point in zip(path[:-2], path[1:-1], path[2:]):
        surface = get_surface(current_point, matrix)
        vector = current_point - prev_point
        new_vector = next_point - current_point
        if max(abs(new_vector - vector)) > RADIUS[surface]:
            raise Exception(
                f'invalid move {new_vector} from {NAMES[surface]} {current_point} with vector {vector}')


def get_surface(point: np.ndarray, matrix: np.ndarray):
    return matrix[point[0], point[1]]


def get_neighbors(node: tuple[int, int], vector: tuple[int, int],
                  matrix: np.ndarray, radius=None) -> list[tuple[int, int]]:
    surface = get_surface(np.array(node), matrix)
    center = np.array(node) + np.array(vector)
    if radius is None:
        radius = RADIUS[surface]

    result = []
    for shift in product(range(-radius, radius + 1), repeat=2):
        point = center + np.array(shift)
        x, y = tuple(point)
        if 0 <= x < matrix.shape[0] and 0 <= y < matrix.shape[1] and \
                get_surface(point, matrix) != WALL:
            result.append(tuple(point))
    return result


def get_min_dist(point: tuple[int, int], targets: set[tuple[int, int]]):
    return min(map(lambda p: dist(point, p), targets))


def get_nearest(point: tuple[int, int], targets: list[tuple[int, int]]):
    return min(targets, key=lambda p: dist(point, p))


def get_targets(matrix: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(p) for p in np.stack(np.where(matrix == DROP), axis=1)]


def get_vector(from_point: tuple[int, int], to_point: tuple[int, int]):
    return tuple(np.array(to_point) - np.array(from_point))
