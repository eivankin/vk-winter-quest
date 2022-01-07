import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, get_named_colors_mapping

from data import Map, Submission
from config import MAP_PATH, EXAMPLE, SUBMISSION_PATH
from util import get_score, steps_to_values, get_start_position, validate


def discrete_imshow(data: np.array, axes: Axes):
    all_colors = get_named_colors_mapping()
    colors = [all_colors['dodgerblue'], all_colors['white'], all_colors['grey'],
              all_colors['red'], all_colors['goldenrod']] + [all_colors['black']] * 5
    cmap = LinearSegmentedColormap.from_list(
        'Custom cmap', colors, len(colors))
    mat = axes.matshow(data, cmap=cmap, vmin=np.min(data) - .5, vmax=np.max(data) + .5)
    plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))


def pre_visualize(matrix: np.ndarray) -> Axes:
    ax: Axes
    fig, ax = plt.subplots()
    discrete_imshow(matrix.T, ax)
    return ax


def draw_path(ax: Axes, path_values: np.ndarray, matrix: np.ndarray, submission=None):
    validate(path_values, matrix)
    result_path = Path(path_values)
    patch = patches.PathPatch(result_path, lw=1, facecolor=(0, 0, 0, 0), edgecolor='blue')
    ax.add_patch(patch)
    if submission is not None:
        ax.set_title(f'Score: {get_score(submission, path_values, matrix):.5f}')


def post_visualize():
    plt.show()


if __name__ == '__main__':
    current_submission = Submission(**json.load(open(SUBMISSION_PATH)))
    map_matrix = np.array(Map(**json.load(open(MAP_PATH))).raw_map).T
    steps = np.array(current_submission.path)
    start = get_start_position(map_matrix)

    ax = pre_visualize(map_matrix)
    draw_path(ax, steps_to_values(steps, start), map_matrix, submission=current_submission)
    post_visualize()
