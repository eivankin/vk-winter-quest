import json

import numpy as np

from data import Map, Submission
from config import MAP_PATH, SUBMISSION_PATH
from util import get_start_position
from visualize import pre_visualize, post_visualize
from algo import a_star_algorithm

if __name__ == '__main__':
    map_matrix = np.array(Map(**json.load(open(MAP_PATH))).raw_map).T
    submission = Submission()
    start = get_start_position(map_matrix)
    ax = pre_visualize(map_matrix)
    try:
        result = a_star_algorithm(tuple(start), map_matrix, ax)
        print(len(result))
    except Exception as ex:
        post_visualize()
        raise ex

    post_visualize()
