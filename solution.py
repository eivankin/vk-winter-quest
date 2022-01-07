import json

import numpy as np

from data import Map, Submission
from config import MAP_PATH, SUBMISSION_PATH
from util import get_start_position, get_score, path_to_steps
from visualize import pre_visualize, post_visualize
from algo import a_star_algorithm

if __name__ == '__main__':
    map_matrix = np.array(Map(**json.load(open(MAP_PATH))).raw_map).T
    submission = Submission()
    start = get_start_position(map_matrix)
    ax = pre_visualize(map_matrix)
    try:
        path = a_star_algorithm(tuple(start), map_matrix, ax)
        steps = path_to_steps(path)
        submission = Submission(job=[], path=steps.tolist())
        print(path)
        print(get_score(submission, path, map_matrix))
        print(len(path), len(set(map(tuple, path.tolist()))))
        json.dump(submission.__dict__, open(SUBMISSION_PATH, 'w'))
    except Exception as ex:
        post_visualize()
        raise ex

    post_visualize()
