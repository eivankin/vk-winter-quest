import os
from copy import copy

DATA_DIR = './data'
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

MAP_PATH = os.path.join(INPUT_DIR, 'map.json')
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, 'submission.json')
EXAMPLE = os.path.join(OUTPUT_DIR, 'sample_submission.json')

for name, path in copy(locals()).items():
    if name.endswith('DIR') and not os.path.exists(path):
        os.mkdir(path)
