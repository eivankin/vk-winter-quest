from dataclasses import dataclass


@dataclass
class Map:
    raw_map: list[list[int]]


@dataclass
class Submission:
    job: list[list[int, int]] = None
    path: list[list[int, int]] = None
