import math

from tsp_core import score_tour


def assert_valid_tour(edges, tour):
    """
    Length is number of vertices
    Not vertices repeated
    Non-infinite score
    """
    assert len(tour) == len(edges)
    assert len(tour) == len(set(tour))
    assert not math.isinf(score_tour(tour, edges))


def assert_valid_tours(edges, stats):
    assert len(stats) > 0
    for stat in stats:
        assert_valid_tour(edges, stat.tour)
