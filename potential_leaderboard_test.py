from tsp_core import Timer, generate_network, score_tour

from tsp_solve import branch_and_bound_smart

from tsp_test_utils import assert_valid_tours

def test_for_leaderboard_branch_and_bound_smart():
    locations, edges = generate_network(
        50,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=4321,
    )

    timer = Timer(10)
    stats = branch_and_bound_smart(edges, timer)
    assert_valid_tours(edges, stats)
    score = score_tour(stats[0].tour, edges)

    # On this same graph, Professor Bean's B&B algorithm
    # got a score of 7.610 in 10 seconds
    # and his modified B&B algorithm
    # got a score of 7.038 in 10 seconds
    # See if can you beat this score
    assert score < 7.039
