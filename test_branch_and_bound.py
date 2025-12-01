# ---------------------------------- Imports --------------------------------- #
from byu_pytest_utils import tier

from tsp_core import Timer, generate_network, score_tour
from tsp_solve import greedy_tour, branch_and_bound, branch_and_bound_smart, \
    PARAMS_FOR_SMART_BRANCH_AND_BOUND_SMART_TEST
from tsp_test_utils import assert_valid_tours


from copy import deepcopy

# -------------------------------- Test tiers -------------------------------- #
core = tier('core', 1)
stretch2 = tier('stretch2', 2)


# -------------------------------- Core tests -------------------------------- #
@core
def test_branch_and_bound():
    """
    - Greedy should run almost instantly.
    - B&B should search the entire space in less than 3 minutes.
      (A good implementation should finish in seconds).
    - B&B should find a better score than greedy (on this graph).
    """

    locations, edges = generate_network(
        15,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=312,
    )

    timer = Timer(5)
    greedy_stats = greedy_tour(deepcopy(edges), timer)
    assert not timer.time_out()
    assert_valid_tours(edges, greedy_stats)
    greedy_score = score_tour(greedy_stats[-1].tour, edges)

    timer = Timer(120)
    stats = branch_and_bound(deepcopy(edges), timer)
    assert not timer.time_out()
    assert_valid_tours(edges, stats)
    bnb_score = score_tour(stats[-1].tour, edges)

    assert bnb_score < greedy_score


# ------------------------------ Stretch 2 tests ----------------------------- #
@stretch2
def test_branch_and_bound_smart():
    """
    Your Smart B&B algorithm should find a better answer
    than your B&B algorithm in the same amount of time.
    """
    timeout = PARAMS_FOR_SMART_BRANCH_AND_BOUND_SMART_TEST.pop('timeout')
    locations, edges = generate_network(
        **PARAMS_FOR_SMART_BRANCH_AND_BOUND_SMART_TEST
    )

    timer = Timer(timeout)
    bnb_stats = branch_and_bound(deepcopy(edges), timer)
    assert_valid_tours(edges, bnb_stats)
    bnb_score = score_tour(bnb_stats[-1].tour, edges)

    timer = Timer(timeout)
    stats = branch_and_bound_smart(deepcopy(edges), timer)
    assert_valid_tours(edges, stats)
    smart_score = score_tour(stats[-1].tour, edges)

    assert smart_score < bnb_score
