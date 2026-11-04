from memory_service.rrf import rrf


def test_item_in_both_lists_beats_item_in_one():
    fused = dict(rrf([["a", "b"], ["b", "c"]]))
    assert fused["b"] > fused["a"] > 0


def test_score_formula_k60():
    fused = dict(rrf([["a"], ["a"]], k=60))
    assert abs(fused["a"] - 2 / 61) < 1e-12


def test_missing_from_one_list_still_counts():
    ids = [d for d, _ in rrf([["x", "y"], []])]
    assert ids == ["x", "y"]


def test_ties_break_by_first_appearance():
    ids = [d for d, _ in rrf([["a", "b"], ["b", "a"]])]
    assert ids == ["a", "b"]


def test_larger_k_flattens_rank_gaps():
    small = dict(rrf([["a", "b"]], k=1))
    large = dict(rrf([["a", "b"]], k=600))
    assert small["a"] / small["b"] > large["a"] / large["b"]
