from task3.hybrid import has_lexical_support, reciprocal_rank_fusion


def test_rrf_uses_one_based_ranks_and_deterministic_ties() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], constant=60)
    assert fused[0][0] == "b"
    assert fused[0][1] == 1 / 62 + 1 / 61
    assert fused[-1][0] == "c"


def test_groundedness_requires_a_complete_normalized_phrase() -> None:
    assert has_lexical_support("The Bill of Rights", ["The first ten amendments are called the Bill of Rights."])
    assert not has_lexical_support("cat", ["Dogs were domesticated long ago."])
    assert not has_lexical_support("I don't know", ["The answer is Mars."])
