"""Checks for errors that would invalidate the reported experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import pytest
from common.metrics import answer_scores, contains_alias, gold_answers, normalize, overlap_score, token_f1
from task3.hybrid import reciprocal_rank_fusion
from task3.groundedness import check


def test_aliases_receive_full_credit():
    refs = gold_answers({"value": "United States", "aliases": ["USA", "United States"]})
    assert refs == ["United States", "USA"]
    assert answer_scores("USA", refs) == {"em": 1.0, "f1": 1.0}


def test_normalization_and_partial_credit():
    assert normalize(" The, Eiffel Tower! ") == "eiffel tower"
    assert token_f1("William Shakespeare", "Shakespeare") == pytest.approx(2 / 3)


def test_repeated_tokens_do_not_inflate_f1():
    assert token_f1("Paris Paris Paris", "Paris") == .5


def test_alias_matching_uses_token_boundaries():
    assert not contains_alias("It is a paradigm", ["Paris"])
    assert contains_alias("He visited Paris, France", ["Paris"])
    assert not contains_alias("anything", ["", "the"])


def test_no_gold_answers_is_an_error():
    with pytest.raises(ValueError):
        answer_scores("x", [])


def test_fusion_favors_agreement_over_one_retriever():
    result = reciprocal_rank_fusion([(0, 999), (1, 3)], [(2, .99), (1, .1)])
    assert result[0][0] == 1
    # Arbitrary raw score rescaling must not affect rank fusion.
    assert result == reciprocal_rank_fusion([(0, .01), (1, .001)], [(2, 100), (1, -80)])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([(0, 2), (0, 1)], [])


def test_empty_answer_is_flagged():
    assert check("", [{"text": "Paris"}])["high_overlap"] is False
    assert check("", [{"text": "Paris"}])["abstained"] is True


def test_overlap_counts_missing_answer_words():
    assert overlap_score("Mira Sen", "Mira wrote a novel") == .5
    assert overlap_score("Mira Mira", "Mira wrote a novel") == .5


def test_overlap_cannot_detect_wrong_relations():
    # This deliberate counterexample documents a limitation, not a good answer.
    assert check("London", [{"text": "London is not the capital of France."}])["high_overlap"]


def test_retrieval_recall_is_monotonic():
    from common.metrics import recall_at_k
    passages = [{"text": "unrelated"}, {"text": "The answer is Paris."}]
    assert recall_at_k(passages, ["Paris"], 1) == 0
    assert recall_at_k(passages, ["Paris"], 2) == 1


def test_classical_extractor_follows_longest_span_rule():
    from task1.baseline import ClassicalExtractor
    extractor = ClassicalExtractor()
    answer, sentence = extractor.answer("Which novel was published?", "A novel was published by the famous British author William Smith in 1920.")
    assert answer == "the famous British author William Smith"
    assert "1920" in sentence


def test_chunking_is_deterministic_and_bounded():
    from prepare import chunk_article
    article = {"pageid": 1, "title": "Example", "revision_id": 10,
               "text": " ".join(["token"] * 260)}
    cfg = {"min_passage_words": 20, "passage_words": 120}
    rows = chunk_article(article, cfg)
    assert [len(r["text"].split()) for r in rows] == [120, 120, 20]
    assert len({r["passage_id"] for r in rows}) == 3


def test_numeric_answer_does_not_require_a_noun_phrase():
    from task1.baseline import ClassicalExtractor
    answer, sentence = ClassicalExtractor().answer("When?", "1920.")
    assert answer == "1920"
    assert sentence == "1920."
