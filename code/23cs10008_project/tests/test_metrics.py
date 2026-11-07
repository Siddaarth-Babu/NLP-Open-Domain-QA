from shared.evaluation.metrics import best_f1, exact_match, normalize_answer, score_predictions


def test_normalization_removes_articles_punctuation_and_case():
    assert normalize_answer("The, Eiffel Tower!") == "eiffel tower"


def test_exact_match_uses_aliases():
    assert exact_match("NYC", ["New York City", "NYC"]) == 1.0


def test_f1_chooses_best_alias():
    assert best_f1("new york", ["New York City", "NYC"]) == 0.8


def test_score_predictions_counts_missing_answer_as_zero():
    result = score_predictions(
        [
            {"id": "one", "answers": ["Paris"]},
            {"id": "two", "answers": ["London"]},
        ],
        {"one": "paris"},
    )
    assert result == {"count": 2.0, "exact_match": 0.5, "f1": 0.5}
