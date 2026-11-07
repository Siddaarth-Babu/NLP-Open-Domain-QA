from task1.bm25 import BM25Index
from task1.build_corpus import search_candidates
from task1.run import contains_answer
from task1.text import extract_answer, split_paragraphs


def test_bm25_ranks_matching_paragraph_first():
    documents = [
        {"id": "a", "text": "Paris is the capital city of France."},
        {"id": "b", "text": "The Pacific Ocean is the largest ocean."},
    ]
    index = BM25Index.build(documents)
    assert index.search("What is the capital of France?", 1)[0][0] == "a"


def test_bm25_serialization_preserves_retrieval(tmp_path):
    documents = [{"id": "a", "text": "Paris is the capital of France."}]
    index = BM25Index.build(documents)
    index_path = tmp_path / "index.json"
    index.save(index_path)
    assert BM25Index.load(index_path).search("capital France", 1)[0][0] == "a"


def test_answer_extractor_prefers_unmentioned_capitalized_entity():
    answer = extract_answer("What is the capital of France?", "Paris is the capital and most populous city of France.")
    assert answer == "Paris"


def test_recall_match_uses_answer_aliases_after_normalization():
    assert contains_answer("New York City is in the United States.", ["NYC", "the New York City"])


def test_recall_match_does_not_match_inside_another_word():
    assert not contains_answer("The animal was domesticated long ago.", ["cat"])


def test_recall_match_accepts_a_complete_multiword_phrase():
    assert contains_answer("The animal was a domestic cat.", ["cat"])


def test_short_paragraphs_are_excluded_from_corpus():
    assert split_paragraphs("Small.\n\n" + "A" * 100) == ["A" * 100]


def test_discovery_candidates_are_deterministic_and_question_only():
    candidates = search_candidates("What is an oblation in relation to a god?")
    assert candidates[0] == {
        "module": "search",
        "strategy": "full_question",
        "query": "What is an oblation in relation to a god",
    }
    assert {candidate["query"] for candidate in candidates} >= {"oblation relation god", "oblation"}


def test_discovery_extracts_quoted_and_title_like_phrases():
    candidates = search_candidates('Which film featured "The Time Warp" with Tim Curry?')
    assert any(candidate["strategy"] == "quoted_phrase" and "The Time Warp" in candidate["query"] for candidate in candidates)
    assert any(candidate["module"] == "prefixsearch" and candidate["query"] == "Tim Curry" for candidate in candidates)
