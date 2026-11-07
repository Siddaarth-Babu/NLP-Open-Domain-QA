from task2.run import contains_answer, truncate_passage


class WhitespaceTokenizer:
    def __call__(self, text, **kwargs):
        tokens = text.split()
        if kwargs.get("truncation"):
            tokens = tokens[: kwargs["max_length"]]
        return {"input_ids": tokens}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_contains_answer_requires_whole_normalized_phrase():
    assert contains_answer("The winner was New York City.", ["New York City"])
    assert not contains_answer("The winner was Yorkshire.", ["York"])


def test_truncate_passage_uses_generator_token_limit():
    assert truncate_passage(WhitespaceTokenizer(), "one two three four", 3) == "one two three"
