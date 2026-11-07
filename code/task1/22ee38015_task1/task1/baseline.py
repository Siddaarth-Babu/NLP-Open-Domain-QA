"""Required BM25 baseline; no LLM, answer labels, or neural QA reader."""
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.index = BM25Okapi([tokenize(p["text"]) for p in corpus], k1=k1, b=b)

    def search(self, question, depth=50):
        scores = self.index.get_scores(tokenize(question))
        # Stable ties follow the frozen corpus order.
        indices = np.argsort(-scores, kind="stable")[:depth]
        return [(int(i), float(scores[i])) for i in indices]


class ClassicalExtractor:
    """Classical linear POS tagging + a rule-based NP grammar; no neural parser."""
    TAGGER_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip"
    TAGGER_SHA256 = "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b"

    def __init__(self):
        import hashlib
        import urllib.request
        import zipfile
        import nltk
        from nltk.tag import PerceptronTagger
        from nltk.tokenize import TreebankWordTokenizer
        from common.io import ROOT
        resource_dir = ROOT / "cache/nltk_data"
        archive = ROOT / "cache/averaged_perceptron_tagger_eng.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            with urllib.request.urlopen(self.TAGGER_URL, timeout=60) as response:
                archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != self.TAGGER_SHA256:
            raise ValueError("NLTK tagger checksum mismatch; refusing an unpinned model")
        target = resource_dir / "taggers"
        if not (target / "averaged_perceptron_tagger_eng").exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
                        raise ValueError("Unsafe tagger archive entry")
                bundle.extractall(target)
        nltk.data.path.insert(0, str(resource_dir))
        self.tagger = PerceptronTagger(lang="eng")
        self.tokenizer = TreebankWordTokenizer()
        self.chunker = nltk.RegexpParser(r"NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+(<POS><JJ.*>*<NN.*>+)*}")
        self.Tree = nltk.Tree

    def answer(self, question, passage):
        # Sentence segmentation is deliberately rule based. It can split after
        # abbreviations; that is a documented weakness of the classical baseline.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", passage.strip()) if s]
        if not sentences:
            return "", ""
        question_words = set(tokenize(question))
        best = max(sentences, key=lambda sentence: len(question_words & set(tokenize(sentence))))
        spans = list(self.tokenizer.span_tokenize(best))
        tokens = [best[start:end] for start, end in spans]
        tree = self.chunker.parse(self.tagger.tag(tokens))
        candidates, offset = [], 0
        for node in tree:
            if isinstance(node, self.Tree):
                size = len(node.leaves())
                if node.label() == "NP":
                    start, end = spans[offset][0], spans[offset + size - 1][1]
                    candidates.append((start, best[start:end]))
            else:
                size = 1
            offset += size
        candidates.extend((match.start(), match.group())
                          for match in re.finditer(r"\b\d+(?:[,.]\d+)*(?:\s+\d+)*\b", best))
        # Follow the required longest-span heuristic without answer-type hints.
        candidates.sort(key=lambda item: (-len(tokenize(item[1])), -len(item[1]), item[0]))
        return (candidates[0][1] if candidates else ""), best
