#!/usr/bin/env python3
"""Build an auditable Wikipedia-paragraph corpus for the fixed TriviaQA questions."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task1.io import read_jsonl, write_jsonl
from task1.text import split_paragraphs

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "OpenDomainQACoursework/0.1 (educational project; contact: local-user)"
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "the",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with", "would",
}
QUESTION_OPENERS = {"what", "which", "who", "when", "where", "how", "name", "the", "in", "on"}


def _append_candidate(candidates: list[dict[str, str]], seen: set[tuple[str, str]], module: str, strategy: str, query: str) -> None:
    query = " ".join(query.split()).strip(" ,.;:?!")
    key = (module, query.casefold())
    if query and key not in seen:
        seen.add(key)
        candidates.append({"module": module, "strategy": strategy, "query": query})


def search_candidates(question: str) -> list[dict[str, str]]:
    """Return deterministic search attempts derived exclusively from question text."""
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    _append_candidate(candidates, seen, "search", "full_question", question)

    quoted_phrases = re.findall(r'["“]([^"”]{3,100})["”]', question)
    for phrase in sorted(quoted_phrases, key=lambda value: (-len(value), value.casefold())):
        _append_candidate(candidates, seen, "search", "quoted_phrase", f'"{phrase}"')
        _append_candidate(candidates, seen, "prefixsearch", "quoted_title_prefix", phrase)

    capitalized_phrases = re.findall(
        r"\b(?:[A-Z][A-Za-z0-9'’.-]+)(?:\s+(?:[A-Z][A-Za-z0-9'’.-]+|of|the|and)){1,6}",
        question,
    )
    for phrase in sorted(capitalized_phrases, key=lambda value: (-len(value.split()), value.casefold())):
        words = phrase.split()
        while words and words[0].casefold() in QUESTION_OPENERS:
            words.pop(0)
        if len(words) >= 2:
            _append_candidate(candidates, seen, "prefixsearch", "capitalized_title_prefix", " ".join(words))

    terms = re.findall(r"[A-Za-z0-9]+", question)
    content_terms = [term for term in terms if term.casefold() not in STOP_WORDS and len(term) > 1]
    if content_terms:
        _append_candidate(candidates, seen, "search", "content_keywords", " ".join(content_terms[:12]))
        for size in (8, 6, 4, 3, 2):
            if len(content_terms) >= size:
                _append_candidate(candidates, seen, "search", f"first_{size}_keywords", " ".join(content_terms[:size]))
        ranked_terms = sorted(enumerate(content_terms), key=lambda item: (-len(item[1]), item[0], item[1].casefold()))
        for _, term in ranked_terms[:6]:
            _append_candidate(candidates, seen, "search", "single_informative_term", term)
            _append_candidate(candidates, seen, "prefixsearch", "single_term_title_prefix", term)
    return candidates


def _api_request(parameters: dict[str, str], retries: int) -> dict[str, Any]:
    request = Request(f"{API_URL}?{urlencode(parameters)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:  # nosec B310: fixed HTTPS endpoint
                return json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt == retries:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
            print(f"Wikipedia rate limit; retrying in {delay:.0f}s.", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable API retry state")


def _search_page_ids(query: str, module: str, limit: int, retries: int) -> list[str]:
    parameters = {"action": "query", "format": "json"}
    if module == "search":
        parameters.update({"list": "search", "srsearch": query, "srlimit": str(limit), "srnamespace": "0"})
        payload = _api_request(parameters, retries)
        results = payload.get("query", {}).get("search", [])
    elif module == "prefixsearch":
        parameters.update({"list": "prefixsearch", "pssearch": query, "pslimit": str(limit), "psnamespace": "0"})
        payload = _api_request(parameters, retries)
        results = payload.get("query", {}).get("prefixsearch", [])
    else:
        raise ValueError(f"unsupported search module: {module}")
    return [str(result["pageid"]) for result in results if "pageid" in result]


def _fetch_pages(page_ids: list[str], retries: int) -> list[dict[str, Any]]:
    if not page_ids:
        return []
    payload = _api_request(
        {
            "action": "query", "format": "json", "pageids": "|".join(page_ids),
            "prop": "extracts|info|revisions", "explaintext": "1", "inprop": "url",
            "rvprop": "ids|timestamp", "redirects": "1",
        },
        retries,
    )
    pages = payload.get("query", {}).get("pages", {})
    articles = []
    for page_id in page_ids:
        page = pages.get(page_id, {})
        if not page.get("extract"):
            continue
        revision = (page.get("revisions") or [{}])[0]
        articles.append(
            {
                "page_id": str(page.get("pageid", page_id)),
                "title": page["title"],
                "url": page.get("fullurl", ""),
                "revision_id": revision.get("revid"),
                "revision_timestamp": revision.get("timestamp"),
                "extract": page["extract"],
            }
        )
    return articles


def discover_articles(question: str, pages_per_question: int, retries: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stop at the first deterministic question-only query that returns articles."""
    attempts = []
    for candidate_rank, candidate in enumerate(search_candidates(question)):
        page_ids = _search_page_ids(candidate["query"], candidate["module"], pages_per_question, retries)
        attempts.append({**candidate, "result_count": len(page_ids)})
        articles = _fetch_pages(page_ids, retries)
        if articles:
            discovery = {**candidate, "candidate_rank": candidate_rank, "attempts": attempts}
            if candidate_rank:
                print(f"Used {candidate['strategy']}: {candidate['query']!r}")
            return articles, discovery
    return [], {"module": None, "strategy": "no_result", "query": None, "candidate_rank": None, "attempts": attempts}


def _read_cache(path: Path, question: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {"module": "generator_search", "strategy": "legacy_cache", "query": question, "candidate_rank": None, "attempts": []}
    return payload["articles"], payload["discovery"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/processed/triviaqa_validation_300.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/wikipedia_paragraphs.jsonl"))
    parser.add_argument("--manifest-output", type=Path, default=Path("data/processed/wikipedia_discovery_manifest.jsonl"))
    parser.add_argument("--pages-per-question", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-forever", action="store_true", help="Keep retrying network failures until they succeed.")
    parser.add_argument("--retry-pause-seconds", type=float, default=60.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/wikipedia_api_cache"))
    parser.add_argument("--refresh-empty-cache", action="store_true", help="Retry only cached zero-result searches.")
    parser.add_argument("--refresh-all-cache", action="store_true", help="Re-run discovery for every question and replace legacy cache records.")
    parser.add_argument("--allow-incomplete-coverage", action="store_true", help="Permit corpus output with uncovered questions; never use for reported results.")
    parser.add_argument("--max-questions", type=int, help="Use only for a smoke test; omit for reported results.")
    args = parser.parse_args()
    if args.pages_per_question < 1 or args.delay_seconds < 0 or args.retries < 0 or args.retry_pause_seconds < 0:
        raise SystemExit("pages-per-question must be positive; delays and retries must be non-negative")

    documents: list[dict[str, object]] = []
    known_paragraphs: dict[tuple[str, str], dict[str, object]] = {}
    manifest: list[dict[str, Any]] = []
    questions = read_jsonl(args.questions)[:args.max_questions]
    for position, example in enumerate(questions, start=1):
        cache_path = args.cache_dir / f"{example['id']}.json"
        cached_articles: list[dict[str, Any]] = []
        cached_discovery: dict[str, Any] = {}
        if cache_path.exists():
            cached_articles, cached_discovery = _read_cache(cache_path, str(example["question"]))
        should_refresh = args.refresh_all_cache or not cache_path.exists() or (args.refresh_empty_cache and not cached_articles)
        if should_refresh:
            while True:
                try:
                    articles, discovery = discover_articles(str(example["question"]), args.pages_per_question, args.retries)
                    break
                except OSError as error:
                    if not args.retry_forever:
                        raise SystemExit(f"Wikipedia request failed for {example['id']}; corpus not replaced: {error}") from error
                    print(f"Request for {example['id']} failed ({error}); retrying in {args.retry_pause_seconds:.0f}s.", file=sys.stderr)
                    time.sleep(args.retry_pause_seconds)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"version": 2, "articles": articles, "discovery": discovery}, ensure_ascii=False), encoding="utf-8")
        else:
            articles, discovery = cached_articles, cached_discovery

        source_record = {
            "question_id": example["id"], "question": example["question"], "article_count": len(articles),
            "article_titles": [article["title"] for article in articles], "discovery": discovery,
        }
        manifest.append(source_record)
        provenance = {
            "question_id": example["id"], "module": discovery.get("module"),
            "strategy": discovery.get("strategy"), "query": discovery.get("query"),
        }
        for article in articles:
            for paragraph_number, paragraph in enumerate(split_paragraphs(article["extract"])):
                key = (article["title"], paragraph)
                if key in known_paragraphs:
                    document = known_paragraphs[key]
                    if example["id"] not in document["source_question_ids"]:
                        document["source_question_ids"].append(example["id"])
                        document["discovery_sources"].append(provenance)
                    continue
                document = {
                    "id": f"wp-{len(documents):07d}", "page_id": article.get("page_id"),
                    "title": article["title"], "url": article["url"],
                    "revision_id": article.get("revision_id"), "revision_timestamp": article.get("revision_timestamp"),
                    "text": paragraph, "source_question_id": example["id"],
                    "source_question_ids": [example["id"]], "discovery_sources": [provenance],
                    "paragraph_number": paragraph_number,
                }
                known_paragraphs[key] = document
                documents.append(document)
        print(f"Processed {position}/{len(questions)} questions; {len(documents)} unique paragraphs.")
        if should_refresh:
            time.sleep(args.delay_seconds)

    write_jsonl(manifest, args.manifest_output)
    represented = {source_id for document in documents for source_id in document["source_question_ids"]}
    missing = [str(example["id"]) for example in questions if example["id"] not in represented]
    if missing and not args.allow_incomplete_coverage:
        raise SystemExit(f"Corpus not replaced: {len(missing)} questions have no paragraph coverage. See {args.manifest_output}. Missing IDs: {missing}")
    if not documents:
        raise SystemExit("No paragraphs were fetched; check network access and input questions.")
    write_jsonl(documents, args.output)
    print(f"Wrote {len(documents)} paragraphs covering {len(represented)}/{len(questions)} questions to {args.output}.")
    print(f"Wrote discovery provenance to {args.manifest_output}.")


if __name__ == "__main__":
    main()
