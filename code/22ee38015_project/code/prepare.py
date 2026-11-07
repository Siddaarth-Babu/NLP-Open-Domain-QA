"""Freeze validation questions and construct a question-only Wikipedia corpus.

Gold answer strings NEVER enter Wikipedia search, passage selection, retrieval,
or prompts. The whole corpus is shared across questions and all three tasks.
Cached article revision IDs and hashes provide an audit trail for live content.
"""
import concurrent.futures
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
from common.io import ROOT, config, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from common.metrics import gold_answers, contains_alias, normalize

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "IITKGP-NLP-Course-Project/1.0 (educational open-domain QA experiment)",
           "Cache-Control": "no-cache"}

def sample_questions(cfg):
    destination = ROOT / "data/questions.jsonl"
    meta_path = ROOT / "data/sample_manifest.json"
    identity = {k: cfg[k] for k in ("seed", "sample_size", "dataset", "dataset_config", "dataset_revision")}
    if destination.exists() and meta_path.exists():
        old = read_json(meta_path)
        if old["sampling_config"] != identity:
            raise ValueError("Sampling config changed. Use a fresh project/data directory.")
        return read_jsonl(destination)
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset"], cfg["dataset_config"], split="validation", revision=cfg["dataset_revision"])
    # Deduplicate IDs BEFORE sampling: one question, one observation.
    unique = {}
    for index, row in enumerate(ds):
        unique.setdefault(row["question_id"], index)
    if cfg["sample_size"] > len(unique):
        raise ValueError("Requested sample exceeds the number of unique validation questions")
    selected_ids = random.Random(cfg["seed"]).sample(sorted(unique), cfg["sample_size"])
    rows = []
    for qid in selected_ids:
        row = ds[unique[qid]]
        rows.append({"question_id": qid, "question": row["question"],
                     "answers": gold_answers(row["answer"]), "source_index": unique[qid]})
    write_jsonl(destination, rows)
    write_json(meta_path, {"sampling_config": identity, "loaded_rows": len(ds),
                          "unique_questions": len(unique), "question_ids": selected_ids,
                          "dataset_fingerprint": ds._fingerprint, "questions_sha256": file_hash(destination),
                          "split": "validation", "training_used": False})
    return rows


def search_queries(question, nlp):
    doc = nlp(question)
    keywords = [t.text for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    queries = [" ".join(keywords)]
    # Full conjunctions can return no pages; entity/noun-phrase fallbacks use
    # question text only, never the gold answer or pre-supplied evidence metadata.
    phrases = [e.text for e in doc.ents if e.label_ not in ("CARDINAL", "ORDINAL", "DATE", "TIME")]
    phrases += [c.text for c in doc.noun_chunks if len(c.text.split()) >= 2]
    phrases.sort(key=lambda s: (-len(s.split()), -len(s), s))
    queries += phrases[:2]
    queries.append(" ".join(keywords[:3]))
    return list(dict.fromkeys(q for q in queries if q.strip()))


def website_get(url, params=None):
    """Public page downloads, with bounded retries and explicit rate-limit waits."""
    last = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 429:
                delay = max(60, int(response.headers.get("Retry-After", "60")))
                print(f"Wikipedia page requested a {delay}s pause", flush=True)
                for _ in range(delay // 10):
                    time.sleep(10)
                time.sleep(delay % 10)
                continue
            response.raise_for_status()
            time.sleep(0.4)  # Keep the small worker pool polite to the public site.
            return response.text
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Wikipedia page download failed: {last}")


def search_one(item):
    from bs4 import BeautifulSoup
    qid, queries, count, cache_dir = item
    cache = cache_dir / f"{qid}.json"
    if cache.exists():
        return read_json(cache)
    attempts, found = [], []
    for query in queries:
        html = website_get("https://en.wikipedia.org/w/index.php", {
            "title": "Special:Search", "search": query, "fulltext": "1", "ns0": "1"})
        soup = BeautifulSoup(html, "html.parser")
        hits = []
        for link in soup.select(".mw-search-result-heading a")[:count]:
            title = link.get("title") or link.get_text(" ", strip=True)
            # The public search HTML does not expose page IDs. A negative hash
            # is a temporary cache key; article HTML supplies the actual page ID.
            key = -int(digest(title)[:14], 16)
            hits.append({"pageid": key, "title": title})
        attempts.append({"query": query, "hits": hits})
        if hits:
            found = hits
            break
    result = {"question_id": qid, "attempts": attempts, "pages": found,
              "search_interface": "public Special:Search HTML"}
    write_json(cache, result)
    return result


def fetch_article(item):
    """Extract paragraphs from the full public article, not just the lead."""
    import urllib.parse
    from bs4 import BeautifulSoup
    page, cache_dir, revision = item
    destination = cache_dir / f'{page["pageid"]}.json'
    if destination.exists():
        cached = read_json(destination)
        if cached.get("source") == "wikipedia-html" and cached.get("text"):
            return cached
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_"), safe="()")
    html = website_get(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.select("table, sup, script, style, .reflist, .navbox, .mw-editsection"):
        tag.decompose()
    paragraphs = soup.select(".mw-parser-output p")
    text = "\n\n".join(p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True))
    if soup.select_one(".mw-parser-output") is None:
        raise RuntimeError(f"Article body missing (blocked page or changed layout): {page['title']}")
    def field(name):
        match = re.search(r'"' + name + r'"\s*:\s*(\d+)', html)
        return int(match.group(1)) if match else None
    pageid = field("wgArticleId") or page["pageid"]
    revision_id = field("wgRevisionId")
    if revision_id is None:
        match = re.search(r'Special:Redirect/revision/(\d+)', html)
        revision_id = int(match.group(1)) if match else None
    title_tag = soup.find("h1", id="firstHeading")
    result = {"pageid": pageid, "title": title_tag.get_text(" ", strip=True) if title_tag else page["title"],
              "revision_id": revision_id, "revision_timestamp": None,
              "text": text, "source": "wikipedia-html", "url": url,
              "retrieved_utc": datetime.now(timezone.utc).isoformat(), "text_sha256": digest(text)}
    write_json(destination, result)
    return result


def chunk_article(article, cfg):
    chunks = []
    for paragraph in re.split(r"\n\s*\n|\n", article["text"]):
        if paragraph.strip().startswith("=="):
            continue
        words = paragraph.split()
        if len(words) < cfg["min_passage_words"]:
            continue
        for start in range(0, len(words), cfg["passage_words"]):
            segment = words[start:start + cfg["passage_words"]]
            if len(segment) >= cfg["min_passage_words"]:
                text = " ".join(segment)
                chunks.append({"passage_id": f'{article["pageid"]}:{len(chunks):04d}',
                               "title": article["title"], "pageid": article["pageid"],
                               "revision_id": article["revision_id"], "text": text,
                               "url": f'https://en.wikipedia.org/?curid={article["pageid"]}'})
    return chunks


def prepare():
    cfg = config()
    questions = sample_questions(cfg)
    corpus_path = ROOT / "data/corpus.jsonl"
    if corpus_path.exists() and (ROOT / "data/corpus_manifest.json").exists():
        manifest = read_json(ROOT / "data/corpus_manifest.json")
        if manifest["corpus_sha256"] != file_hash(corpus_path):
            raise ValueError("Frozen corpus checksum failed")
        print("Frozen corpus already present:", manifest["passages"], flush=True)
        return
    import spacy
    nlp = spacy.load("en_core_web_sm")
    search_dir, article_dir = ROOT / "data/search_cache", ROOT / "data/articles"
    search_dir.mkdir(parents=True, exist_ok=True)
    article_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(q["question_id"], search_queries(q["question"], nlp),
             cfg["search_pages_per_question"], search_dir) for q in questions]
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(search_one, job): job[0] for job in jobs}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"stage": "search", "question_id": futures[future], "error": str(exc)})
            if count % 20 == 0:
                print(f"Wikipedia searches: {count}/{len(jobs)}", flush=True)
    results.sort(key=lambda r: r["question_id"])
    write_json(ROOT / "data/search_manifest.json", results)
    pages = {p["pageid"]: p for result in results for p in result["pages"][:cfg["search_pages_per_question"]]}
    articles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_article, (page, article_dir, None)): pid for pid, page in sorted(pages.items())}
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                articles.append(future.result())
            except Exception as exc:
                errors.append({"stage": "article", "pageid": futures[future], "error": str(exc)})
            if count % 30 == 0:
                print(f"Wikipedia full articles: {count}/{len(pages)}", flush=True)
    # Stable ordering; identical passage text is indexed only once globally.
    corpus, seen = [], set()
    articles = list({a["pageid"]: a for a in articles}.values())
    for article in sorted(articles, key=lambda a: a["pageid"]):
        for passage in chunk_article(article, cfg):
            identity = digest(passage["text"])
            if identity not in seen:
                corpus.append(passage)
                seen.add(identity)
    write_json(ROOT / "data/preparation_errors.json", errors)
    if errors:
        # Cached successful work survives. A rerun retries missing requests.
        raise RuntimeError(f"{len(errors)} network failures; rerun prepare after checking data/preparation_errors.json")
    if not corpus:
        raise RuntimeError("No passages collected; refusing to run an empty experiment")
    write_jsonl(corpus_path, corpus)
    # Corpus coverage is measured only AFTER construction, never used to select it.
    normalized_passages = [" " + normalize(p["text"]) + " " for p in corpus]
    coverage = 0
    for question in questions:
        needles = [" " + normalize(answer) + " " for answer in question["answers"] if normalize(answer)]
        coverage += any(any(needle in passage for needle in needles) for passage in normalized_passages)
    write_json(ROOT / "data/corpus_manifest.json", {
        "articles": len(articles), "passages": len(corpus), "corpus_sha256": file_hash(corpus_path),
        "articles_without_indexable_paragraphs": [a["title"] for a in articles if not chunk_article(a, cfg)],
        "questions_sha256": file_hash(ROOT / "data/questions.jsonl"),
        "config_sha256": digest(cfg), "construction": f"question-only Wikipedia search (cached API + public search HTML); first nonempty query; top {cfg['search_pages_per_question']} page per question; full article HTML paragraphs",
        "gold_answers_used_for_construction": False, "answer_containing_corpus_coverage_percent": 100 * coverage / len(questions),
        "questions_with_no_search_results": sum(not r["pages"] for r in results),
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "article_revisions": [{k: a.get(k) for k in ("pageid", "title", "revision_id", "revision_timestamp", "text_sha256")} for a in sorted(articles, key=lambda a: a["pageid"])],
        "caveat": "Dataset 10.2 construction details were not supplied; this explicit procedure is an implementation assumption."})
    print(f"Frozen {len(corpus)} passages from {len(articles)} articles; coverage {coverage}/{len(questions)}", flush=True)


if __name__ == "__main__":
    prepare()
