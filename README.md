# Project 10 — Open-Domain Question Answering: From Lexical Retrieval to RAG

Dataset assigned: **TriviaQA (`rc.nocontext`)** — Dataset 10.3.

## This is a group submission

**Group No. 30.** Three of us worked on this assignment, and each of us independently implemented
**all three tasks** end to end, rather than splitting the assignment by task:

| # | Name | Roll No. |
| --- | --- | --- |
| 1 | B. Siddaarth | `23CS30011` |
| 2 | Bala Dharnesh | `23CS10008` |
| 3 | Subham Ghosh | `22EE38015` |

We did this on purpose: three independently-built pipelines on the same dataset let us
cross-check our own results against two other implementations instead of trusting a single
codebase, and the differences between the three approaches (different Wikipedia-corpus
construction strategies, different generators, different code architectures) turned out to be
genuinely informative on their own — see `report.pdf` for the full comparison.

**We did not merge our code into one shared pipeline.** Each person's implementation is kept
exactly as they wrote it; nothing below rewrites anyone's logic, only reorganizes where files live
so all three can be read and (where the original design allows it) run side by side. We combined
the report, the README, and the presentation into one each, instead of submitting three of each.
If you only want to run *one* person's submission, jump straight to that person's section below.

## Folder structure

```
NLP/
├── README.md
├── report.tex / report.pdf          # one merged report, segregated by author per task (see below)
├── code/
│   ├── task1/
│   │   ├── 23cs30011_task1/          # B. Siddaarth  — notebook, self-contained
│   │   ├── 23cs10008_task1/          # Bala Dharnesh — standalone runnable slice of his Task 1
│   │   └── 22ee38015_task1/          # Subham Ghosh  — standalone runnable slice of his Task 1
│   ├── task2/
│   │   ├── 23cs30011_task2/          # two notebooks: zero-shot and few-shot iterations
│   │   ├── 23cs10008_task2/          # standalone runnable slice of his Task 2
│   │   └── 22ee38015_task2/          # his Task 2 modules (run via his full project, see below)
│   ├── task3/
│   │   ├── 23cs30011_task3/          # two notebooks: zero-shot and few-shot iterations
│   │   ├── 23cs10008_task3/          # standalone runnable slice of his Task 3
│   │   └── 22ee38015_task3/          # his Task 3 modules (run via his full project, see below)
│   ├── 23cs30011_project/            # B. Siddaarth's generated data/, results/, and requirements.txt
│   ├── 23cs10008_project/            # Bala's complete original repo: Docker, tests, one-command pipeline
│   └── 22ee38015_project/            # Subham's complete original repo: code, data, results, Colab notebook
```

Every person's generated data and results live in exactly one place —
`code/<roll_no>_project/` — so the layout is uniform across all three submissions; nothing sits
loose at the repository root. Each `_project/` folder also holds that person's own
`requirements.txt`/environment files, so the three environments never mix.

We kept each submission down to what's actually needed to read, run, and verify the code. Each
person originally had their own README and report; those are now just this one README and the one
`report.pdf` at the root, with the run commands and numbers they held folded in below, so nothing
is lost. A few other things that only made sense for an individual delivery — a slide deck, a
video-narration script, font files bundled purely to typeset someone's own separate report, a
downloaded NLTK cache — didn't carry over either. The actual code, configs, tests, and saved
results/predictions are all still here in full, as is Bala's `RESULTS.md`, since his submission
has no report of its own to fold those numbers into.

Why the per-task split differs slightly per person: 23CS30011's Task 1/2/3 notebooks are each
fully self-contained, so they sit directly under `code/task<n>/23cs30011_task<n>/`. Bala's
`task*/run.py` scripts are real standalone CLI entry points that only need a handful of small
shared files (`shared/`, and occasionally a Task 1/2 helper module) copied alongside them, so his
`code/task<n>/23cs10008_task<n>/` folders are genuinely runnable on their own — we verified this by
importing every module and running his unit test suite against the reorganized layout (17/17
pass, unmodified). Subham's Task 2/3 code is architected as library modules dispatched by a shared
orchestrator (`common/experiment.py`, driven by `python code/run.py task2` etc.), so splitting it
into independently-runnable per-task scripts would have meant rewriting his orchestration logic —
something we deliberately avoided. His `code/task<n>/22ee38015_task<n>/` folders therefore hold
that task's actual module files for reading (and his Task 1 module, which has no cross-task
dependency, *is* directly runnable), with a clear note pointing to his full project
(`code/22ee38015_project/`) for how to actually execute Tasks 2 and 3. Both Bala's and Subham's
complete original repositories are additionally kept under `code/<roll_no>_project/` for anyone
who wants to run either submission's full pipeline (Docker, notebook, tests, and all) rather than
one task at a time.

---

## Running 23CS30011's submission (B. Siddaarth)

### Environment setup

```bash
cd NLP
python3 -m venv venv
source venv/bin/activate
pip install -r code/23cs30011_project/requirements.txt
# torch is pulled in as a dependency; if you want a CPU-only install instead of the
# default (which may include CUDA packages), run this first:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name=nlp-venv --display-name="Python 3 (NLP venv)"
```

All random seeds are fixed to `SEED = 42` in every notebook. Generated data and results live in
`code/23cs30011_project/data/` and `code/23cs30011_project/results/` (already populated from our
own run; the notebooks recreate them from scratch if deleted).

### Dataset

`mandarjoshi/trivia_qa`, config `rc.nocontext`, split `validation` — downloaded automatically via
HuggingFace `datasets` the first time Task 1 runs (cached under `~/.cache/huggingface`, 17,944
questions total, no manual download needed). Task 1 then draws a fixed random sample of 300 of
these questions (seed 42) and reuses that same 300-question subset for Tasks 2 and 3 — see below.

### How to run (in order — each task depends on the previous one's outputs)

#### Task 1 — `code/task1/23cs30011_task1/task1_baseline_retrieval.ipynb`
Run top-to-bottom locally (CPU, no GPU needed), either via the CLI:
```bash
jupyter nbconvert --to notebook --execute --inplace code/task1/23cs30011_task1/task1_baseline_retrieval.ipynb
```
or by opening the notebook in Jupyter and using Run All.

Writes (into `code/23cs30011_project/`):
- `data/triviaqa_sample_300.jsonl` — the fixed 300-question sample (reused by Tasks 2 & 3)
- `data/wiki_pages_cache.json`, `data/wiki_paragraph_index.jsonl` — the Wikipedia-paragraph index built from scratch (reused by Tasks 2 & 3)
- `results/task1_metrics.json`, `results/task1_per_question_results.csv`

#### Task 2 — two prompting iterations, both evaluated
We tried two prompting strategies for the generator and report both (see `report.pdf` for the
comparison); Iteration 2 (few-shot) scores higher and is the final result.

**Iteration 1 (zero-shot) — `code/task2/23cs30011_task2/task2_rag.ipynb`**
Needs a GPU for the generator (Qwen2.5-7B-Instruct), so run it on **Google Colab**:
1. Open the notebook in Colab, set Runtime → Change runtime type → T4 GPU.
2. Run top-to-bottom. When prompted, upload `code/23cs30011_project/data/triviaqa_sample_300.jsonl` and `code/23cs30011_project/data/wiki_paragraph_index.jsonl` from this repo.
3. Download the three files it offers at the end and place them in `code/23cs30011_project/results/`.

(The same notebook can also be run locally with no GPU — it will skip generation and reuse a
previously-downloaded `results/task2_generation_results.jsonl` to redo scoring/analysis only.)

Writes: `results/task2_generation_results.jsonl`, `results/task2_per_question_results.csv`, `results/task2_metrics.json`.

**Iteration 2 (few-shot, final) — `code/task2/23cs30011_task2/task2_rag_fewshot.ipynb`**
Same retrieval pipeline and Colab workflow as Iteration 1; the only change is 3 few-shot exemplars
(drawn from the TriviaQA `train` split, never used for training — in-context demonstrations only)
prepended to the generator's prompt. Results go to separate `*_fewshot_*` files, so Iteration 1's
results are never overwritten and both remain independently reproducible.

Writes: `results/task2_fewshot_generation_results.jsonl`, `results/task2_fewshot_per_question_results.csv`, `results/task2_fewshot_metrics.json`.

#### Task 3 — two prompting iterations, both evaluated
Same pattern as Task 2: both iterations reuse Task 3's BM25+dense RRF fusion unchanged, and differ
only in the generator's prompt. Iteration 2 (few-shot) is again the final result.

**Iteration 1 (zero-shot) — `code/task3/23cs30011_task3/task3_hybrid_groundedness.ipynb`**
Same Colab workflow as Task 2 (needs the generator GPU): upload the same two Task 1 files when
prompted, run top-to-bottom, download the three result files into `code/23cs30011_project/results/`.

Writes: `results/task3_generation_results.jsonl`, `results/task3_per_question_results.csv`, `results/task3_metrics.json`.

**Iteration 2 (few-shot, final) — `code/task3/23cs30011_task3/task3_hybrid_groundedness_fewshot.ipynb`**
Same hybrid retrieval, groundedness check, and retry experiment as Iteration 1; only the
generator's prompt changes (same 3 exemplars as Task 2's Iteration 2). Same Colab workflow;
results go to separate `*_fewshot_*` files.

Writes: `results/task3_fewshot_generation_results.jsonl`, `results/task3_fewshot_per_question_results.csv`, `results/task3_fewshot_metrics.json`.

### Pretrained checkpoints used

- `sentence-transformers/all-MiniLM-L6-v2` (Tasks 2 & 3, dense retriever) — downloaded automatically from HuggingFace.
- `Qwen/Qwen2.5-7B-Instruct` (Tasks 2 & 3, generator, loaded in 4-bit via `bitsandbytes`) — downloaded automatically from HuggingFace.

---

## Running Bala Dharnesh's submission (23CS10008)

Bala's is a Docker-first, CLI-based submission (no notebook), with a one-command pipeline, a
pinned dependency environment, and its own unit test suite.

### Option A — one command, via Docker (what Bala's own setup recommends)

Requires Docker, an NVIDIA driver, and the NVIDIA Container Toolkit:

```bash
cd code/23cs10008_project
python code/scripts/run_docker_pipeline.py
```

This builds the image, prepares any missing dataset/corpus, rebuilds stale indexes, and runs
Task 1 → Task 2 → Task 3 → `audit_results.py` in sequence inside one disposable GPU container,
with `data/` and `artifacts/` bind-mounted so outputs survive. A completely fresh run needs
network access (it queries Wikipedia and downloads both model checkpoints) and can take a long
time; see `code/23cs10008_project/docker/README.md` for troubleshooting, and `--skip-build` to
reuse an existing image.

### Option B — task by task, locally

```bash
cd code/23cs10008_project
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-task1.txt   # CPU-only Task 1
python -m pip install -e .

python code/scripts/prepare_triviaqa.py --split validation --sample-size 300 --seed 42
python code/task1/build_corpus.py --pages-per-question 3 --delay-seconds 1.0 --retry-forever
python code/task1/build_index.py --k1 1.5 --b 0.75
python code/task1/run.py --answer-k 5 --recall-k 1 3 5 10

# Task 2/3 need a GPU and the full requirements.txt (not requirements-task1.txt):
python code/task2/build_index.py --device cuda --batch-size 128
python code/task2/run.py --device cuda --seed 42 --recall-k 1 3 5
python code/task3/run.py --device cuda --seed 42
```

Or, equivalently, run every task in one process: `python code/run_all.py --device cuda --seed 42`.

Each individual task also has a standalone, read-and-run copy under
`code/task1/23cs10008_task1/`, `code/task2/23cs10008_task2/`, and `code/task3/23cs10008_task3/`
respectively — the same commands above work from inside each of those folders too (just drop the
`code/` prefix, e.g. `python task1/run.py ...`), useful if you only want to inspect or run one
task without cloning his whole repository layout. Task 2 and 3 additionally need the Docker image
built from `docker/Dockerfile.task2` if running on the same 4GB-VRAM-class GPU he tested on; see
`code/23cs10008_project/docker/README.md`.

Verify results independently with `python -m pytest -q` and `python code/scripts/audit_results.py`
(both re-run from `code/23cs10008_project/`).

**Note:** this submission does not include a `report.pdf` of its own; its method, settings, and
results are documented in `code/23cs10008_project/RESULTS.md`, and we have folded those into the
relevant author subsections of our shared `report.pdf`/`report.tex`.

---

## Running Subham Ghosh's submission (22EE38015)

Subham's is a Colab-notebook-first submission with its own full report and results.

### Option A — Google Colab (what his own setup recommends)

1. Open `code/22ee38015_project/Run_Experiment.ipynb` in Colab.
2. Upload the project (and, optionally, its data snapshot) when prompted.
3. Run all cells — it runs preparation, Tasks 1/2/3, analysis, and report generation in order.

### Option B — locally

```bash
cd code/22ee38015_project
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
python -m pytest tests -q
python code/run.py all
```

Or stage by stage: `python code/run.py prepare|task1|task2|task3|analyze|report`. All scripts
resolve their own paths independently of the working directory. We verified `python -m pytest
tests -q` passes unmodified in this environment (14/14).

Each individual task's library module is also copied, unmodified, under
`code/task1/22ee38015_task1/`, `code/task2/22ee38015_task2/`, and `code/task3/22ee38015_task3/`
for side-by-side reading next to the other two submissions' code. Of those, only Task 1
(`code/task1/22ee38015_task1/task1/baseline.py`) is independently runnable — it has no
cross-task dependency beyond `numpy`/`rank_bm25`/NLTK. Tasks 2 and 3 are library modules with no
CLI of their own (dispatched by `common/experiment.py` via `code/run.py task2`/`task3`, as
explained above), so run those from `code/22ee38015_project/` rather than from their per-task
copies.

His own saved results (`results/`) are kept at `code/22ee38015_project/`; his report was folded
into our shared `report.pdf`, same as the other two submissions.

---

## Report

`report.pdf` (built from `report.tex`) is **one shared report covering all three submissions**.
Its section structure mirrors the assignment (Task 1 → Task 2 → Task 3 → Discussion), and inside
each Task section there is one subsection per author — same subsection headings
(Method / Quantitative Results / Qualitative Analysis) in the same order for all three — followed
by a short subsection comparing all three approaches on that task. Compile it yourself with:

```bash
tectonic report.tex   # or: pdflatex report.tex (run twice for the table of contents)
```
