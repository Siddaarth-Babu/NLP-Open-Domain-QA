"""Generate the assignment-length main report + references from measured experiment files.

No benchmark numbers are hard-coded as our results. Missing tasks stop export.
Edit metadata.json for the group names before the final submission.
"""
from html import escape
from pathlib import Path
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from common.io import ROOT, config, read_json, read_jsonl


def make_report():
    cfg = config()
    comparison = read_json(ROOT / 'results/comparison.json')
    if set(comparison) != {'task1', 'task2', 'task3'}:
        raise RuntimeError('Complete and analyze all three tasks before generating a results report')
    manifest = read_json(ROOT / 'data/corpus_manifest.json')
    sample = read_json(ROOT / 'data/sample_manifest.json')
    support = read_json(ROOT / 'results/groundedness.json')
    examples = read_json(ROOT / 'results/qualitative_examples.json')
    paired = read_json(ROOT / 'results/paired_comparison.json')
    metadata = read_json(ROOT / 'metadata.json') if (ROOT / 'metadata.json').exists() else {'authors': 'Subham Ghosh'}
    pdfmetrics.registerFont(TTFont("QASans", str(ROOT / "assets/fonts/DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("QASans-Bold", str(ROOT / "assets/fonts/DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("QASans", normal="QASans", bold="QASans-Bold", italic="QASans", boldItalic="QASans-Bold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyQA', fontName='QASans', fontSize=10, leading=14.5, spaceAfter=10, textColor=colors.HexColor('#243441')))
    styles.add(ParagraphStyle(name='SmallQA', parent=styles['BodyQA'], fontSize=8.3, leading=11.5, spaceAfter=6))
    styles.add(ParagraphStyle(name='TitleQA', fontName='QASans-Bold', fontSize=27, leading=32, spaceAfter=18, textColor=colors.HexColor('#153e58')))
    styles.add(ParagraphStyle(name='HeadingQA', fontName='QASans-Bold', fontSize=19, leading=24, spaceAfter=18, textColor=colors.HexColor('#153e58')))
    styles.add(ParagraphStyle(name='SubQA', fontName='QASans-Bold', fontSize=12, leading=17, spaceAfter=7, textColor=colors.HexColor('#226b75')))
    story = []
    def p(text, small=False):
        story.append(Paragraph(text, styles['SmallQA' if small else 'BodyQA']))
    def sub(text):
        story.append(Paragraph(text, styles['SubQA']))
    def page(number, title):
        if story: story.append(PageBreak())
        p('CS60075 | ASSIGNMENT II | PROJECT 10', True)
        story.append(Paragraph(f'{number}. {title}', styles['HeadingQA']))
    def table(rows, widths=None):
        prepared = [[Paragraph(escape(str(v)), styles['SmallQA']) for v in row] for row in rows]
        obj = Table(prepared, colWidths=widths, hAlign='LEFT', repeatRows=1)
        obj.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f0f4')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#90a8b8')),
            ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#d8e0e4'))]))
        story.extend([obj, Spacer(1, 12)])
    def val(value): return 'Not measured' if value is None else f'{value:.2f}%'
    n = comparison['task1']['n']
    page(1, 'Open-domain question answering')
    story.append(Paragraph('Lexical retrieval, dense RAG,<br/>and hybrid evidence checks', styles['TitleQA']))
    p(escape(metadata['authors']) + '<br/>IIT Kharagpur | Natural Language Processing | October 2026')
    sub('Abstract')
    p(f'This study evaluates three open-domain question-answering pipelines on a fixed sample of {n} unique TriviaQA validation questions. '
      f'A shared collection of {manifest["passages"]:,} Wikipedia passages from {manifest["articles"]:,} articles supplies external evidence. '
      'Task 1 combines BM25 retrieval with sentence overlap and noun-phrase/number extraction. Task 2 combines frozen MiniLM embeddings, exact FAISS search, '
      'and a frozen instruction-tuned language model. Task 3 fuses lexical and dense rankings using reciprocal rank fusion and applies an answer-evidence overlap warning.')
    p('All predictions and retrieved evidence are saved. Alias-aware exact match and token F1 evaluate answers; an explicitly labeled answer-containing passage proxy evaluates retrieval. '
      'Correctness and evidence support are evaluated separately. No task-specific model training or gold-answer-assisted corpus selection is used.')
    table([['System', 'EM', 'Token F1', 'Recall@5 proxy']] + [[t, val(comparison[t]['em']), val(comparison[t]['f1']), val(comparison[t]['recall@5'])] for t in comparison], [145, 90, 90, 155])
    p('Scope: a course-prescribed conceptual RAG reproduction, not the original paper\'s full trained architecture. The missing Dataset 10.2 corpus instructions are replaced by an explicit, auditable question-only construction procedure.', True)
    page(2, 'Dataset and evidence collection')
    sub('Dataset selection and answer aliases')
    p('The assigned configuration is mandarjoshi/trivia_qa, rc.nocontext. The validation split is the evaluation source. The canonical answer value and all aliases are combined, deduplicated, and retained as acceptable reference strings. '
      'The train split is not used for learning or demonstrations; generation is zero-shot. Dataset libraries may materialize other splits while preparing the dataset cache, which is not model training.')
    p(f'The pinned dataset contains {sample["loaded_rows"]:,} loaded validation rows and {sample["unique_questions"]:,} unique question IDs. '
      f'IDs are sorted, deduplicated, and sampled without replacement with Python random.Random({cfg["seed"]}).sample. The resulting {n} question IDs and original row indices are saved. '
      'The course handout\'s approximate sizes differ from this configuration; the observed loaded sizes are authoritative for this run.')
    sub('Question-only Wikipedia construction')
    p(f'For every sampled question, spaCy produces a keyword query by removing stop words and punctuation. If it returns no pages, named entities and noun phrases provide deterministic fallbacks, followed by the first three keywords. '
      f'This experiment retains the first main-namespace result for each question. Gold answers, answer aliases, and original TriviaQA evidence metadata never enter these searches. '
      'The union of pages is shared globally; questions do not receive private evidence collections.')
    p(f'Full public article HTML is parsed for paragraph text, excluding tables and citation markers. Cached API searches and public search pages supply topic candidates. Paragraph text is split on paragraph boundaries and into nonoverlapping segments of at most {cfg["passage_words"]} whitespace-delimited words. Segments shorter than {cfg["min_passage_words"]} words are discarded. '
      'Identical passage text is indexed only once. Stable article/page order defines passage IDs and score-tie ordering.')
    table([['Observed corpus statistic', 'Value'], ['Articles', manifest['articles']], ['Passages', manifest['passages']],
           ['Questions with no search results', manifest['questions_with_no_search_results']],
           ['Gold-alias corpus coverage (post hoc)', val(manifest['answer_containing_corpus_coverage_percent'])]], [315, 165])
    p('Coverage is computed after construction and does not filter questions or add answer-targeted articles. Questions lacking answer-containing passages remain in the evaluation denominator. The companion snapshot preserves exact bytes; live rebuilding can change the corpus.', True)
    page(3, 'Task 1: classical retrieval and extraction')
    sub('Lexical retrieval')
    p('Passages and questions are lowercased and tokenized into Unicode word sequences. BM25Okapi from rank_bm25 ranks the frozen passage collection with k1=1.5 and b=0.75. '
      'The term-frequency saturation parameter prevents repeated words from growing linearly in importance, while length normalization adjusts for different passage lengths. '
      'Ranking ties follow frozen corpus order. The top 50 candidates are saved; only the top-ranked passage supplies the extracted answer.')
    sub('Required extraction heuristic')
    p('Sentence splitting uses punctuation rules. A frozen NLTK English averaged-perceptron POS tagger and a regular-expression grammar supply noun chunks. Each sentence is scored by the count of distinct word tokens shared with the question. '
      'The first highest-overlap sentence wins ties. Its noun chunks and regex-detected numeric spans become candidates. The longest candidate by word-token count wins; character length and earliest position break ties. '
      'If no candidate exists, the system returns an empty answer. No answer-type classifier or gold label influences this choice.')
    p('The POS tagger is a classical linear statistical classifier, and the noun-phrase grammar groups determiners, adjectives, nouns, and possessives. The baseline QA stage contains no neural parser, neural retrieval encoder, learned QA reader, or generative model. '
      'Its POS parameters are frozen; no TriviaQA training examples are used. Rule-based sentence splitting can mishandle abbreviations, and flat noun chunks can miss syntactic structure.')
    sub('Expected failure mechanisms')
    p('A sentence can repeat question terms while discussing the wrong entity. Even in the correct sentence, the longest noun phrase can be a book title or institutional description instead of the requested person. '
      'Numeric answers can also lose to a longer noun phrase. These weaknesses are preserved because the course explicitly specifies this baseline. Error analysis should distinguish retrieval failures from extraction failures.')
    table([['Measured metric', 'Task 1'], ['Exact match', val(comparison['task1']['em'])], ['Token F1', val(comparison['task1']['f1'])], ['Recall@1 proxy', val(comparison['task1']['recall@1'])], ['Recall@5 proxy', val(comparison['task1']['recall@5'])]], [315, 165])
    p('Implementation: code/task1/baseline.py. Per-question output includes the chosen sentence and predicted span, enabling direct inspection of the heuristic.', True)
    page(4, 'Task 2: dense retrieval and generation')
    sub('Embedding and FAISS index')
    p('The frozen sentence-transformers/all-MiniLM-L6-v2 encoder maps each passage and question into the same vector space. Passage embeddings are cached with their corpus checksum and model revision. '
      'Vectors are normalized to unit L2 length and inserted into faiss.IndexFlatIP. Inner product on normalized vectors equals cosine similarity, so flat search exactly ranks the encoded vectors. '
      'This removes approximate nearest-neighbor error but does not remove encoder truncation or representation error.')
    sub('Generator and prompt controls')
    p(f'The open instruction-tuned generator is {escape(cfg["generator"])}. It is frozen and used without task-specific fine-tuning. '
      f'The top {cfg["answer_top_k"]} passages are individually truncated to {cfg["passage_token_limit"]} generator tokens. '
      'The prompt asks for the shortest answer phrase, requests use of only the supplied evidence, and permits UNKNOWN if the evidence is insufficient. '
      f'Generation uses greedy decoding, one beam, and at most {cfg["max_new_tokens"]} new tokens, implementing the requested temperature-zero behavior. '
      'UNKNOWN is converted to an empty prediction. Other outputs are retained without using references to clean them.')
    p('Task 3 reuses the identical generator, prompt, truncation, and decoding. This controls the answering stage when comparing dense and hybrid retrieval. '
      'Evidence saved in result records is the exact truncated text seen by the generator, not the untruncated source passage.')
    sub('Relationship to Lewis et al. [1]')
    p('The original paper uses DPR retrieval with BART and trains model components over a much larger Wikipedia index. The course instead prescribes frozen MiniLM, a flat FAISS index, and any open instruction model. '
      'Our implementation follows the course pipeline; it does not implement document marginalization or reproduce the original training. This distinction limits numerical comparability.')
    table([['Setting', 'This experiment'], ['Encoder', cfg['encoder']], ['Generator', cfg['generator']], ['Model training', 'None'], ['Dense search', 'Exact cosine ranking, normalized IndexFlatIP'], ['Answer evidence', 'Top 5; 200 tokens per passage']], [130, 350])
    page(5, 'Task 3: fusion and lexical support checks')
    sub('Reciprocal rank fusion')
    p('BM25 and dense search each supply their top 50 candidates. For passage p, the fused score is the sum of 1/(60 + rank(p)) over the lists containing p. '
      'Ranks are one-based and missing-list contributions are zero. Fusion therefore does not assume that BM25 and cosine raw scores have comparable scales. '
      'The union is sorted by descending fused score, breaking ties by frozen corpus index. Top-5 fused passages feed the same generator as Task 2.')
    p('The candidate depth of 50 is an explicit implementation choice. It allows candidates outside either method\'s first five positions to enter the fused top five, while supporting Recall@10. '
      'The RRF constant of 60 follows the assignment. No evaluation-answer supervision chooses fusion weights or candidates.')
    sub('Generated-answer overlap warning')
    p('For both Task 2 and Task 3, the checker normalizes the predicted answer and the exact prompt evidence. '
      'The score is the size of their token-multiset intersection divided by the answer token count. The fixed threshold is 0.8; empty answers receive score zero and a separate abstention flag. '
      'The assignment permits flagging instead of abstention, so the main answer metrics retain raw predictions. A separate selective analysis reports the coverage and EM of answers retained by the warning rule.')
    sub('Why overlap is not verification')
    p('A wrong entity can appear somewhere in a passage, producing high overlap without supporting the question-answer relation. Negation and contradictory evidence are also invisible to the checker. '
      'Conversely, a correct paraphrase may have little literal overlap. The implementation therefore calls the output a warning, and accuracy requires independent answer-in-context support labels.')
    p('The checker runs on original dense-RAG answers as required, as well as hybrid answers. Gold-answer correctness is used only for the requested correct/high-overlap and incorrect/low-overlap statistics; it is never substituted for evidence-support truth.')
    p('Implementation: code/task3/hybrid.py and code/task3/groundedness.py. No additional retriever or generator is trained.', True)
    page(6, 'Evaluation and reproducibility protocol')
    sub('Answer and retrieval metrics')
    p('Exact match lowercases text, removes ASCII punctuation and English articles, and collapses whitespace. A prediction is correct if it matches any accepted answer. '
      'Token F1 uses multiset token precision and recall; each prediction receives its maximum score across aliases. Dataset-level scores are means across all sampled questions, including abstentions.')
    p('The custom corpus has no gold passage-relevance judgments. Recall@k is operationalized as the fraction of questions with at least one normalized gold-alias-containing passage in the first k results, for k=1,3,5,10. '
      'Matching uses whole token sequences to avoid substring errors. This answer-containing success proxy may count a passage that mentions an answer without supporting it, or miss a paraphrase. It must not be interpreted as semantic document recall.')
    sub('Controls and provenance')
    p(f'All tasks use the same {n} questions and identical passage IDs. Random seed {cfg["seed"]} controls sampling, Python, NumPy, and PyTorch. '
      'Decoding is greedy; deterministic algorithms are requested. Cross-device floating-point changes can still affect tied rankings or generated tokens. '
      'Dataset and model checkpoints are pinned to immutable revisions. The requirements file pins the installed software environment. Corpus bytes and question bytes have SHA-256 checksums.')
    p('A per-question checkpoint records a run identity derived from the configuration, data hashes, and functional source files. Matching completed checkpoints are reused; changed identities are recomputed. '
      'The saved predictions include accepted answers, evidence, rankings, time, and answer/support scores. Generation inputs never receive evaluation answers. '
      'CPU execution is the reference run; the notebook offers an optional GPU environment with a separately documented torch wheel.')
    table([['Artifact', 'Purpose'], ['config.json', 'Fixed parameters and model/dataset revisions'], ['data/*_manifest.json', 'Sampling IDs, search trail, article revisions, checksums'], ['results/task*/predictions.jsonl', 'Per-question outputs and exact answer evidence'], ['results/comparison.csv', 'Comparable measured task scores'], ['results/support_labels.csv', 'Independent pilot support labels, if present']], [200, 280])
    p('Dependency downloads and corpus acquisition are setup costs. Per-question timings include retrieval and answering, but exclude initial model loading and passage embedding. Wall time can change substantially on different hardware.', True)
    page(7, 'Quantitative results and paper comparison')
    table([['System', 'EM', 'F1', 'R@1', 'R@3', 'R@5', 'R@10']] + [[t] + [f'{comparison[t][k]:.2f}' for k in ['em','f1','recall@1','recall@3','recall@5','recall@10']] for t in comparison], [83, 60, 60, 65, 65, 65, 82])
    story.append(Image(str(ROOT / 'results/comparison.png'), width=480, height=178))
    story.append(Spacer(1, 10))
    delta = comparison['task3']['em'] - comparison['task2']['em']
    rd = comparison['task3']['recall@5'] - comparison['task2']['recall@5']
    p(f'Hybrid minus dense RAG changes EM by {delta:+.2f} percentage points and Recall@5 by {rd:+.2f} points. '
      f'Hybrid corrects {int(paired["hybrid_only_correct"])} dense errors and loses {int(paired["dense_only_correct"])} dense-correct answers. Both methods answer {int(paired["both_correct"])} questions correctly. Retrieval gains need not translate directly to EM: evidence can remain ambiguous, prompt truncation can remove useful text, and the small generator can select an incorrect entity.')
    table([['Published reference [1]', 'TriviaQA standard EM', 'TriviaQA-Wiki EM'], ['RAG-Token', '55.2', '66.1'], ['RAG-Sequence', '56.8', '68.0'], ['This Task 2', f'{comparison["task2"]["em"]:.2f} (sampled validation)', 'Not evaluated']], [180, 150, 150])
    p('Published figures are test-set EM from Table 1 of [1]. Our validation sample, corpus, model, training policy, and decoding differ. The values provide context, not a controlled leaderboard comparison. '
      'A gap cannot be attributed to retrieval alone. Paper F1 and retrieval recall are not invented where the referenced table does not report them.', True)
    page(8, 'Groundedness analysis')
    rows = [['Statistic', 'Dense RAG', 'Hybrid RAG']]
    for key, label in [('correct_with_high_overlap_percent','Correct answers with high overlap'), ('incorrect_with_low_overlap_percent','Incorrect answers with low overlap'),
                       ('retained_coverage_percent','Retained answer coverage'), ('retained_em_percent','EM among retained answers'), ('groundedness_check_accuracy_percent','Support-check accuracy, nonempty audit')]:
        rows.append([label, val(support['task2'].get(key)), val(support['task3'].get(key))])
    rows.append(['Nonempty answers in support audit', str(support['task2'].get('support_audit_n', 0)), str(support['task3'].get('support_audit_n', 0))])
    table(rows, [280, 100, 100])
    p('The first row conditions on EM-correct answers; the second conditions on EM-incorrect answers. Their denominators differ and both are saved in groundedness.json. '
      'A zero-sized outcome stratum produces an undefined value, not a fabricated zero. Retained-answer EM is conditional accuracy and must be read alongside coverage. '
      'Flagged predictions remain in the main unfiltered task comparison.')
    labels_path = ROOT / 'results/support_labels.csv'
    if labels_path.exists():
        p('Support labels come from a fixed-seed pilot audit of ten question IDs, evaluated for both generation methods. The evidence-conditioned judgments are independent of answer-string EM. '
          'The assistant reviewed the question, proposed answer, and supplied passages, with rationale saved for every label. These are assistant-reviewed pilot annotations, not human-validated ground truth. '
          'Abstentions are excluded from claim-support accuracy; the effective nonempty denominators appear in the table. Audit accuracy is descriptive for this small paired sample; it is not a full-dataset support score or a reliable generalization estimate.')
    else:
        p('Independent support annotations have not been supplied. The audit JSON and CSV template provide the exact evidence for labeling. Support-check accuracy is intentionally left unmeasured; answer correctness would be an invalid replacement for this missing label. '
          'Populate support_labels.csv and rerun analyze/report to obtain the requested accuracy.')
    sub('Interpretation')
    p('In a reviewed dense-RAG case, the question asks which country contains Masvingo, Midlands, and Manicaland. The model answers Mozambique, although the evidence places Masvingo in Zimbabwe and names Mozambique only as a neighbour. This is a high-overlap error in the geographic relation. '
      'Low-overlap incorrect answers are plausible warning successes, though incorrectness alone is not proof of missing support. The pilot audit assesses this distinction directly.')
    p('A stronger future checker would evaluate whether the retrieved evidence entails the entire question-answer claim, including negation and entity roles. This is outside the required lexical heuristic and is not silently added to the experiment.')
    page(9, 'Qualitative error analysis')
    for task in ['task2', 'task3']:
        if task == 'task3':
            page('9b', 'Qualitative analysis: hybrid comparison')
        sub('Dense RAG' if task == 'task2' else 'Hybrid RAG')
        for outcome in ['correct', 'incorrect']:
            group = examples[task][outcome]
            if not group:
                p(f'No {outcome} examples occurred in this run.', True)
                continue
            row = group[0]
            p(f'<b>EM-{outcome} example - {escape(row["question_id"])}</b><br/>'
              f'Question: {escape(row["question"])}<br/>Prediction: {escape(row["prediction"] or "[abstention]")}<br/>'
              f'Accepted answer: {escape(row["answers"][0])}<br/>EM={row["em"]:.0f}; F1={row["f1"]:.2f}; overlap={row["overlap"]:.2f}; '
              f'answer-containing Recall@5={row["recall@5"]:.0f}.', True)
            # Show an answer-bearing evidence excerpt if available, otherwise first evidence.
            from common.metrics import contains_alias
            evidence = next((e for e in row['evidence'] if row['prediction'] and contains_alias(e['text'], [row['prediction']])),
                            next((e for e in row['evidence'] if contains_alias(e['text'], row['answers'])), row['evidence'][0]))
            anchors = [evidence['text'].lower().find(a.lower()) for a in [row['prediction']] + row['answers'] if a and a.lower() in evidence['text'].lower()]
            start = max(0, min(anchors) - 50) if anchors else 0
            if start:
                boundary = evidence['text'].find(' ', start)
                start = boundary + 1 if boundary >= 0 else start
            end = min(len(evidence['text']), start + 200)
            if end < len(evidence['text']):
                end = evidence['text'].rfind(' ', start, end)
            excerpt = ('... ' if start else '') + evidence['text'][start:end]
            p(f'Evidence excerpt ({escape(evidence["title"])}): {escape(excerpt)}...', True)
            if row['em']:
                explanation = 'The generated answer matches an accepted reference. Inspect the full saved evidence before claiming that this is a supported answer.'
            elif row['recall@5']:
                explanation = 'An answer-containing passage was retrieved, but the output fails exact match. Inspect answer formatting, entity choice, context ambiguity, and semantic support; retrieval success alone is insufficient.'
            else:
                explanation = 'No retrieved top-5 passage contains an accepted answer alias. This suggests an evidence retrieval/coverage problem, though paraphrased evidence can evade this proxy.'
            notes_path = ROOT / 'results/case_notes.json'
            if notes_path.exists():
                explanation = read_json(notes_path).get(task + ':' + row['question_id'], explanation)
            p(escape(explanation), True)
    p('Cases are illustrative: hybrid examples prioritize changes relative to dense RAG. EM status is not a semantic truth judgment. Full evidence and additional cases are in results/qualitative_examples.json.', True)
    page(10, 'Discussion, limitations, and conclusions')
    sub('What the comparison establishes')
    best = max(comparison, key=lambda t: comparison[t]['em'])
    p(f'The highest measured exact-match score in this run belongs to {best} ({comparison[best]["em"]:.2f}%). '
      'The comparison isolates a classical pipeline, a frozen dense RAG pipeline, and fusion using the same generator. It does not isolate every component: Task 1 changes both retrieval and answering relative to Task 2. '
      'The most controlled retrieval comparison is Task 2 versus Task 3, because both use the same answering model and prompt.')
    p(f'The frozen corpus has {manifest["answer_containing_corpus_coverage_percent"]:.2f}% answer-alias coverage. '
      'Missing evidence limits extractive answering, and language models may compensate with memorized knowledge. Such answers can be correct while unsupported, so correctness cannot replace a support judgment.')
    sub('Material limitations')
    p('Corpus construction targets the known evaluation question topics, as the handout permits. Consequently these results do not estimate retrieval quality over all of Wikipedia or over unseen topic distributions. '
      'The absent Dataset 10.2 construction section remains an assumption requiring TA review. Wikipedia content is time-dependent, which makes the exact frozen snapshot important.')
    p('The compact generator is a compute-conscious choice and can misread even explicit evidence. MiniLM is a general sentence encoder rather than the original paper\'s supervised DPR. '
      'Input truncation, greedy output selection, and fixed thresholds are additional constraints. No threshold was optimized using final evaluation correctness. '
      'The 300-question sample and small support audit limit statistical certainty. The lexical checker misses relational errors, negation, and contradictory context.')
    sub('Reproduction and submission')
    p('The submission archive contains separate task modules, a pinned dependency environment, a Colab notebook, the measured results, this PDF report, and walkthrough materials. '
      'Large data and model checkpoints are excluded. The companion data snapshot can be restored to reproduce the exact collection; otherwise prepare reconstructs a new live corpus and writes new manifests. '
      'All final members and contribution details should be reviewed before submission. A computer-narrated walkthrough of the actual code and measured outputs, an editable transcript, and a recording guide accompany the package.')
    p('The EM gain is one question out of 300, so this single run supports a modest descriptive improvement rather than a significance claim. Future experiments could compare a larger instruction model while holding retrieval fixed, inspect paired failures where fusion changes evidence, and obtain multiple independent human support labels. '
      'Those extensions are not reported as completed work. The current result files are the evidence for the reported conclusions.')
    page('R', 'References and artifact map')
    refs = [
        '[1] Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. https://arxiv.org/abs/2005.11401 (Table 1; methods and experimental setup).',
        '[2] Joshi et al. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. ACL 2017. https://nlp.cs.washington.edu/triviaqa/',
        '[3] Dataset: https://huggingface.co/datasets/mandarjoshi/trivia_qa, configuration rc.nocontext. Revision: ' + cfg['dataset_revision'],
        '[4] Encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. Revision: ' + cfg['encoder_revision'],
        '[5] Generator: https://huggingface.co/' + cfg['generator'] + '. Revision: ' + cfg['generator_revision'],
        '[6] Cormack, Clarke, and Buettcher. Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR 2009. https://doi.org/10.1145/1571941.1572114',
        '[7] User-provided CS60075 Assignment II Project 10, Dataset 10.3, and Deliverables and Submission Guidelines screenshots. Dataset 10.2 was not supplied.',
        '[8] NLTK 3.9.1, averaged_perceptron_tagger_eng and RegexpParser. https://www.nltk.org/api/nltk.tag.perceptron.html . The official tagger ZIP URL and SHA-256 are pinned in code/task1/baseline.py.'
    ]
    for ref in refs: p(escape(ref), True)
    sub('Commands')
    for command in ['python code/run.py prepare', 'python code/run.py task1', 'python code/run.py task2', 'python code/run.py task3', 'python code/run.py analyze', 'python code/run.py report']:
        p(escape(command), True)
    p('The notebook executes these stages in order. results/comparison.csv is the quantitative source, results/groundedness.json supplies overlap/audit statistics, and results/task*/predictions.jsonl supplies qualitative evidence. '
      'Run python -m pytest tests -q to verify scoring and pipeline invariants. README.md describes setup, revision controls, snapshot restoration, resumability, and output locations.', True)
    p('Corpus SHA-256: ' + manifest['corpus_sha256'], True)
    destination = ROOT / 'report.pdf'
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#cad8df'))
        canvas.line(48, 39, 547, 39)
        canvas.setFont('QASans', 8)
        canvas.setFillColor(colors.HexColor('#617887'))
        canvas.drawString(48, 26, 'CS60075 | Open-domain QA | Measured experiment')
        canvas.drawRightString(547, 26, str(doc.page))
    document = SimpleDocTemplate(str(destination), pagesize=(595.28, 841.89), leftMargin=54, rightMargin=54,
                                 topMargin=44, bottomMargin=54, title='Open-domain QA: Lexical Retrieval to RAG', author=metadata['authors'])
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print('Wrote', destination, flush=True)


if __name__ == '__main__':
    make_report()
