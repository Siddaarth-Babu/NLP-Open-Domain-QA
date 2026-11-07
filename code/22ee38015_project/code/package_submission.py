"""Create the requested source/report/results archive; exclude large data/models."""
import zipfile
from pathlib import Path
from common.io import ROOT


def package():
    archive = ROOT.parent / "NLP_QA_Submission.zip"
    include_suffixes = {".py", ".ipynb", ".md", ".txt", ".json", ".csv", ".jsonl", ".png", ".pdf", ".pptx", ".ttf", ".mp4"}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.suffix not in include_suffixes:
                continue
            relative = path.relative_to(ROOT)
            if any(part in ("__pycache__", ".pytest_cache", "cache", "records") for part in relative.parts):
                continue
            # Only small provenance manifests belong in the submitted data folder.
            if relative.parts[0] == "data" and path.name not in (
                "sample_manifest.json", "corpus_manifest.json", "search_manifest.json", "preparation_errors.json"):
                continue
            bundle.write(path, Path(ROOT.name) / relative)
    print(archive, archive.stat().st_size, "bytes")


if __name__ == "__main__":
    package()
