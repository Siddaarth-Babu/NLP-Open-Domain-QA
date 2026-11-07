#!/usr/bin/env python3
"""Create the required source-only submission archive after validation."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/open-domain-qa-submission.tar.gz"))
    args = parser.parse_args()
    required = [ROOT / "README.md", ROOT / "requirements.txt", ROOT / "report.pdf"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot package submission; missing required file(s): {', '.join(missing)}")
    sources = sorted((ROOT / "code").rglob("*.py"))
    if not sources:
        raise SystemExit("Cannot package submission; code/ contains no Python sources.")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in [*required, *sources]:
            archive.add(path, arcname=path.relative_to(ROOT))
    print(f"Wrote {output} with {len(sources)} Python source files and the three required root files.")


if __name__ == "__main__":
    main()
