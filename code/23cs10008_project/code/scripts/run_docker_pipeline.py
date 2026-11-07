#!/usr/bin/env python3
"""Build the CUDA image and run the complete pipeline with one host command."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="odqa:local")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an image that is already built.")
    parser.add_argument("--rebuild-indexes", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile.task2", "-t", args.image, "."],
            cwd=ROOT,
            check=True,
        )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{ROOT / 'data'}:/app/data",
        "-v",
        f"{ROOT / 'artifacts'}:/app/artifacts",
        args.image,
    ]
    if args.rebuild_indexes:
        command.extend(["python", "code/run_all.py", "--device", "cuda", "--rebuild-indexes"])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
