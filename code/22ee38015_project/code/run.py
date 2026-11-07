"""Single entry point; individual stages remain independently inspectable.

Examples: python code/run.py all ; python code/run.py task1
Run from any directory. The project root comes from common.io.ROOT.
"""
import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "task1", "task2", "task3", "analyze", "report", "all"])
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        from prepare import prepare
        prepare()
    if args.stage in ("task1", "task2", "task3", "all"):
        from common.experiment import run_task
        tasks = ["task1", "task2", "task3"] if args.stage == "all" else [args.stage]
        resources = {}
        for task in tasks:
            resources = run_task(task, resources)
    if args.stage in ("analyze", "all"):
        from analyze import analyze
        analyze()
    if args.stage in ("report", "all"):
        from make_report import make_report
        make_report()


if __name__ == "__main__":
    main()
