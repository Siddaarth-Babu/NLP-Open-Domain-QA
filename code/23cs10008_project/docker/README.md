# Docker execution

Docker keeps the pinned software image separate from generated data. Run commands from the repository root. Both images bind-mount `data/` and `artifacts/`; those directories are not copied into images or the submission.

## Entire pipeline with one command

Prerequisites are Docker, an NVIDIA GPU/driver, and NVIDIA Container Toolkit. Verify GPU passthrough first:

```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-base-ubuntu24.04 nvidia-smi
```

Then build the GPU image and run dataset preparation, stale-index construction, Tasks 1–3, and the result audit with one host command:

```bash
python code/scripts/run_docker_pipeline.py
```

The helper calls Docker through an argument-safe Python subprocess; it requires no Compose plugin. Use `--skip-build` to reuse an existing `odqa:local` image or `--rebuild-indexes` to force both indexes to be rebuilt. On a fresh workspace it downloads TriviaQA and model checkpoints and performs rate-limited Wikipedia discovery. Completed question responses and model files are cached under `data/`, so interrupted or later runs reuse them. The corpus builder retries transient MediaWiki/network errors indefinitely when invoked by the orchestrator.

After the image exists, the equivalent direct command is:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  odqa:local
```

Pass `python code/run_all.py --device cuda --rebuild-indexes` after the image name to force both indexes to be rebuilt.

## Lightweight Task 1 image

Task 1 does not need CUDA or neural packages:

```bash
docker build -f docker/Dockerfile -t odqa-task1:local .

docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  odqa-task1:local \
  python code/task1/run.py --answer-k 5 --recall-k 1 3 5 10
```

To build its prerequisites separately, replace the command after the image name with, in order:

```text
python code/scripts/prepare_triviaqa.py --split validation --sample-size 300 --seed 42
python code/task1/build_corpus.py --pages-per-question 3 --delay-seconds 1.0 --retry-forever
python code/task1/build_index.py --k1 1.5 --b 0.75
```

## GPU image and individual neural tasks

```bash
docker build -f docker/Dockerfile.task2 -t odqa:local .

docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  odqa:local \
  python code/task2/build_index.py --device cuda --batch-size 128

docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  odqa:local \
  python code/task2/run.py --device cuda --seed 42 --recall-k 1 3 5

docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  odqa:local \
  python code/task3/run.py --device cuda --seed 42
```

`docker/Dockerfile.task2` installs the CUDA 13.0 PyTorch wheel before the complete pinned `requirements.txt`. Triton requires the included compiler during its first bitsandbytes generation kernels.

## Image, container, and persistent files

| Item | Location | Lifetime |
| --- | --- | --- |
| Python, dependencies, code, configs | Docker image | Rebuilt when definitions change |
| One pipeline process | Disposable container | Removed by `--rm` |
| Dataset, Wikipedia cache, indexes, models | Host `data/` | Persistent and regenerable |
| Predictions, metrics, audit | Host `artifacts/` | Persistent and regenerable |

If `docker run task1-corpus-build` fails, remember that `task1-corpus-build` was a container name, not an image. Start a command from an image such as `odqa-task1:local` or use `docker start task1-corpus-build` only when that stopped container still exists.
