from __future__ import annotations

import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Callable

from src.agents.agents import configure_torch_runtime


def make_training_job(
    *,
    experiment_name: str,
    train_function: Callable,
    training_seeds: list[int | str],
    test_seeds: list[int | str],
    settings,
    reverse_positions: bool,
    results_dir: str | Path,
) -> dict:
    return {
        "experiment_name": experiment_name,
        "train_function_path": f"{train_function.__module__}:{train_function.__name__}",
        "training_seeds": list(training_seeds),
        "test_seeds": list(test_seeds),
        "settings": deepcopy(settings),
        "reverse_positions": bool(reverse_positions),
        "result_path": str(Path(results_dir) / f"{experiment_name}.csv"),
    }


def _load_train_function(function_path: str) -> Callable:
    module_name, function_name = function_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _run_training_job(job: dict) -> dict:
    settings = job["settings"]
    configure_torch_runtime(settings)

    train_function = _load_train_function(job["train_function_path"])
    result_path = Path(job["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)

    df = train_function(
        e_name=job["experiment_name"],
        training_seeds=job["training_seeds"],
        test_seeds=job["test_seeds"],
        settings=settings,
        reverse_positions=job["reverse_positions"],
    )

    df.to_csv(result_path, index=False)

    return {
        "experiment_name": job["experiment_name"],
        "rows": len(df),
        "result_path": str(result_path),
    }


def run_parallel_training_jobs(
    jobs: list[dict],
    max_workers: int = 4,
) -> list[dict]:
    if len(jobs) == 0:
        return []

    max_workers = max(1, min(int(max_workers), len(jobs)))
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_training_job, job): job["experiment_name"]
            for job in jobs
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[done] {result['experiment_name']} "
                f"rows={result['rows']} -> {result['result_path']}",
                flush=True,
            )

    return results
