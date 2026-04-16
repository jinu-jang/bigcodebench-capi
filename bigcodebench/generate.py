from __future__ import annotations

import json
import os
from typing import Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from bigcodebench.data import get_bigcodebench, write_jsonl
from bigcodebench.provider import DecoderBase, make_model
from bigcodebench.sanitize import sanitize


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_existing_sample_counts(target_path: str) -> dict[str, int]:
    sample_counts: dict[str, int] = {}
    if not os.path.exists(target_path):
        return sample_counts

    with open(target_path, "r") as file:
        for line in file:
            item = json.loads(line)
            task_id = item["task_id"]
            sample_counts[task_id] = sample_counts.get(task_id, 0) + 1
    return sample_counts


def _should_flush_batch(
    *,
    batch_size: int,
    batch_prompts: list[str],
    item_index: int,
    item_count: int,
    id_range: tuple[int, int] | None,
) -> bool:
    return (
        (batch_size and len(batch_prompts) == batch_size)
        or item_index == item_count - 1
        or (id_range is not None and item_index == id_range[1] - 1)
    )


def _build_samples(
    *,
    model: DecoderBase,
    task_ids: list[str],
    prompts: list[str],
    entry_points: list[str],
    sample_counts: list[int],
    outputs: list[list[str]],
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for task_id, prompt, entry_point, requested_samples, task_outputs in zip(
        task_ids,
        prompts,
        entry_points,
        sample_counts,
        outputs,
    ):
        if model.is_direct_completion():
            samples.extend(
                {
                    "task_id": task_id,
                    "solution": sanitize(prompt + completion, entry_point),
                    "raw_solution": prompt + completion,
                }
                for completion in task_outputs[:requested_samples]
            )
            continue

        samples.extend(
            {
                "task_id": task_id,
                "solution": sanitize(completion, entry_point),
                "raw_solution": completion,
            }
            for completion in task_outputs[:requested_samples]
        )
    return samples


def codegen(
    model: DecoderBase,
    target_path: str,
    split: str,
    subset: str,
    greedy: bool = False,
    strip_newlines: bool = False,
    n_samples: int = 1,
    id_range: tuple[int, int] | None = None,
    resume: bool = True,
    batch_size: int = -1,
) -> None:
    with Progress(
        TextColumn(
            f"BigCodeBench--{split.capitalize()} ({subset.capitalize()}) •"
            "[progress.percentage]{task.percentage:>3.0f}%"
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    ) as progress:
        dataset = get_bigcodebench(subset=subset)
        if model.is_direct_completion() and split == "instruct":
            raise Exception("Base model does not support direct completion for instruct tasks")

        _ensure_parent_dir(target_path)

        existing_sample_counts = (
            _load_existing_sample_counts(target_path) if resume else {}
        )
        batch_prompts: list[str] = []
        batch_task_ids: list[str] = []
        batch_sample_counts: list[int] = []
        batch_entry_points: list[str] = []
        dataset_items = list(dataset.items())

        for item_index, (task_id, task) in enumerate(progress.track(dataset_items)):
            if id_range is not None:
                low, high = id_range
                if item_index < low:
                    progress.console.print(
                        f"Skipping {task_id} as it is not in {id_range}"
                    )
                    continue
                if item_index >= high:
                    break

            pending_samples = n_samples - existing_sample_counts.get(task_id, 0)
            prompt_key = f"{split}_prompt"
            try:
                prompt = task[prompt_key]
            except KeyError as error:
                raise Exception(
                    f"Invalid split {split} for bigcodebench-{subset}"
                ) from error

            if strip_newlines:
                prompt = prompt.strip("\n")

            if pending_samples > 0:
                batch_prompts.append(prompt)
                batch_task_ids.append(task_id)
                batch_sample_counts.append(pending_samples)
                batch_entry_points.append(task["entry_point"])

                log_message = f"Codegen: {task_id.replace('/', '_')} @ {model}"
                existing_samples = existing_sample_counts.get(task_id, 0)
                if existing_samples > 0:
                    log_message += f" (resuming from {existing_samples})"
                progress.console.print(log_message)

            should_flush = _should_flush_batch(
                batch_size=batch_size,
                batch_prompts=batch_prompts,
                item_index=item_index,
                item_count=len(dataset_items),
                id_range=id_range,
            )
            if not should_flush:
                continue

            if not batch_prompts:
                break

            outputs = model.codegen(
                batch_prompts,
                do_sample=not greedy,
                num_samples=max(batch_sample_counts),
            )
            assert outputs, "No outputs from model!"

            samples = _build_samples(
                model=model,
                task_ids=batch_task_ids,
                prompts=batch_prompts,
                entry_points=batch_entry_points,
                sample_counts=batch_sample_counts,
                outputs=outputs,
            )
            print(f"Generated {len(samples)} samples")
            write_jsonl(target_path, samples, append=True)

            batch_prompts = []
            batch_task_ids = []
            batch_sample_counts = []
            batch_entry_points = []


def build_output_identifier(
    *,
    model: str,
    subset: str,
    split: str,
    backend: str,
    revision: str,
    temperature: float,
    n_samples: int,
    skip_prefill: bool,
) -> str:
    subset_suffix = f"-{subset}" if subset != "full" else ""
    identifier = model.replace("/", "--")
    if skip_prefill:
        identifier += "--skip_prefill"
    return (
        f"{identifier}--{revision}--bigcodebench{subset_suffix}-{split}"
        f"--{backend}-{temperature}-{n_samples}-sanitized_calibrated.jsonl"
    )


def run_codegen(
    model: str,
    split: str,
    subset: str,
    root: str = "bcb_results",
    lora_path: str = None,
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 1280,
    max_model_len: int = 12800,
    greedy: bool = False,
    reasoning_effort: str = "medium",
    reasoning_budget: int = 0,
    reasoning_beta: str = "output-128k-2025-02-19",
    strip_newlines: bool = False,
    direct_completion: bool = False,
    resume: bool = True,
    id_range: str = None,
    backend: str = "vllm",
    base_url: str = None,
    openai_parallel_workers: int = 1,
    tp: int = 1,
    instruction_prefix: str = "Please provide a self-contained Python script that solves the following problem in a markdown code block:",
    response_prefix: str = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:",
    skip_prefill: bool = False,
    revision: str = "main",
    trust_remote_code: bool = False,
    tokenizer_name: str = None,
    tokenizer_legacy: bool = False,
) -> str:
    if greedy or (temperature == 0 and n_samples == 1):
        temperature = 0
        n_samples = 1
        greedy = True
        print("Greedy decoding ON (--greedy): setting n_samples=1, temperature=0")

    requested_range: tuple[int, int] | None = None
    if id_range is not None:
        range_parts = [int(part) for part in id_range.split("-")]
        assert len(range_parts) == 2, "id_range must be a list of length 2"
        assert range_parts[0] < range_parts[1], "id_range must be increasing"
        requested_range = tuple(range_parts)

    if openai_parallel_workers < 1:
        raise ValueError("openai_parallel_workers must be at least 1")
    if backend != "openai" and openai_parallel_workers != 1:
        raise ValueError("openai_parallel_workers is only supported with backend='openai'")
    if openai_parallel_workers > 1 and n_samples != 1:
        raise ValueError("openai_parallel_workers currently supports n_samples=1")
    if backend == "openai" and openai_parallel_workers > 1 and bs is None:
        bs = openai_parallel_workers

    os.makedirs(root, exist_ok=True)

    model_runner = make_model(
        model=model,
        backend=backend,
        subset=subset,
        split=split,
        lora_path=lora_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_model_len=max_model_len,
        reasoning_effort=reasoning_effort,
        reasoning_budget=reasoning_budget,
        reasoning_beta=reasoning_beta,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        prefill=not skip_prefill,
        base_url=base_url,
        parallel_workers=openai_parallel_workers,
        tp=tp,
        revision=revision,
        trust_remote_code=trust_remote_code,
        direct_completion=direct_completion,
        tokenizer_name=tokenizer_name,
        tokenizer_legacy=tokenizer_legacy,
    )

    output_model = model
    if backend == "openai" and openai_parallel_workers > 1:
        output_model = f"{output_model}--parallel-{openai_parallel_workers}"
    if (
        backend == "openai"
        and reasoning_effort
        and any(
            model.startswith(prefix) or model.endswith(prefix)
            for prefix in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]
        )
    ):
        output_model = f"{output_model}--{reasoning_effort}"

    if lora_path:
        output_model = f"{output_model}--lora-{lora_path}"

    if backend == "anthropic" and reasoning_budget and reasoning_beta:
        output_model = f"{output_model}--{reasoning_budget}-{reasoning_beta}"

    identifier = build_output_identifier(
        model=output_model,
        subset=subset,
        split=split,
        backend=backend,
        revision=revision,
        temperature=temperature,
        n_samples=n_samples,
        skip_prefill=skip_prefill,
    )
    target_path = os.path.join(root, identifier)

    if not resume and os.path.exists(target_path):
        os.remove(target_path)

    codegen(
        model=model_runner,
        target_path=target_path,
        split=split,
        subset=subset,
        greedy=greedy,
        strip_newlines=strip_newlines,
        n_samples=n_samples,
        resume=resume,
        id_range=requested_range,
        batch_size=bs,
    )
    return target_path


def main() -> None:
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
