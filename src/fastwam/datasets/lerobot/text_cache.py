import hashlib
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_TEXT_ENCODER_ID = "wan22ti2v5b"


def resolve_task_text_embedding_cache_dirs(
    task_names: Optional[Iterable[str]],
    cache_root: str | Path,
) -> dict[str, Path]:
    """Resolve the conventional ``<cache_root>/<task_name>`` directories."""
    if task_names is None:
        raise ValueError("`task_names` is required when a task cache root is configured.")
    root = Path(str(cache_root))
    resolved = {}
    for raw_task_name in task_names:
        task_name = str(raw_task_name).strip()
        task_path = Path(task_name)
        if (
            not task_name
            or task_path.is_absolute()
            or task_path.name != task_name
            or task_name in {".", ".."}
        ):
            raise ValueError(
                "Each selected task must be a non-empty directory name, "
                f"got {raw_task_name!r}."
            )
        if task_name in resolved:
            raise ValueError(f"Duplicate selected task: {task_name!r}.")
        resolved[task_name] = root / task_name
    return resolved


def text_embedding_cache_filename(
    prompt: str,
    context_len: int,
    encoder_id: str = DEFAULT_TEXT_ENCODER_ID,
) -> str:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{prompt_hash}.t5_len{context_len}.{encoder_id}.pt"
