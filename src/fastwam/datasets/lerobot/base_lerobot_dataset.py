import re
from pathlib import Path
from typing import List, Literal, Dict, Optional, Any, DefaultDict

import numpy as np
import torch
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from fastwam.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor
from .text_cache import (
    DEFAULT_PROMPT,
    DEFAULT_TEXT_ENCODER_ID,
    resolve_task_text_embedding_cache_dirs,
    text_embedding_cache_filename,
)

logger = get_logger(__name__)

MAX_GETITEM_ATTEMPT = 5


def _normalize_task_name(task_name: str) -> str:
    """Normalize task metadata so names such as ``lift pot`` and ``lift_pot`` match."""
    return re.sub(r"[^a-z0-9]+", "_", task_name.strip().lower()).strip("_")


def _raw_file_matches_task(raw_file_name: Any, normalized_task_name: str) -> bool:
    """Match canonical RoboTwin task names against path components conservatively."""
    if not isinstance(raw_file_name, str) or not raw_file_name.strip():
        return False

    path_components = [
        _normalize_task_name(component)
        for component in re.split(r"[/\\]+", raw_file_name)
        if component
    ]
    accepted_components = {
        normalized_task_name,
        f"{normalized_task_name}_demo_clean",
        f"{normalized_task_name}_demo_randomized",
    }
    return any(
        component in accepted_components
        or component.startswith(f"{normalized_task_name}_demo_clean_")
        or component.startswith(f"{normalized_task_name}_demo_randomized_")
        for component in path_components
    )


def _episode_instruction_tasks(
    meta: LeRobotDatasetMetadata,
    episode_index: int,
    episode: Dict[str, Any],
) -> set[str]:
    """Resolve the low-level instruction(s), excluding coarse/quality annotations."""
    episode_stats = getattr(meta, "episodes_stats", {}).get(episode_index, {})
    task_index_stats = episode_stats.get("task_index")
    if task_index_stats is not None:
        task_indices = set()
        for statistic in ("min", "max"):
            if statistic not in task_index_stats:
                continue
            values = np.asarray(task_index_stats[statistic]).reshape(-1)
            task_indices.update(int(value) for value in values)
        resolved_tasks = {
            meta.tasks[task_index]
            for task_index in task_indices
            if task_index in meta.tasks
        }
        if resolved_tasks:
            return resolved_tasks

    # Compatibility fallback for datasets whose episode stats omit task_index.
    return {task for task in episode.get("tasks", []) if isinstance(task, str)}


def _select_episode_indices(
    meta: LeRobotDatasetMetadata,
    task_names: Optional[List[str]],
    task_text_embedding_cache_root: Optional[str] = None,
    text_embedding_context_len: int = 128,
    expected_episodes_per_task: Optional[int] = None,
) -> List[int]:
    """Return episode indices whose metadata contains one of ``task_names``."""
    if task_names is None:
        return list(range(meta.total_episodes))
    if len(task_names) == 0:
        raise ValueError("`task_names` must be null (all tasks) or a non-empty list.")

    requested_by_normalized = {}
    for task_name in task_names:
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError(f"Every entry in `task_names` must be a non-empty string, got {task_name!r}.")
        normalized = _normalize_task_name(task_name)
        if normalized in requested_by_normalized:
            raise ValueError(
                "Duplicate task after normalization: "
                f"{requested_by_normalized[normalized]!r} and {task_name!r}."
            )
        requested_by_normalized[normalized] = task_name

    metadata_tasks_by_normalized = {}
    for metadata_task in meta.task_to_task_index:
        metadata_tasks_by_normalized.setdefault(_normalize_task_name(metadata_task), []).append(metadata_task)

    cache_filenames_by_normalized_task = None
    if task_text_embedding_cache_root is not None:
        configured_cache_dirs = {}
        resolved_cache_dirs = resolve_task_text_embedding_cache_dirs(
            task_names,
            task_text_embedding_cache_root,
        )
        for task_name, cache_dir in resolved_cache_dirs.items():
            normalized = _normalize_task_name(str(task_name))
            if normalized in configured_cache_dirs:
                raise ValueError(f"Duplicate selected task after normalization: {task_name!r}.")
            configured_cache_dirs[normalized] = cache_dir

        cache_filenames_by_normalized_task = {}
        filename_pattern = (
            f"*.t5_len{text_embedding_context_len}.{DEFAULT_TEXT_ENCODER_ID}.pt"
        )
        for normalized, cache_dir in configured_cache_dirs.items():
            if not cache_dir.is_dir():
                raise FileNotFoundError(
                    f"Text embedding cache directory for {requested_by_normalized[normalized]!r} "
                    f"does not exist: {cache_dir}"
                )
            cache_filenames = {path.name for path in cache_dir.rglob(filename_pattern)}
            if not cache_filenames:
                raise FileNotFoundError(
                    f"No {filename_pattern} files found for {requested_by_normalized[normalized]!r} "
                    f"under {cache_dir}."
                )
            cache_filenames_by_normalized_task[normalized] = cache_filenames
            logger.info(
                "Indexed %d precomputed text embeddings for task %s from %s.",
                len(cache_filenames),
                requested_by_normalized[normalized],
                cache_dir,
            )

    matched_metadata_tasks = {
        normalized: set(metadata_tasks_by_normalized.get(normalized, []))
        for normalized in requested_by_normalized
    }
    episode_counts = {original: 0 for original in requested_by_normalized.values()}
    episode_indices = []
    for episode_index, episode in meta.episodes.items():
        episode_tasks = set(episode.get("tasks", []))
        episode_instruction_tasks = _episode_instruction_tasks(meta, episode_index, episode)
        episode_cache_filenames = {
            text_embedding_cache_filename(
                DEFAULT_PROMPT.format(task=episode_task),
                context_len=text_embedding_context_len,
            )
            for episode_task in episode_instruction_tasks
        }
        include_episode = False
        for normalized, original in requested_by_normalized.items():
            matches_task_metadata = bool(matched_metadata_tasks[normalized].intersection(episode_tasks))
            matches_raw_file = _raw_file_matches_task(episode.get("raw_file_name"), normalized)
            if cache_filenames_by_normalized_task is not None:
                matches_selected_cache = bool(
                    cache_filenames_by_normalized_task[normalized].intersection(episode_cache_filenames)
                )
                raw_file_identifies_selected_task = any(
                    _raw_file_matches_task(episode.get("raw_file_name"), candidate)
                    for candidate in requested_by_normalized
                )
                matches_episode = matches_selected_cache and (
                    matches_raw_file or not raw_file_identifies_selected_task
                )
            else:
                matches_episode = matches_task_metadata or matches_raw_file
            if matches_episode:
                episode_counts[original] += 1
                include_episode = True
        if include_episode:
            episode_indices.append(episode_index)

    tasks_without_episodes = [task_name for task_name, count in episode_counts.items() if count == 0]
    if tasks_without_episodes:
        raw_file_examples = [
            episode.get("raw_file_name")
            for episode in meta.episodes.values()
            if episode.get("raw_file_name") is not None
        ][:10]
        available = sorted(metadata_tasks_by_normalized)
        raise ValueError(
            f"No episodes matched requested task(s) {tasks_without_episodes} in {meta.root}. "
            "Selected-task matching uses low-level `task_index` instructions and their precomputed cache "
            "files, with canonical `raw_file_name` components as an additional disambiguator. "
            "It does not guess from instruction keywords. "
            f"Available normalized metadata tasks include: {available[:50]}. "
            f"Example raw_file_name values: {raw_file_examples}"
        )
    if expected_episodes_per_task is not None:
        if expected_episodes_per_task <= 0:
            raise ValueError(
                f"`expected_episodes_per_task` must be positive or null, got {expected_episodes_per_task}."
            )
        unexpected_counts = {
            task_name: count
            for task_name, count in episode_counts.items()
            if count != expected_episodes_per_task
        }
        if unexpected_counts:
            raise ValueError(
                f"Selected-task episode counts do not equal expected_episodes_per_task="
                f"{expected_episodes_per_task}: {unexpected_counts}."
            )
        if len(episode_indices) != sum(episode_counts.values()):
            raise ValueError(
                "At least one episode matched multiple selected task names; refusing an ambiguous selection. "
                f"Unique episodes={len(episode_indices)}, per-task counts={episode_counts}."
            )

    logger.info(
        "Selected %d/%d episodes from %s for task_names=%s (per-task counts=%s)",
        len(episode_indices),
        meta.total_episodes,
        meta.root,
        list(task_names),
        episode_counts,
    )
    return sorted(episode_indices)


class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],

        # shapes
        shape_meta: Dict[str, Any],
        action_size: int = 1, 
        past_action_size: int = 0, # Excludes the current frame
        obs_size: int = 1, # should be 
        past_obs_size: int = 0,

        # train vs val
        val_set_proportion: float = 0.05, 
        is_training_set: bool = False,
        seed: int = 42,
        task_names: Optional[List[str]] = None,
        task_text_embedding_cache_root: Optional[str] = None,
        text_embedding_context_len: int = 128,
        expected_episodes_per_task: Optional[int] = None,

        # sampling
        global_sample_stride: int = 1,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"
        
        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None  # Will be set externally
        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            metas.append(meta)

        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        
        self.global_sample_stride = global_sample_stride

        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]

        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"
            delta_timestamps[meta["lerobot_key"]] = [
                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"
            delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]

        if not 0 <= val_set_proportion < 1:
            raise ValueError(f"`val_set_proportion` must be in [0, 1), got {val_set_proportion}.")

        episodes = {}
        for meta in metas:
            episode_indices = _select_episode_indices(
                meta,
                task_names,
                task_text_embedding_cache_root=task_text_embedding_cache_root,
                text_embedding_context_len=text_embedding_context_len,
                expected_episodes_per_task=expected_episodes_per_task,
            )
            if val_set_proportion >= 1e-6:
                split_idx = int(len(episode_indices) * (1 - val_set_proportion))
                # Shuffle after filtering so the train/val ratio applies to the selected tasks only.
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    episode_indices = episode_indices[:split_idx]
                else:
                    episode_indices = episode_indices[split_idx:]
            if not episode_indices:
                split_name = "training" if self.is_training_set else "validation"
                raise ValueError(
                    f"The {split_name} split for dataset {meta.root} contains no episodes after filtering."
                )
            episodes[meta.repo_id] = episode_indices
            logger.info(
                "Using %d episodes from %s for the %s split.",
                len(episode_indices),
                meta.root,
                "training" if self.is_training_set else "validation",
            )

        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )
        
        # HACK: lerobot 3.0 will fix this
        episode_data_index = []
        end_index = 0
        for dataset in self.multi_dataset._datasets:
            multi_episode_data_index = {
                "from": dataset.episode_data_index["from"] + end_index,
                "to": dataset.episode_data_index["to"] + end_index,
            }
            episode_data_index.append(multi_episode_data_index)
            end_index = multi_episode_data_index["to"][-1]

        self.episode_data_index = {
            "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
            "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
        }

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        action: torch.Tensor = lerobot_sample[lerobot_key] # [T, action_dim]
        if action.ndim == 1: # for shape of 1, like gripper
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1: # for shape of 1, like gripper
            state = state.unsqueeze(-1)
        # state = state[..., :-1, :]  # use state_{t} as observation_t
        assert state.shape[-1] == raw_shape, f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        return state
    
    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = meta["key"], meta["lerobot_key"], meta["raw_shape"]
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3: # time dim will lost when obs_size is 1
            image = image.unsqueeze(0)        
        image = (image * 255).to(torch.uint8) # (1, 3, H, W)
        # For config simplication
        # assert image.shape[1:] == raw_shape, f"Image '{key}' shape {image.shape[1:]} mismatch with {raw_shape}."
        return image
    
    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample
    
    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = {}, {}
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        return {"action": action, "state": state}

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return self.multi_dataset.num_frames

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        # Retry with random indices until we successfully load a frame.
        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). "
                    "Retrying with a random index. "
                    f"Error: {err}"
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts "
                f"for index {idx}."
            ) from last_exception

        # Get data from lerobot, organized in nested dict
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        # Preprocess the sample using the processor
        # for quick data loading
        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample

    def set_processor(self, processor: BaseProcessor):
        """Set processor instance from external initialization."""
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)

        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)

        episodes_num = self.multi_dataset.num_episodes
        
        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx) 
            batch = preprocessor.action_state_transform(batch)
            return batch
        
        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(range(episodes_num), desc="Iterating dataset to get normalization"):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))
        
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
                
                for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating dataset to get normalization"):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key] # (B, T, dim)
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                            state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))

                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key] # (B, T, dim)
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                            action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        # assume that each minibatch has equal number of samples
        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {"state": DefaultDict(dict), "action": DefaultDict(dict), "num_episodes": episodes_num, "num_transition": self.multi_dataset.num_frames}
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"], 
                stats["action"][key]["stepwise_std"], 
                stats["action"][key]["global_mean"], 
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Construct a sliding-window tensor from the input tensor x (shape: [N, D]).
    The output shape is [N, window_size, D].
    
    For each starting index i:
        out[i, j, :] =
            x[i + j, :]      if i + j < N
            x[-1, :]         otherwise (replicate the last row when out of bounds)
    
    Args:
        x (torch.Tensor): Input tensor of shape [N, D]
        window_size (int): Size of the sliding window
    
    Returns:
        torch.Tensor: Tensor of shape [N, window_size, D]
    """
    assert x.dim() == 2
    assert window_size > 0
    
    N, D = x.shape
    
    # shape [N, window_size]
    # indices[i, j] = i + j
    i_indices = torch.arange(N).unsqueeze(1)            # [N, 1]
    j_indices = torch.arange(window_size).unsqueeze(0)  # [1, window_size]
    indices = i_indices + j_indices                     # [N, window_size]

    # N-1
    # torch.clamp  [0, N-1]
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)

    # clamped_indices [N, window_size]，x [N, D]
    # out[i, j, :] = x[clamped_indices[i, j], :]
    out = x[clamped_indices]  # [N, window_size, D]

    return out
