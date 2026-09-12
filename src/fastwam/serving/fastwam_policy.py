"""FastWAM policy adapter for stateless observation-to-action inference."""

from __future__ import annotations

import inspect
import logging
import math
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from fastwam.datasets.lerobot.text_cache import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.config_resolvers import register_default_resolvers

if TYPE_CHECKING:
    from fastwam.evaluation.smoothness.recording import ActionTraceRecorder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"

register_default_resolvers()


def _model_dtype(mixed_precision: str) -> torch.dtype:
    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported mixed precision {mixed_precision!r}; expected no, fp16, or bf16."
    )


def _find_ancestor_file(checkpoint_path: Path, filename: str) -> Path | None:
    for parent in checkpoint_path.parents:
        candidate = parent / filename
        if candidate.is_file():
            return candidate
    return None


def _resolve_stats_path(checkpoint_path: Path, dataset_stats_path: str | None) -> Path:
    if dataset_stats_path:
        path = Path(dataset_stats_path).expanduser().resolve()
    else:
        path = _find_ancestor_file(checkpoint_path, "dataset_stats.json")
        if path is None:
            raise FileNotFoundError(
                "Could not derive dataset_stats.json from the checkpoint. "
                "Pass --dataset-stats explicitly."
            )
    if not path.is_file():
        raise FileNotFoundError(f"Dataset statistics not found: {path}")
    return path


def _compose_task_config(task_config: str) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(config_name="train", overrides=[f"task={task_config}"])


def _load_training_config(
    checkpoint_path: Path,
    config_path: str | None,
    task_config: str | None,
) -> tuple[DictConfig, Path | None]:
    if config_path:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Training config not found: {resolved}")
        return OmegaConf.load(resolved), resolved

    run_config = _find_ancestor_file(checkpoint_path, "config.yaml")
    if run_config is not None and run_config.is_file():
        return OmegaConf.load(run_config), run_config

    if not task_config:
        raise FileNotFoundError(
            "Could not derive config.yaml from the checkpoint. Pass --config or "
            "--task-config explicitly."
        )

    logger.warning(
        "No config.yaml found above %s; composing current task config %s. "
        "Pass --config to avoid configuration drift.",
        checkpoint_path,
        task_config,
    )
    return _compose_task_config(task_config), None


def _load_inference_checkpoint(model: Any, checkpoint_path: Path) -> int | None:
    """Load every model component required for action inference."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict, got {type(payload).__name__}.")

    required = {"mot", "proprio_encoder"}
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "The checkpoint is missing components required for action inference: "
            f"{sorted(missing)}."
        )
    if model.proprio_encoder is None:
        raise ValueError("The configured model has no proprio encoder.")

    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    step = payload.get("step")
    del payload
    return None if step is None else int(step)


def _flat_dimension(raw_shape: Any, *, name: str) -> int:
    if isinstance(raw_shape, int):
        dimension = int(raw_shape)
    else:
        try:
            dimension = math.prod(int(value) for value in raw_shape)
        except TypeError as exc:
            raise ValueError(f"{name} raw_shape must be an integer or a sequence.") from exc
    if dimension < 1:
        raise ValueError(f"{name} raw_shape must contain at least one value.")
    return dimension


class FastWAMPolicy:
    """Serve image/state observations using the preprocessing saved with a checkpoint."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        video_size: tuple[int, int],
        action_horizon: int,
        num_video_frames: int | None = None,
        num_inference_steps: int = 10,
        seed: int | None = None,
        compile_action_infer: bool = False,
        embodiment: str = "default",
        image_keys: Sequence[str] | None = None,
        state_key: str | None = None,
        action_key: str | None = None,
        concat_multi_camera: str | None = None,
        default_instruction: str = "",
        fps: float,
        execute_horizon: int | None = None,
        save_imagined_rollouts: bool = False,
        imagined_dir: str | None = None,
        action_trace_recorder: ActionTraceRecorder | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.video_size = tuple(int(v) for v in video_size)
        self.num_inference_steps = int(num_inference_steps)
        self.action_horizon = int(action_horizon)
        self.num_video_frames = None if num_video_frames is None else int(num_video_frames)
        self._infer_uses_video_frames = (
            "num_video_frames" in inspect.signature(self.model.infer_action).parameters
        )
        self.seed = seed
        self.compile_action_infer = bool(compile_action_infer)
        self.embodiment = str(embodiment)
        self.default_instruction = str(default_instruction)
        self.fps = float(fps)
        self.execute_horizon = (
            self.action_horizon if execute_horizon is None else int(execute_horizon)
        )
        self._lock = threading.RLock()
        self.save_imagined_rollouts = bool(save_imagined_rollouts)
        self._action_trace_recorder = action_trace_recorder
        self._trace_episode_started = False
        if action_trace_recorder is not None and action_trace_recorder.source != "predicted_command":
            raise ValueError("Server action traces require source='predicted_command'.")

        if len(self.video_size) != 2 or min(self.video_size) < 1:
            raise ValueError(f"video_size must be positive [H,W], got {self.video_size}.")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive.")
        if not 1 <= self.execute_horizon <= self.action_horizon:
            raise ValueError(
                "execute_horizon must be between 1 and action_horizon, got "
                f"{self.execute_horizon} for action_horizon={self.action_horizon}."
            )
        if self._infer_uses_video_frames or self.save_imagined_rollouts:
            if self.num_video_frames is None:
                raise ValueError("This model requires num_video_frames from its training config.")
            if self.num_video_frames <= 1 or self.num_video_frames % 4 != 1:
                raise ValueError("num_video_frames must be greater than 1 and satisfy T % 4 == 1.")
            if self.action_horizon % (self.num_video_frames - 1) != 0:
                raise ValueError("action_horizon must be divisible by num_video_frames - 1.")
        if self.compile_action_infer:
            from fastwam.models.wan22.fastwam_joint import FastWAMJoint

            if isinstance(self.model, FastWAMJoint):
                raise ValueError("JointWAM does not support compile_action_infer.")
        if self.processor.action_state_transforms is not None:
            raise ValueError(
                "This policy requires action_state_transforms=null, as in the G1 config."
            )
        if self.processor.delta_action_dim_mask is not None:
            raise ValueError(
                "This policy requires delta_action_dim_mask=null, as in the G1 config."
            )

        image_meta = self.processor.shape_meta["images"]
        state_meta = self.processor.shape_meta["state"]
        action_meta = self.processor.shape_meta["action"]
        if not image_meta:
            raise ValueError("Action inference requires at least one image field.")
        if len(state_meta) != 1 or len(action_meta) != 1:
            raise ValueError("Action inference requires exactly one state and one action field.")
        if int(self.processor.num_output_cameras) != len(image_meta):
            raise ValueError(
                f"Processor num_output_cameras={self.processor.num_output_cameras} does not match "
                f"the {len(image_meta)} configured image fields."
            )

        self._image_meta = list(image_meta)
        self._state_meta = state_meta[0]
        self._action_meta = action_meta[0]
        self.state_dim = _flat_dimension(self._state_meta["raw_shape"], name="state")
        self.action_dim = _flat_dimension(self._action_meta["raw_shape"], name="action")
        if int(self.model.action_expert.action_dim) != self.action_dim:
            raise ValueError(
                f"Model action dimension {self.model.action_expert.action_dim} does not match "
                f"processor dimension {self.action_dim}."
            )
        if self.model.proprio_dim is None or int(self.model.proprio_dim) != self.state_dim:
            raise ValueError(
                f"Model proprio dimension {self.model.proprio_dim} does not match processor "
                f"dimension {self.state_dim}."
            )
        if isinstance(image_keys, str):
            image_keys = [image_keys]
        if image_keys is None:
            image_keys = [str(meta["key"]) for meta in self._image_meta]
        self.image_keys = tuple(str(key) for key in image_keys)
        if len(self.image_keys) != len(self._image_meta):
            raise ValueError(
                f"Expected {len(self._image_meta)} request image keys, got {len(self.image_keys)}."
            )
        self.state_key = str(state_key or self._state_meta["key"])
        self.action_key = str(action_key or self._action_meta["key"])
        self.concat_multi_camera = (
            None if concat_multi_camera is None else str(concat_multi_camera).strip().lower()
        )
        for name, key in (
            ("state_key", self.state_key),
            ("action_key", self.action_key),
            ("embodiment", self.embodiment),
        ):
            if not key.strip():
                raise ValueError(f"{name} must not be empty.")
        if any(not key.strip() for key in self.image_keys):
            raise ValueError("Request image keys must not be empty.")
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError(f"Request image keys must be unique, got {self.image_keys}.")
        if len(self.image_keys) > 1 and self.concat_multi_camera not in {
            "horizontal",
            "vertical",
            "robotwin",
        }:
            raise ValueError(
                "Multiple image fields require concat_multi_camera to be horizontal, "
                "vertical, or robotwin."
            )
        self._final_resize = ResizeSmallestSideAspectPreserving(
            args={"img_h": self.video_size[0], "img_w": self.video_size[1]}
        )
        self._final_crop = CenterCrop(
            args={"img_h": self.video_size[0], "img_w": self.video_size[1]}
        )
        self._final_normalize = Normalize(args={"mean": 0.5, "std": 0.5})
        self.imagined_dir: Path | None = None
        self._imagined_writer = None
        self._imagined_path: Path | None = None
        self._imagined_chunks = 0
        self._imagined_session = -1
        self._active_instruction: str | None = None
        if self.save_imagined_rollouts:
            if imagined_dir is None or not str(imagined_dir).strip():
                raise ValueError("imagined_dir is required when saving imagined rollouts.")
            root = Path(imagined_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self.imagined_dir = Path(tempfile.mkdtemp(
                prefix=datetime.now().strftime("run_%Y%m%d_%H%M%S_"), dir=root,
            ))
            logger.info("Saving imagined rollouts to %s", self.imagined_dir)
        self.reset()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        dataset_stats_path: str | None = None,
        config_path: str | None = None,
        task_config: str | None = None,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        num_inference_steps: int = 10,
        seed: int | None = None,
        compile_action_infer: bool = False,
        embodiment: str = "default",
        image_keys: Sequence[str] | None = None,
        state_key: str | None = None,
        action_key: str | None = None,
        default_instruction: str = "",
        fps: float,
        execute_horizon: int | None = None,
        save_imagined_rollouts: bool = False,
        imagined_dir: str | None = None,
        action_trace_recorder: ActionTraceRecorder | None = None,
    ) -> "FastWAMPolicy":
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"FastWAM checkpoint not found: {checkpoint}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested, but CUDA is unavailable.")

        cfg, loaded_config = _load_training_config(checkpoint, config_path, task_config)
        stats_path = _resolve_stats_path(checkpoint, dataset_stats_path)
        action_horizon = int(cfg.data.train.num_frames) - 1
        video_ratio = int(cfg.data.train.get("action_video_freq_ratio", 1))
        if action_horizon < 1 or video_ratio < 1 or action_horizon % video_ratio != 0:
            raise ValueError("Training num_frames - 1 must be positive and divisible by action_video_freq_ratio.")
        num_video_frames = action_horizon // video_ratio + 1
        if num_video_frames % 4 != 1:
            raise ValueError("Training video frames must satisfy T % 4 == 1.")

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
        if "compile_training_denoise" in model_cfg:
            model_cfg.compile_training_denoise = False

        model = instantiate(
            model_cfg,
            model_dtype=_model_dtype(mixed_precision),
            device=device,
        )
        checkpoint_step = _load_inference_checkpoint(model, checkpoint)
        model = model.to(device).eval()

        processor_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.data.train.processor, resolve=True)
        )
        processor = instantiate(processor_cfg).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

        logger.info(
            "Loaded FastWAM policy | checkpoint=%s | step=%s | config=%s | stats=%s",
            checkpoint,
            checkpoint_step,
            loaded_config or f"task:{task_config}",
            stats_path,
        )
        return cls(
            model=model,
            processor=processor,
            video_size=tuple(cfg.data.train.video_size),
            # num_frames counts observations at action rate; the ratio subsamples
            # video only. RobotVideoDataset always targets num_frames - 1 actions.
            action_horizon=action_horizon,
            num_video_frames=num_video_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            compile_action_infer=compile_action_infer,
            embodiment=embodiment,
            image_keys=image_keys,
            state_key=state_key,
            action_key=action_key,
            concat_multi_camera=cfg.data.train.get("concat_multi_camera"),
            default_instruction=default_instruction,
            fps=fps,
            execute_horizon=execute_horizon,
            save_imagined_rollouts=save_imagined_rollouts,
            imagined_dir=imagined_dir,
            action_trace_recorder=action_trace_recorder,
        )

    def _preprocess_image(
        self,
        image: Any,
        *,
        request_key: str,
        image_meta: Mapping[str, Any],
    ) -> torch.Tensor:
        image_np = np.asarray(image)
        if image_np.dtype != np.uint8:
            raise TypeError(f"{request_key} must be uint8 RGB, got {image_np.dtype}.")
        if image_np.ndim != 3 or image_np.shape[-1] != 3:
            raise ValueError(
                f"{request_key} must be HWC RGB with shape [H,W,3], got {image_np.shape}."
            )

        # MessagePack arrays view an immutable bytes buffer; give PyTorch owned memory.
        frame = torch.from_numpy(image_np.copy(order="C")).permute(2, 0, 1).unsqueeze(0)
        transforms = self.processor.val_transforms
        if isinstance(transforms, Mapping):
            transforms = transforms[image_meta["key"]]
        if transforms is not None:
            for transform in transforms:
                frame = transform(frame)

        expected = (1, *tuple(int(v) for v in image_meta["shape"]))
        if tuple(frame.shape) != expected:
            raise ValueError(
                f"Configured transforms for {request_key} must produce {expected}, "
                f"got {tuple(frame.shape)}."
            )
        return frame

    def _compose_image_views(self, frames: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(frames) == 1:
            return frames[0]
        if self.concat_multi_camera == "horizontal":
            return torch.cat(list(frames), dim=-1)
        if self.concat_multi_camera == "vertical":
            return torch.cat(list(frames), dim=-2)
        if self.concat_multi_camera == "robotwin":
            if len(frames) != 3:
                raise ValueError(
                    "concat_multi_camera='robotwin' requires exactly three image fields."
                )
            top = transforms_F.resize(
                frames[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            lower_left = transforms_F.resize(
                frames[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            lower_right = transforms_F.resize(
                frames[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            lower = torch.cat([lower_left, lower_right], dim=-1)
            return torch.cat([top, lower], dim=-2)
        raise ValueError(f"Unsupported camera concatenation mode: {self.concat_multi_camera!r}.")

    def _preprocess_images(self, images: Any) -> torch.Tensor:
        if not isinstance(images, Mapping):
            raise TypeError(f"images must be a mapping, got {type(images).__name__}.")
        frames = []
        for request_key, image_meta in zip(self.image_keys, self._image_meta, strict=True):
            if request_key not in images:
                raise KeyError(f"Observation is missing images.{request_key}.")
            frames.append(
                self._preprocess_image(
                    images[request_key],
                    request_key=request_key,
                    image_meta=image_meta,
                )
            )

        frame = self._compose_image_views(frames)
        frame = self._final_resize(frame)
        frame = self._final_crop(frame)
        frame = self._final_normalize(frame)
        if tuple(frame.shape) != (1, 3, *self.video_size):
            raise ValueError(
                f"Final frame must be [1,3,{self.video_size[0]},{self.video_size[1]}], "
                f"got {tuple(frame.shape)}."
            )
        return frame.to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )

    def _normalize_state(self, state: Any) -> torch.Tensor:
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.ndim != 1 or state_np.shape[0] != self.state_dim:
            raise ValueError(
                f"{self.state_key} must have shape [{self.state_dim}], got {state_np.shape}."
            )
        if not np.isfinite(state_np).all():
            raise ValueError(f"{self.state_key} contains non-finite values.")

        key = self._state_meta["key"]
        batch = {
            "state": {
                key: torch.from_numpy(state_np.copy(order="C")).unsqueeze(0)
            }
        }
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim != 2 or tuple(action.shape) != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "Model action must have shape "
                f"[{self.action_horizon},{self.action_dim}], got {tuple(action.shape)}."
            )
        key = self._action_meta["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        action_np = normalizer.backward(
            action.unsqueeze(0).to(device="cpu", dtype=torch.float32)
        )[0].numpy()
        if not np.isfinite(action_np).all():
            raise ValueError("Model produced non-finite actions.")
        return np.asarray(action_np, dtype=np.float32)

    def _resolve_instruction(self, obs: Mapping[str, Any]) -> str:
        instruction = obs.get("text", self.default_instruction)
        if instruction is None or (isinstance(instruction, str) and not instruction.strip()):
            instruction = self.default_instruction
        if not isinstance(instruction, str):
            raise TypeError(f"text must be a string, got {type(instruction).__name__}.")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError(
                "A non-empty task instruction is required in request['text'] or "
                "via --default-instruction."
            )
        return instruction

    def _validate_embodiment(self, obs: Mapping[str, Any]) -> None:
        requested = obs.get("embodiment_tag")
        if requested is not None and str(requested).lower() != self.embodiment.lower():
            raise ValueError(
                f"This server only hosts {self.embodiment!r}, got embodiment_tag={requested!r}."
            )

    @torch.inference_mode()
    def infer(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Predict the full training horizon from the latest observation."""
        if not isinstance(obs, Mapping):
            raise TypeError(f"Observation must be a mapping, got {type(obs).__name__}.")
        with self._lock:
            self._validate_embodiment(obs)
            try:
                images = obs["images"]
                state = obs["states"][self.state_key]
            except (KeyError, TypeError) as exc:
                raise KeyError(
                    f"Observation must contain images and states.{self.state_key}."
                ) from exc

            instruction = self._resolve_instruction(obs)
            input_image = self._preprocess_images(images)
            proprio = self._normalize_state(state)
            if self._active_instruction is not None and instruction != self._active_instruction:
                self.reset()
            self._active_instruction = instruction
            prompt = DEFAULT_PROMPT.format(task=instruction)
            infer_kwargs = dict(
                input_image=input_image,
                action_horizon=self.action_horizon,
                prompt=prompt,
                proprio=proprio,
                seed=self.seed,
                num_inference_steps=self.num_inference_steps,
                compile_action_infer=self.compile_action_infer,
            )
            if self._infer_uses_video_frames:
                infer_kwargs["num_video_frames"] = self.num_video_frames
                if self.save_imagined_rollouts:
                    infer_kwargs["return_video_latents"] = True
            prediction = self.model.infer_action(**infer_kwargs)
            action_np = self._denormalize_action(prediction["action"])
            self._record_action_chunk(action_np, instruction)
            if self.save_imagined_rollouts:
                # Keep the native action result independent of additional model work.
                action_np = action_np.copy()
                self._record_imagined_rollout(prediction, infer_kwargs)
            return {self.action_key: action_np}

    def _record_action_chunk(self, action: np.ndarray, instruction: str) -> None:
        """Record the advertised prefix; the server cannot verify client execution."""
        recorder = self._action_trace_recorder
        if recorder is None:
            return
        if not self._trace_episode_started:
            recorder.start_episode(metadata={
                "task": instruction,
                "instruction": instruction,
                "fps": self.fps,
                "num_inference_steps": self.num_inference_steps,
                "actions_per_chunk": self.execute_horizon,
                "prediction_horizon": self.action_horizon,
                "execute_horizon": self.execute_horizon,
                "capture": "advertised_execution_prefix",
            })
            self._trace_episode_started = True
        recorder.start_chunk()
        # The client may discard the prediction tail before replanning. Including
        # it would measure a boundary the advertised command stream never reaches.
        for row in action[:self.execute_horizon]:
            recorder.record_action(row)

    def _record_imagined_rollout(
        self, prediction: Mapping[str, Any], infer_kwargs: Mapping[str, Any],
    ) -> None:
        """Append the executed prefix, resampled at control FPS, to this session's MP4."""
        try:
            device = torch.device(self.model.device)
            rng_devices = []
            if device.type == "cuda":
                rng_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            # Extra video sampling must not change later seed=None action predictions.
            with torch.random.fork_rng(devices=rng_devices):
                if self._infer_uses_video_frames:
                    # JointWAM already denoised these latents during action inference.
                    frames = self.model._decode_latents(prediction["video_latents"])
                else:
                    # FastWAM's native action-only path remains unchanged, including
                    # compilation. Generate video separately using its own attention mask.
                    video_kwargs = dict(infer_kwargs)
                    video_kwargs.update(
                        num_video_frames=self.num_video_frames,
                        test_action_with_infer_action=False,
                        compile_action_infer=False,
                    )
                    frames = self.model.infer_joint(**video_kwargs)["video"]
            if len(frames) != self.num_video_frames:
                raise ValueError(
                    f"Expected {self.num_video_frames} imagined frames, got {len(frames)}."
                )
            if self._imagined_writer is None:
                self._imagined_path = self.imagined_dir / f"session{self._imagined_session:04d}.mp4"
                self._imagined_writer = imageio.get_writer(
                    str(self._imagined_path), fps=self.fps, codec="libx264",
                    format="FFMPEG", pixelformat="yuv420p",
                )
                self._imagined_writer.append_data(np.asarray(frames[0], dtype=np.uint8))
                logger.info("Recording imagined rollout to %s", self._imagined_path)
            # Save only the portion the client is configured to execute before
            # replanning. Hold frames when video is sampled slower than actions.
            # Include the initial observation once, not at every chunk boundary.
            for action_step in range(1, self.execute_horizon + 1):
                frame_index = action_step * (self.num_video_frames - 1) // self.action_horizon
                self._imagined_writer.append_data(np.asarray(frames[frame_index], dtype=np.uint8))
            self._imagined_chunks += 1
        except Exception:
            logger.exception("Imagined rollout recording failed; disabling it for this server.")
            self.save_imagined_rollouts = False
            self._close_imagined_rollout()
            try:
                self.model.vae.model.clear_cache()
            except Exception:
                logger.debug("Could not clear VAE caches after recording failure.", exc_info=True)

    def _close_imagined_rollout(self) -> None:
        writer, path, chunks = self._imagined_writer, self._imagined_path, self._imagined_chunks
        self._imagined_writer = None
        self._imagined_path = None
        self._imagined_chunks = 0
        if writer is None:
            return
        try:
            writer.close()
            logger.info("Saved imagined rollout (%d chunks) to %s", chunks, path)
        except Exception:
            logger.exception("Could not finalize imagined rollout %s; disabling recording.", path)
            self.save_imagined_rollouts = False

    def reset(self) -> None:
        """Finalize recording; FastWAM keeps no observation/action stream state."""
        with self._lock:
            if self._action_trace_recorder is not None and self._trace_episode_started:
                self._action_trace_recorder.finish_episode(success=None)
                self._trace_episode_started = False
            self._close_imagined_rollout()
            self._imagined_session += 1
            self._active_instruction = None

    def server_metadata(self) -> dict[str, Any]:
        horizon = self.action_horizon
        return {
            "embodiments": {
                self.embodiment: {
                    "image_keys": list(self.image_keys),
                    "state_keys": {self.state_key: self.state_dim},
                    "action_keys": {self.action_key: self.action_dim},
                    "action_horizon": horizon,
                }
            },
            "default_embodiment": self.embodiment,
            "stateless": True,
            "image": {"dtype": "uint8", "layout": "HWC", "channels": "RGB"},
            "state": {"dtype": "float32"},
            "action": {
                "dtype": "float32",
                "shape": "(action_horizon, dim_per_key)",
                "space": "unnormalized",
            },
            "control": {
                "fps": self.fps,
                "execute_horizon": self.execute_horizon,
                "replan_after_actions": self.execute_horizon,
            },
        }
