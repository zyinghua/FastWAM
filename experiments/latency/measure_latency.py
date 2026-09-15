#!/usr/bin/env python
"""GPU-synchronized Fast-WAM and Joint-WAM inference latency benchmark.

This keeps the measurement boundary and paired direct/profile protocol used by
Faster-WAM's latency benchmark while loading one Fast-WAM checkpoint into both
architecturally compatible inference paths and using the RoboTwin input shapes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "evaluate_results" / "latency"
DEFAULT_TASK = "robotwin_uncond_3cam_384_1e-4"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints" / "fastwam_release" / "robotwin_uncond_3cam_384.pt"
)
MODEL_SPECS = (("fastwam", "Fast-WAM"), ("jointwam", "Joint-WAM"))


@dataclass(frozen=True)
class RunSpec:
    task: str
    image_height: int
    image_width: int
    context_len: int
    text_dim: int
    action_dim: int
    proprio_dim: int
    action_horizon: int
    num_video_frames: int
    num_inference_steps: int
    model_seed: int
    synthetic_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure checkpoint-loaded Fast-WAM and Joint-WAM latency on "
            "RoboTwin shapes."
        )
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be > 0")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0")
    if args.action_horizon is not None and args.action_horizon <= 0:
        parser.error("--action-horizon must be > 0")
    return args


def load_config(task: str) -> Any:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=[f"task={task}"])


def make_run_spec(cfg: Any, args: argparse.Namespace) -> RunSpec:
    height, width = (int(x) for x in cfg.data.train.video_size)
    action_horizon = int(args.action_horizon or (int(cfg.data.train.num_frames) - 1))
    ratio = int(cfg.data.train.action_video_freq_ratio)
    return RunSpec(
        task=str(args.task),
        image_height=height,
        image_width=width,
        context_len=int(cfg.data.train.context_len),
        text_dim=int(cfg.model.video_dit_config.text_dim),
        action_dim=int(cfg.data.train.processor.action_output_dim),
        proprio_dim=int(cfg.data.train.processor.proprio_output_dim),
        action_horizon=action_horizon,
        num_video_frames=action_horizon // ratio + 1,
        num_inference_steps=int(args.num_inference_steps),
        model_seed=int(cfg.seed),
        synthetic_seed=int(args.synthetic_seed),
    )


def build_model(
    cfg: Any,
    checkpoint: Path,
    device: torch.device,
    model_key: str,
) -> torch.nn.Module:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    if model_key == "fastwam":
        model_cfg._target_ = "fastwam.runtime.create_fastwam"
        model_cfg.compile_training_denoise = False
    elif model_key == "jointwam":
        model_cfg._target_ = "fastwam.runtime.create_fastwam_joint"
        if "compile_training_denoise" in model_cfg:
            del model_cfg["compile_training_denoise"]
    else:
        raise ValueError(f"Unsupported model key: {model_key}")
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=str(device))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    required = {"mot", "proprio_encoder"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required components: {sorted(missing)}"
        )
    if model.proprio_encoder is None:
        raise RuntimeError("Configured model has no proprio encoder")
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    return model.to(device).eval()


def prepare_inputs(spec: RunSpec, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(spec.synthetic_seed)
    image = (
        torch.rand(
            (1, 3, spec.image_height, spec.image_width),
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(2.0)
        .sub_(1.0)
        .to(device=device, dtype=torch.bfloat16)
    )
    return {
        "input_image": image,
        "context": torch.zeros(
            (1, spec.context_len, spec.text_dim),
            device=device,
            dtype=torch.bfloat16,
        ),
        "context_mask": torch.ones(
            (1, spec.context_len), device=device, dtype=torch.bool
        ),
        "proprio": torch.zeros((1, spec.proprio_dim), dtype=torch.float32),
    }


def infer_kwargs(
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
    model_key: str,
) -> dict[str, Any]:
    kwargs = {
        "prompt": None,
        "input_image": inputs["input_image"],
        "action_horizon": spec.action_horizon,
        "proprio": inputs["proprio"],
        "context": inputs["context"],
        "context_mask": inputs["context_mask"],
        "negative_prompt": "",
        "text_cfg_scale": 1.0,
        "num_inference_steps": spec.num_inference_steps,
        "sigma_shift": None,
        "seed": spec.model_seed,
        "rand_device": "cpu",
        "tiled": False,
    }
    if model_key == "jointwam":
        kwargs["num_video_frames"] = spec.num_video_frames
    return kwargs


def cuda_sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def run_direct(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
    model_key: str,
) -> tuple[dict[str, Any], float]:
    cuda_sync(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.infer_action(**infer_kwargs(spec, inputs, model_key))
    cuda_sync(model.device)
    return output, (time.perf_counter() - start) * 1000.0


def timed_stage(
    device: torch.device,
    callback: Callable[[], Any],
) -> tuple[Any, float]:
    start = time.perf_counter()
    result = callback()
    cuda_sync(device)
    return result, (time.perf_counter() - start) * 1000.0


@torch.no_grad()
def run_fastwam_profile(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    total_start = time.perf_counter()

    def init_action() -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        return torch.randn(
            (1, spec.action_horizon, model.action_expert.action_dim),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)

    latents_action, action_latent_init_ms = timed_stage(model.device, init_action)

    def encode_image() -> torch.Tensor:
        image = inputs["input_image"].to(device=model.device, dtype=model.torch_dtype)
        return model._encode_input_image_latents_tensor(input_image=image, tiled=False)

    first_frame_latents, vae_encode_ms = timed_stage(model.device, encode_image)

    def prepare_context() -> tuple[torch.Tensor, torch.Tensor]:
        context = inputs["context"].to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        context_mask = inputs["context_mask"].to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
        proprio = inputs["proprio"].to(device=model.device, dtype=model.torch_dtype)
        return model._append_proprio_to_context(context, context_mask, proprio)

    (context, context_mask), context_prepare_ms = timed_stage(
        model.device, prepare_context
    )

    def prepare_visual() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=model.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = model.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(tokens_per_frame),
            device=video_tokens.device,
        )
        video_cache_k, video_cache_v = model.mot.prefill_video_cache_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return video_cache_k, video_cache_v, attention_mask[video_seq_len:, :]

    (
        (video_cache_k, video_cache_v, action_attention_mask),
        visual_branch_ms,
    ) = timed_stage(model.device, prepare_visual)

    def denoise_action() -> torch.Tensor:
        steps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=spec.num_inference_steps,
            device=model.device,
            dtype=latents_action.dtype,
            shift_override=None,
        )
        current = latents_action
        for step_t, step_delta in zip(steps, deltas):
            timestep = step_t.unsqueeze(0).to(dtype=current.dtype, device=model.device)
            prediction = model._denoise_action_with_video_cache(
                latents_action=current,
                timestep_action=timestep,
                context=context,
                context_mask=context_mask,
                video_cache_k=video_cache_k,
                video_cache_v=video_cache_v,
                action_attention_mask=action_attention_mask,
            )
            current = model.infer_action_scheduler.step(prediction, step_delta, current)
        return current

    latents_action, action_denoise_loop_ms = timed_stage(model.device, denoise_action)
    action, action_to_cpu_ms = timed_stage(
        model.device,
        lambda: latents_action[0].detach().to(device="cpu", dtype=torch.float32),
    )
    profile_model_total_ms = (time.perf_counter() - total_start) * 1000.0
    return {"action": action}, {
        "profile_model_total_ms": profile_model_total_ms,
        "action_latent_init_ms": action_latent_init_ms,
        "vae_encode_ms": vae_encode_ms,
        "context_prepare_ms": context_prepare_ms,
        "visual_branch_ms": visual_branch_ms,
        "action_denoise_loop_ms": action_denoise_loop_ms,
        "action_to_cpu_ms": action_to_cpu_ms,
        "action_branch_ms": (
            action_latent_init_ms + action_denoise_loop_ms + action_to_cpu_ms
        ),
    }


@torch.no_grad()
def run_jointwam_profile(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    total_start = time.perf_counter()

    latent_t = (spec.num_video_frames - 1) // model.vae.temporal_downsample_factor + 1
    latent_h = spec.image_height // model.vae.upsampling_factor
    latent_w = spec.image_width // model.vae.upsampling_factor

    def init_latents() -> tuple[torch.Tensor, torch.Tensor]:
        video_generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        action_generator = torch.Generator(device="cpu").manual_seed(spec.model_seed)
        video = torch.randn(
            (1, model.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)
        action = torch.randn(
            (1, spec.action_horizon, model.action_expert.action_dim),
            generator=action_generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)
        return video, action

    (latents_video, latents_action), latent_init_ms = timed_stage(
        model.device, init_latents
    )

    def encode_image() -> torch.Tensor:
        image = inputs["input_image"].to(device=model.device, dtype=model.torch_dtype)
        return model._encode_input_image_latents_tensor(input_image=image, tiled=False)

    first_frame_latents, vae_encode_ms = timed_stage(model.device, encode_image)
    latents_video[:, :, 0:1] = first_frame_latents.clone()

    def prepare_context() -> tuple[torch.Tensor, torch.Tensor]:
        context = inputs["context"].to(
            device=model.device, dtype=model.torch_dtype, non_blocking=True
        )
        context_mask = inputs["context_mask"].to(
            device=model.device, dtype=torch.bool, non_blocking=True
        )
        proprio = inputs["proprio"].to(device=model.device, dtype=model.torch_dtype)
        return model._append_proprio_to_context(context, context_mask, proprio)

    (context, context_mask), context_prepare_ms = timed_stage(
        model.device, prepare_context
    )
    fuse_flag = bool(
        getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
    )

    def denoise_joint() -> tuple[torch.Tensor, torch.Tensor]:
        (
            video_steps,
            video_deltas,
        ) = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=spec.num_inference_steps,
            device=model.device,
            dtype=latents_video.dtype,
            shift_override=None,
        )
        (
            action_steps,
            action_deltas,
        ) = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=spec.num_inference_steps,
            device=model.device,
            dtype=latents_action.dtype,
            shift_override=None,
        )
        current_video = latents_video
        current_action = latents_action
        for step_video, delta_video, step_action, delta_action in zip(
            video_steps, video_deltas, action_steps, action_deltas
        ):
            timestep_video = step_video.unsqueeze(0).to(
                dtype=current_video.dtype, device=model.device
            )
            timestep_action = step_action.unsqueeze(0).to(
                dtype=current_action.dtype, device=model.device
            )
            pred_video, pred_action = model._predict_joint_noise(
                latents_video=current_video,
                latents_action=current_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=None,
            )
            current_video = model.infer_video_scheduler.step(
                pred_video, delta_video, current_video
            )
            current_action = model.infer_action_scheduler.step(
                pred_action, delta_action, current_action
            )
            current_video[:, :, 0:1] = first_frame_latents.clone()
        return current_video, current_action

    (latents_video, latents_action), joint_denoise_loop_ms = timed_stage(
        model.device, denoise_joint
    )
    action, action_to_cpu_ms = timed_stage(
        model.device,
        lambda: latents_action[0].detach().to(device="cpu", dtype=torch.float32),
    )
    profile_model_total_ms = (time.perf_counter() - total_start) * 1000.0
    return {"action": action}, {
        "profile_model_total_ms": profile_model_total_ms,
        "video_action_latent_init_ms": latent_init_ms,
        "vae_encode_ms": vae_encode_ms,
        "context_prepare_ms": context_prepare_ms,
        "joint_denoise_loop_ms": joint_denoise_loop_ms,
        "action_to_cpu_ms": action_to_cpu_ms,
        "joint_branch_ms": (latent_init_ms + joint_denoise_loop_ms + action_to_cpu_ms),
    }


def run_pair(
    model: torch.nn.Module,
    spec: RunSpec,
    inputs: dict[str, torch.Tensor],
    model_key: str,
    profile_first: bool,
) -> dict[str, Any]:
    profile_fn = run_fastwam_profile if model_key == "fastwam" else run_jointwam_profile
    if profile_first:
        profile_output, internal = profile_fn(model, spec, inputs)
        direct_output, direct_ms = run_direct(model, spec, inputs, model_key)
        order = "profile_then_direct"
    else:
        direct_output, direct_ms = run_direct(model, spec, inputs, model_key)
        profile_output, internal = profile_fn(model, spec, inputs)
        order = "direct_then_profile"
    expected_shape = (spec.action_horizon, spec.action_dim)
    for name, output in (("direct", direct_output), ("profile", profile_output)):
        action = output.get("action")
        if (
            not isinstance(action, torch.Tensor)
            or tuple(action.shape) != expected_shape
        ):
            raise RuntimeError(
                f"{name} output has shape {getattr(action, 'shape', None)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(action).all():
            raise RuntimeError(f"{name} output contains non-finite values")
    max_abs_diff = float(
        (direct_output["action"] - profile_output["action"]).abs().max().item()
    )
    if not torch.allclose(
        direct_output["action"], profile_output["action"], atol=1e-4, rtol=1e-4
    ):
        raise RuntimeError(
            f"Direct/profile outputs differ (max_abs_diff={max_abs_diff:.6g})"
        )
    return {
        "measurement_order": order,
        "model_infer_ms": float(direct_ms),
        "output_allclose": True,
        "output_max_abs_diff": max_abs_diff,
        "action_shape": list(direct_output["action"].shape),
        **internal,
    }


def metric(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=0)),
        "min_ms": float(array.min()),
        "median_ms": float(np.median(array)),
        "max_ms": float(array.max()),
    }


def summarize_records(
    model_key: str,
    model_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    direct_metric = metric([float(row["model_infer_ms"]) for row in records])
    profile_metric = metric([float(row["profile_model_total_ms"]) for row in records])
    mean_delta_pct = (
        abs(profile_metric["mean_ms"] - direct_metric["mean_ms"])
        / direct_metric["mean_ms"]
        * 100.0
    )
    if mean_delta_pct > 5.0:
        raise RuntimeError(
            f"Direct/profile means differ by {mean_delta_pct:.2f}% (limit 5.00%)"
        )
    summary: dict[str, Any] = {
        "model": model_name,
        "model_infer_ms": direct_metric,
        "profile_model_total_ms": profile_metric,
        "direct_profile_mean_delta_pct": mean_delta_pct,
        "vae_encode_ms": metric([float(row["vae_encode_ms"]) for row in records]),
    }
    if model_key == "fastwam":
        summary["visual_branch_ms"] = metric(
            [float(row["visual_branch_ms"]) for row in records]
        )
        summary["action_branch_ms"] = metric(
            [float(row["action_branch_ms"]) for row in records]
        )
        summary["core_branch_ms"] = metric(
            [
                float(row["visual_branch_ms"]) + float(row["action_branch_ms"])
                for row in records
            ]
        )
    else:
        summary["joint_branch_ms"] = metric(
            [float(row["joint_branch_ms"]) for row in records]
        )
        summary["core_branch_ms"] = summary["joint_branch_ms"]
    return summary


def write_results(
    output_dir: Path,
    settings: dict[str, Any],
    records_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summaries = [
        summarize_records(model_key, model_name, records_by_model[model_key])
        for model_key, model_name in MODEL_SPECS
    ]
    payload = {
        "settings": settings,
        "results": summaries,
        "records": records_by_model,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latency_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Inference Latency",
        "",
        f"- GPU: {settings['gpu']}",
        f"- Checkpoint: `{settings['checkpoint']}`",
        f"- Setting: BF16, batch 1, {settings['input_size'][0]} × "
        f"{settings['input_size'][1]}, action horizon {settings['action_horizon']}, "
        f"{settings['action_denoising_steps']} denoising steps",
        f"- Protocol: {settings['warmup']} warm-up pairs followed by "
        f"{settings['iterations']} GPU-synchronized measurement pairs",
        "- Joint-WAM uses the same Fast-WAM checkpoint as a latency-only proxy.",
        "",
        "| Model | VAE Encode | Core Branch | Model Infer |",
        "|---|---:|---:|---:|",
    ]
    for summary in summaries:
        report.append(
            f"| {summary['model']} | "
            f"{summary['vae_encode_ms']['mean_ms']:.3f} ± "
            f"{summary['vae_encode_ms']['std_ms']:.3f} | "
            f"{summary['core_branch_ms']['mean_ms']:.3f} ± "
            f"{summary['core_branch_ms']['std_ms']:.3f} | "
            f"{summary['model_infer_ms']['mean_ms']:.3f} ± "
            f"{summary['model_infer_ms']['std_ms']:.3f} |"
        )
    (output_dir / "latency_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summaries


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this latency benchmark")
    if args.gpu_id < 0 or args.gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id {args.gpu_id}; found {torch.cuda.device_count()} device(s)"
        )
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    cfg = load_config(args.task)
    spec = make_run_spec(cfg, args)
    checkpoint = args.checkpoint.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or (DEFAULT_RESULTS_ROOT / f"fastwam_joint_robotwin_{timestamp}")
    ).resolve()
    settings = {
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint": str(checkpoint),
        "checkpoint_loaded": True,
        "models": [model_name for _, model_name in MODEL_SPECS],
        "jointwam_weights": "Fast-WAM checkpoint; latency-only proxy",
        "dtype": "bfloat16",
        "batch_size": 1,
        "warmup": int(args.warmup),
        "iterations": int(args.iters),
        "input_size": [spec.image_height, spec.image_width],
        "action_horizon": spec.action_horizon,
        "video_frames": spec.num_video_frames,
        "action_denoising_steps": spec.num_inference_steps,
        "task_config": spec.task,
        "synthetic_inputs": True,
        "run_spec": asdict(spec),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", flush=True)
    inputs = prepare_inputs(spec, device)
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_key, model_name in MODEL_SPECS:
        print(f"Loading checkpoint-backed {model_name} on {device} ...", flush=True)
        setup_start = time.perf_counter()
        model = build_model(cfg, checkpoint, device, model_key)
        cuda_sync(device)
        print(
            f"Loaded {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B "
            f"parameters in {time.perf_counter() - setup_start:.1f}s",
            flush=True,
        )
        for index in range(args.warmup):
            run_pair(
                model,
                spec,
                inputs,
                model_key,
                profile_first=bool(index % 2),
            )
            print(
                f"{model_name} warm-up pair {index + 1}/{args.warmup}",
                flush=True,
            )
        records: list[dict[str, Any]] = []
        for index in range(args.iters):
            record = run_pair(
                model,
                spec,
                inputs,
                model_key,
                profile_first=bool(index % 2),
            )
            record.update(iteration=index, **asdict(spec))
            records.append(record)
            print(
                f"{model_name} measurement {index + 1}/{args.iters}: "
                f"native={record['model_infer_ms']:.3f} ms, "
                f"profile={record['profile_model_total_ms']:.3f} ms",
                flush=True,
            )
        records_by_model[model_key] = records
        del model
        torch.cuda.empty_cache()

    summaries = write_results(output_dir, settings, records_by_model)
    for summary in summaries:
        direct = summary["model_infer_ms"]
        print(
            f"{summary['model']} native inference: {direct['mean_ms']:.3f} ± "
            f"{direct['std_ms']:.3f} ms (n={args.iters})",
            flush=True,
        )
    print(f"Report: {output_dir / 'latency_report.md'}", flush=True)


if __name__ == "__main__":
    main()
