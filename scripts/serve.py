#!/usr/bin/env python3
"""Serve a trained FastWAM policy over WebSocket."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a FastWAM checkpoint over WebSocket.")
    parser.add_argument("--checkpoint", required=True, help="Trained FastWAM weights file, e.g. step_000666.pt.")
    parser.add_argument("--dataset-stats", default=None, help="dataset_stats.json; defaults to the checkpoint's run directory.")
    parser.add_argument("--config", default=None, help="Training config.yaml; defaults to the checkpoint's run directory.")
    parser.add_argument("--task-config", default=None, help="Fallback Hydra task name when no saved config.yaml is available.")
    parser.add_argument("--embodiment", default="default", help="Embodiment identifier advertised to clients.")
    parser.add_argument("--image-key", action="append", default=None, help="Request image key in processor camera order; repeat for multiple cameras.")
    parser.add_argument("--state-key", default=None, help="Request state key; defaults to the processor's state key.")
    parser.add_argument("--action-key", default=None, help="Response action key; defaults to the processor's action key.")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of action diffusion inference steps.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, required=True, help="Action execution frequency advertised to clients.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--default-instruction", default="", help="Used when a client omits or sends an empty text field.")
    parser.add_argument("--compile-action-infer", action="store_true", help="Enable FastWAM's existing compiled action inference path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Keep --help available before importing model or optional serving dependencies.
    from fastwam.serving.fastwam_policy import FastWAMPolicy
    from fastwam.serving.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    policy = FastWAMPolicy.from_checkpoint(
        args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        config_path=args.config,
        task_config=args.task_config,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_steps,
        seed=args.seed,
        compile_action_infer=args.compile_action_infer,
        embodiment=args.embodiment,
        image_keys=args.image_key,
        state_key=args.state_key,
        action_key=args.action_key,
        default_instruction=args.default_instruction,
        fps=args.fps,
    )
    server = WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.server_metadata())
    server.serve_forever()


if __name__ == "__main__":
    main()
