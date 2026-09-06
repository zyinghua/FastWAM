# G1 training and deployment with FastWAM

This recipe trains the original FastWAM or JointWAM model on the same G1
pick/place and pour LeRobot v3 data used by RollingWAM. It reads aggregated
Parquet and MP4 files directly; do not convert them to per-episode v2 files.
Commands below run from the `ref/FastWAM` repository root in an environment
with the repository's training dependencies installed (see the root README).

## Data and model configuration

The task is `g1_pnp_pour_uncond_1cam_320_1e-4` and its data configuration is
`configs/data/g1_pnp_pour_v30.yaml`.
Use `g1_pnp_pour_joint_1cam_320_1e-4` for JointWAM. Both tasks use the same G1
data and training settings; the JointWAM task selects `model=fastwam_joint`.

| Setting | Value |
| --- | --- |
| Dataset root | `/Omni-G1/data/self_collected/g1_pnp_pour_v3` |
| Camera column | `observation.images.egocentric` |
| Proprioception column | `states`, 43 values |
| Action column | `action`, 78 values: motion token 64, left hand 7, right hand 7 |
| Temporal sampling | 33 consecutive observations and 32 actions, stride 1, action/video ratio 1 |
| Image processing | Resize to 240×320, crop to 224×320 for Wan's spatial divisibility |
| Normalization | Per-dimension min/max; no delta-action subtraction |
| Text cache | `./artifacts/g1_pnp_pour_v3/text_embeds_cache`, context length 128 |
| Training | ZeRO-2, BF16; original FastWAM checkpointing and compilation settings |

The 32-action default follows FastWAM's existing `configs/data/libero_2cam.yaml`
and `configs/data/robotwin.yaml`: both use `num_frames: 33`, and
`RobotVideoDataset` creates `num_frames - 1` action targets. The G1 schema does
not require a 48-action window. Longer horizons are an experimental override,
not part of the G1 integration default. The server derives its prediction
horizon from the saved run config; checkpoints trained with 48 actions continue
to return 48 when loaded with their original config.

FastWAM's current video pipeline requires `(action_horizon / video_ratio)`
to be divisible by 4. Thus 32 and 48 are valid at ratio 1, while copying a
50-action horizon directly from a policy such as pi0.5 is not compatible with
this dataset pipeline. A different model's prediction horizon is not evidence
that the G1 data requires the same horizon here.

The G1 recipe retains RollingWAM's `action_video_freq_ratio: 1`, so all 33
observations are used as video. FastWAM's LIBERO/RoboTwin recipes use ratio 4,
which subsamples those observations to 9 video frames while retaining 32 action
targets. This is a dataset sampling setting; it changes video supervision and
compute. It does not introduce a rolling schedule or rolling inference state.

FastWAM jointly denoises the video/action horizon. This task retains its original
video shift `5.0` and action shift `1.0`; it does not add RollingWAM's rolling
schedule or A2A options. Gradient checkpointing and training-denoiser compilation
remain disabled, as in the base FastWAM config. For a separate comparison matching RollingWAM's action
shift, append both `model.action_scheduler.train_shift=5.0` and
`model.action_scheduler.infer_shift=5.0` and record that change in the run name.

The G1 v3 dataset must contain `meta/info.json`, `meta/tasks.parquet`, episode
metadata, data Parquet files, and camera videos at the paths declared by its
metadata. The text-cache tool also supports `meta/tasks.jsonl` for existing v2
datasets. Normalization statistics are
computed from the training data and written to the run directory. Generated
text caches stay outside the source dataset. Both task instructions are read
from metadata; `uncond` names the video model variant, not removal of text or
proprioception conditioning.

## Prepare local checkpoints

Set these once in the shell used for preprocessing and training:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export G1_DATA_ROOT=/Omni-G1/data/self_collected/g1_pnp_pour_v3
export G1_TEXT_CACHE="$(pwd)/artifacts/g1_pnp_pour_v3/text_embeds_cache"
```

Reuse the already prepared local Wan checkpoints from RollingWAM, or point
`DIFFSYNTH_MODEL_BASE_PATH` to the existing model directory. With the default
`model.redirect_common_files=true`, the required layout under that directory is:

```text
Wan-AI/
  Wan2.2-TI2V-5B/
    diffusion_pytorch_model*.safetensors
  Wan2.1-T2V-1.3B/
    google/umt5-xxl/                 # Complete tokenizer files
DiffSynth-Studio/
  Wan-Series-Converted-Safetensors/
    Wan2.2_VAE.safetensors
    models_t5_umt5-xxl-enc-bf16.safetensors
```

The wrappers disable automatic model downloads by default. Populate these
paths from local files before running them. The text encoder and tokenizer are
needed for text-cache generation; training uses the cached embeddings.

The model also needs the interpolated ActionDiT backbone at
`checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`. Reuse an
existing matching 1024-hidden-dimension backbone or generate it from local Wan
weights:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

This backbone excludes the action input/output projections, so its preprocessing
default of action dimension 7 does not fix the training action dimension. The
G1 task initializes its own 78-dimensional action projections and
43-dimensional proprioception encoder. To use a backbone stored elsewhere,
append `model.action_dit_pretrained_path=/absolute/path/backbone.pt` to training.
Changing `DIFFSYNTH_MODEL_BASE_PATH` alone does not change that separate path.

## Precompute task text embeddings

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/g1/precompute_g1_text_embeds.sh 1 ++overwrite=false
```

For a different dataset/cache, set `G1_DATA_ROOT` and `G1_TEXT_CACHE` before both
precomputation and training, or pass the same Hydra overrides to both commands:

```bash
bash scripts/g1/precompute_g1_text_embeds.sh 1 \
  'data.train.dataset_dirs=[/data/g1_pnp_pour_v3]' \
  data.train.text_embedding_cache_dir=/data/caches/g1_text
```

## Train on eight GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/g1/train_g1.sh 8
```

To train JointWAM instead:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/g1/train_g1.sh 8 task=g1_pnp_pour_joint_1cam_320_1e-4
```

The text cache is shared between these tasks. FastWAM conditions action
inference on the observed frame; JointWAM jointly denoises future video and
actions, with actions attending to the full video horizon. Each uses its
existing model implementation and noise schedules.

The task config currently uses a 3,000-step budget, per-GPU batch size 4,
gradient accumulation 4, learning rate `1e-4`, checkpoint interval 666 optimizer
steps, and online W&B. On eight GPUs the effective batch size is 128. Validation is disabled because
this task uses the entire dataset for training. Outputs go to
`runs/g1_pnp_pour_uncond_1cam_320_1e-4/<run_id>/` unless overridden.

Use fewer GPUs and a smaller batch when needed:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
bash scripts/g1/train_g1.sh 2 \
  batch_size=2 gradient_accumulation_steps=4 max_steps=20000 \
  wandb.enabled=false output_dir=./runs/g1_fastwam_2gpu
```

The training and text-cache wrappers accept an optional GPU count followed by Hydra overrides and
preserve `CUDA_VISIBLE_DEVICES`. If it is unset, they select device IDs from 0
through `num_gpus-1`. Match the count to the visible devices. Trailing overrides
take precedence over wrapper defaults. Run any wrapper with `--help` for short
usage examples; no training or model loading happens for help.

## Resume training or initialize from weights

Resume the complete optimizer/scheduler/model state with its **directory**:

```bash
bash scripts/g1/train_g1.sh 8 \
  resume=/absolute/path/run/checkpoints/state/step_000666 \
  data.train.pretrained_norm_stats=/absolute/path/run/dataset_stats.json \
  output_dir=/absolute/path/run
```

Keep the original task, data, normalization, batch/accumulation, process count,
and training budget when resuming. `max_steps` is the total target step count,
not additional steps; keep the original budget for a faithful continuation.

Load only trained G1 model weights into a new run with the **file** instead:

```bash
bash scripts/g1/train_g1.sh 8 \
  resume=/absolute/path/run/checkpoints/weights/step_000666.pt \
  data.train.pretrained_norm_stats=/absolute/path/run/dataset_stats.json \
  output_dir=./runs/g1_from_weights
```

A weights-only load starts a fresh optimizer, scheduler, and training step
counter. Use compatible G1 weights; a trained 7-dimensional LIBERO policy is
not a 78-dimensional G1 initializer. Keep `config.yaml` and
`dataset_stats.json` with the run for later interpretation and inference.

## Serve a trained G1 checkpoint

The server supports both trained tasks and selects FastWAM or JointWAM from
the checkpoint's saved `config.yaml`. JointWAM also gets its required video
frame count from that config, including the video subsampling ratio. Keep the
correct run config with the weights: the two methods have compatible weight
shapes but different attention and inference behavior. `--compile-action-infer`
is supported for FastWAM only.

The server and template client use the same binary WebSocket and NumPy
MessagePack protocol as RollingWAM. Install the serving dependencies in the
FastWAM environment:

```bash
pip install -e '.[serving]'
```

Start the server on one GPU with a trained **weights file**:

```bash
CUDA_VISIBLE_DEVICES=7 \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/FastWAM/checkpoints \
DIFFSYNTH_SKIP_DOWNLOAD=true \
bash scripts/g1/serve_g1_policy.sh \
  /workspace/FastWAM/runs/g1_pnp_pour_uncond_1cam_320_1e-4/RUN_ID/checkpoints/weights/step_000666.pt
```

The wrapper selects G1 request keys `ego_view`, `state`, and `action`, embodiment
`unitree_g1_sonic`, BF16, 10 denoising steps, a 10 Hz action rate, and port 8000.
It preserves `CUDA_VISIBLE_DEVICES`; the selected GPU appears as `cuda:0` to
the process. Model downloads are disabled by default. Unlike cached training,
serving also needs the local text encoder and tokenizer to encode instructions.
The wrapper's 10 Hz setting follows the RollingWAM launcher; it is not inferred
from dataset metadata. Set `--fps` to the action sampling rate used for training
when configuring the robot client.

Trailing arguments are passed to `scripts/serve.py` and override its defaults:

```bash
bash scripts/g1/serve_g1_policy.sh /models/g1.pt \
  --config /models/config.yaml \
  --dataset-stats /models/dataset_stats.json \
  --host 127.0.0.1 --port 8001 --num-steps 10
```

Without explicit paths, the server finds the resolved `config.yaml` and
`dataset_stats.json` in the checkpoint's run directory. A copied checkpoint
must retain these files or supply their paths as above. The config defines the
trained architecture, observation processing, and action horizon; the statistics
normalize the 43D state and unnormalize the 78D predictions. Reuse the training
statistics when deploying the weights.

Run `bash scripts/g1/serve_g1_policy.sh --help` for wrapper examples or
`python scripts/serve.py --help` for all server options. Help does not load a
model or start a server.

## Client template and robot integration

The template sends one synthetic observation and prints predictions without
actuating a robot:

```bash
bash scripts/g1/client_template.sh \
  ws://127.0.0.1:8000 \
  "pick up the container and pour its contents"
```

Replace the RGB and state arrays in `scripts/g1/client_template.sh` with live
inputs when adapting the robot-side client. On connection, the server first
sends metadata with its request keys, state/action dimensions, and action
horizon. The template reads that schema and validates the returned shape,
float32 dtype, and finite values. With the wrapper's defaults, the payloads are:

```python
request = {
    "images": {"ego_view": rgb_uint8_hwc},      # 480 x 640 x 3, RGB
    "states": {"state": state_float32_43},     # unnormalized robot state
    "text": "the task instruction",
    "embodiment_tag": "unitree_g1_sonic",
}

response = {
    "action": action_float32_32_by_78,         # unnormalized action values
}
```

`text` is required unless the server has a `--default-instruction`. The request
keys are protocol names; they need not equal the dataset's raw column names.
Use `fastwam.serving.msgpack_numpy.packb` and `unpackb` for array encoding, as
shown in the template. A text WebSocket reply reports an inference error.

The 78 action values retain the same SONIC layout as RollingWAM: 64 motion-token
values, 7 left-hand values, and 7 right-hand values. The template uses
`experiments/g1/action_layout.py` to validate the 43D state and split these
action segments without changing their values. The robot client remains
responsible for controller transport and actuation.

The current FastWAM task predicts **32 actions per request**, covering 3.2
seconds at 10 Hz. RollingWAM's current G1 task instead returns 8 actions from
its rolling window. Always read
`metadata["embodiments"][metadata["default_embodiment"]]["action_horizon"]`
instead of hardcoding either length.

FastWAM is stateless: every request makes a fresh prediction from its current
image, state, and instruction. A robot client can execute a chosen prefix of
the 32 actions, obtain new observations, and request a new prediction. It does
not need to consume the whole horizon to keep a rolling cache synchronized.
The metadata's `control.execute_horizon` and `control.replan_after_actions`
advertise the full horizon by default; explicitly select the executed prefix
in your robot-side control loop when using shorter replanning intervals. No
part of this template sends action commands. The server allows one connected
client at a time.

For the same 0.8-second replanning interval as the current RollingWAM G1 setup
at 10 Hz, execute the first 8 predicted rows, discard the remaining 24, and send
a fresh observation. The prediction horizon remains 32; the executed prefix is
a robot-client choice.
