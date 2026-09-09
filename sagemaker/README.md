# SageMaker training

The SageMaker image launches FastWAM through the repository's Accelerate and
DeepSpeed configuration. It supports checkpoint syncing, spot restart, and
one or more training nodes.

## Prepare data

Upload the complete LeRobot v3 root once so each task and the shared text
embedding cache are available through one input channel:

```bash
aws s3 sync /datasets/OmniEmbodied_Data \
  s3://tri-ml-sandbox-16011-us-west-2-datasets/junjie/data/OmniEmbodied_Data \
  --exclude '.cache/*'
```

The expected S3 children are `bottle`, `dog`, `plate`, `pour`, and
`text_embeds_cache`.

## Build

Build and push the image after code or configuration changes:

```bash
python3 sagemaker/launch_sm.py --config g1 --build-only
```

## Submit

```bash
SKIP_BUILD=1 bash sagemaker/run_sm.sh g1 1 g1-bottle-fastwam \
  task=g1_uncond_1cam_320_1e-4 \
  'data.dataset_dirs=[/opt/ml/input/data/g1/bottle]'
```

Use `task=g1_joint_1cam_320_1e-4` for JointWAM. Replace `bottle` with `dog` or
`plate` for the other datasets. Put `WANDB_API_KEY` in the repository `.env`;
the wrapper forwards it to the job.

The G1 target uses one eight-GPU `ml.p5en.48xlarge` spot instance. With batch
size 4 and gradient accumulation 6, the effective global batch size is 192.

Use `DRY_RUN=1` to inspect a launch without building or submitting:

```bash
DRY_RUN=1 bash sagemaker/run_sm.sh g1 1 check \
  'data.dataset_dirs=[/opt/ml/input/data/g1/bottle]'
```
