"""FastWAM SageMaker launcher (host-side control plane).

Builds/pushes the training image, turns a target YAML in ``sagemaker/configs/``
into SageMaker input channels + Hydra overrides, and submits a
``sagemaker.pytorch.PyTorch`` estimator (via an AWS Batch fair-share queue, or
``estimator.fit``).

    python sagemaker/launch_sm.py --config g1 --instance-count 1 \
        -- task=g1_uncond_1cam_320_1e-4 \
        'data.dataset_dirs=[/opt/ml/input/data/g1/bottle]'

Everything after ``--`` is forwarded verbatim to ``scripts/train.py`` as Hydra
overrides, and wins over the target YAML's own ``overrides``.

Build and submit can happen on different machines (image build needs docker +
ECR; submission needs only boto3 + the SageMaker SDK)::

    python sagemaker/launch_sm.py --config g1 --build-only   # docker host
    python sagemaker/launch_sm.py --config g1 --skip-build -- task=...

No FastWAM code is imported here: the launcher runs on a host with no torch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

# Repo root: sagemaker/launch_sm.py -> parents[1].
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# ECR-repo / S3-path component identifying this project.
NAME = "fastwam"
DEFAULT_REGION = "us-west-2"

#: In-container entry. Code is baked into the image at /opt/ml/code, so this is
#: a repo-relative path and no source_dir is uploaded.
ENTRY_POINT = "sagemaker/entry.py"

#: Env namespace for the launcher -> container channel contract. MUST match
#: ``sm_env.ENV_PREFIX``; the launcher deliberately does not import sm_env
#: (control plane vs. runtime), so keep the two constants in sync by hand.
ENV_PREFIX = "FASTWAM_SM"

INSTANCE_MAPPER = {
    "p4d":    "ml.p4d.24xlarge",
    "p4de":   "ml.p4de.24xlarge",
    "p5":     "ml.p5.48xlarge",       # 8x H100 80GB
    "p5en":   "ml.p5en.48xlarge",     # 8x H200 144GB
    "p6":     "ml.p6-b200.48xlarge",  # 8x B200 192GB
    "g6e":    "ml.g6e.48xlarge",
    "g5":     "ml.g5.48xlarge",
}

# Fair-share Batch queues: aliases -> (fss_queue_name, instance_type_key).
QUEUE_MAP = {
    ("ml", "ml-p5"):                            ("fss-ml-p5-48xlarge-us-west-2",               "p5"),
    ("testing", "testing-p5"):                  ("fss-testing-p5-48xlarge-us-west-2",          "p5"),
    ("testing-p6",):                            ("fss-testing-p6-b200-48xlarge-us-west-2",     "p6"),
    ("vla", "vla-p5"):                          ("fss-vla-p5-48xlarge-us-west-2",              "p5"),
    ("vla-p5en",):                              ("fss-vla-p5en-48xlarge-us-west-2",            "p5en"),
    ("cv", "cv-wfm", "cv-p5en", "cv-wfm-p5en"): ("fss-cv-wfm-p5en-48xlarge-us-west-2",         "p5en"),
    # ad2 account. Its service environment ad2-wfm-us-west-2 reserves
    # 16 x ml.p5en.48xlarge and sits completely idle (0 running, 0 queued), but
    # RoboticsWFM-BatchOperator is NOT authorized on it: submitting fails with
    # "no identity-based policy allows the batch:SubmitServiceJob action" — a
    # missing grant, unlike the tag-conditioned SCP deny on fss-cv-wfm-p5en, so
    # no amount of tagging helps. The alias is kept so it works the moment the
    # any4d owners grant access; ask them before assuming their reservation.
    ("any4d",):                                 ("any4d-training-queue",                       "p5en"),
    # Spot FSS queues (robotics-old / 124224456861 only). Submit with --spot.
    ("cv-spot", "cv-wfm-spot-p5en"):            ("fss-cv-wfm-spot-p5en-48xlarge-us-west-2",    "p5en"),
    ("cv-spot-p5", "cv-wfm-spot-p5"):           ("fss-cv-wfm-spot-p5-48xlarge-us-west-2",      "p5"),
    ("vla-spot", "vla-spot-p5"):                ("fss-vla-spot-p5-48xlarge-us-west-2",         "p5"),
    ("cv-spot-p6", "cv-wfm-spot-p6"):           ("fss-cv-wfm-spot-p6-b200-48xlarge-us-west-2", "p6"),
    ("cam", "cam-p5"):                          ("fss-tri-cam-humanoid-p5-48xlarge-us-west-2", "p5"),
}

ACCOUNT_CONFIGS = {
    # TRI robotics account. FSS queues, the SageMaker execution role, the ECR
    # repo and the tri-ml-sandbox bucket all live here.
    #
    # Bucket choice matters: the ml-dgx-botuser identity exported on the DGX
    # hosts is READ-ONLY on robotics-manip-lbm (its bucket policy grants no
    # PutObject) but read-write on tri-ml-sandbox-16011-us-west-2-datasets, so
    # data upload and checkpoint sync both target the sandbox bucket.
    "robotics-old": dict(
        account_id="124224456861",
        profile="Robotics-LBM-PowerUserAccess-124224456861",
        # The reserved-capacity FSS pool authorizes this service role only.
        arn="arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess",
        s3_bucket="tri-ml-sandbox-16011-us-west-2-datasets",
        s3_checkpoint_base="s3://tri-ml-sandbox-16011-us-west-2-datasets/sagemaker/junjie/fastwam",
        use_queue=True,
        max_run=25 * 24 * 60 * 60 - 360,  # 25 days
        volume_size=3000,
    ),
    "robotics-new": dict(
        account_id="385697366450",
        profile="rob-sm",
        arn="arn:aws:iam::385697366450:role/Robotics-WFM-Sagemaker-role-us-west-2",
        s3_bucket="tri-ml-sandbox-16011-us-west-2-datasets",
        s3_checkpoint_base="s3://tri-ml-sandbox-16011-us-west-2-datasets/sagemaker/junjie/fastwam",
        use_queue=True,
        max_run=5 * 24 * 60 * 60 - 360,
        volume_size=1000,
    ),
}

# Only needed for the FSS queue submission path.
try:
    from sagemaker.aws_batch.training_queue import TrainingQueue as Queue  # noqa: F401
    _HAS_QUEUE = True
except ImportError:
    _HAS_QUEUE = False


def tag_map() -> dict[str, str]:
    return {
        "tri.project":     os.environ.get("TRI_PROJECT", "MM:PJ-0077"),
        "tri.owner.email": os.environ.get("TRI_OWNER_EMAIL", "CHANGE.ME@tri.global"),
    }


def tags() -> list[dict]:
    return [{"Key": k, "Value": v} for k, v in tag_map().items()]


def run_command(command: str) -> None:
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def resolve_queue(queue: str) -> tuple[str, str]:
    for aliases, (queue_name, instance_type) in QUEUE_MAP.items():
        if queue in aliases:
            return queue_name, instance_type
    raise ValueError(
        f"Invalid queue name: {queue!r}. Valid aliases: "
        f"{[a for aliases in QUEUE_MAP for a in aliases]}"
    )


def load_target(name: str) -> dict:
    """Load a target YAML from sagemaker/configs (with or without .yaml)."""
    path = Path(name)
    if not path.is_file():
        path = _CONFIG_DIR / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if not path.is_file():
        available = sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))
        raise SystemExit(f"Target config not found: {name!r}. Available: {available}")
    cfg = yaml.safe_load(path.read_text()) or {}
    for key in ("channels", "overrides"):
        if key not in cfg:
            raise SystemExit(f"{path} is missing the required `{key}` block.")
    return cfg


def _sanitize(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", part).strip("-").lower()


def make_job_name(exp: str, user: str, timestamp: str) -> str:
    """SageMaker-legal training-job name: ``{user}-{exp}-{ts}`` (<=63 chars)."""
    user_s, exp_s, ts_s = _sanitize(user), _sanitize(exp), _sanitize(timestamp)
    prefix = f"{user_s}-" if user_s else ""
    suffix = f"-{ts_s}" if ts_s else ""
    budget = max(1, 63 - len(prefix) - len(suffix))
    exp_s = exp_s[:budget].strip("-") or "exp"
    return f"{prefix}{exp_s}{suffix}"[:63].strip("-")


def build_and_push_image(
    *, account: dict, image_tag: str, dockerfile: str, region: str,
    profile: str | None, dlc_profile: str | None = None, skip_build: bool = False,
) -> str:
    """Build the image, push to ECR, return the image URI.

    The AWS DLC base lives in a cross-account registry (763104351884); some
    restricted job roles cannot pull it, so ``dlc_profile`` (env
    ``SM_DLC_PROFILE``) can override the profile used for that one login.
    """
    registry = f"{account['account_id']}.dkr.ecr.{region}.amazonaws.com"
    uri = f"{registry}/{NAME}:{image_tag}"
    if skip_build:
        return uri

    prof = f"--profile {profile}" if profile else ""
    login = (f"aws ecr get-login-password --region {region} {prof} "
             f"| docker login --username AWS --password-stdin")
    dlc_prof = f"--profile {dlc_profile}" if dlc_profile else prof
    dlc_login = (f"aws ecr get-login-password --region {region} {dlc_prof} "
                 f"| docker login --username AWS --password-stdin")
    commands = [
        f"{dlc_login} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build -f {dockerfile} --build-arg AWS_REGION={region} -t {NAME}:{image_tag} .",
        f"docker tag {NAME}:{image_tag} {uri}",
        f"{login} {registry}",
        (f"aws --region {region} {prof} ecr describe-repositories --repository-names {NAME} "
         f"|| aws --region {region} {prof} ecr create-repository --repository-name {NAME}"),
    ]
    run_command("\n".join(f"{cmd} || exit 1" for cmd in commands))
    run_command(f"docker push {uri}")
    return uri


def build_training_inputs(channels: dict):
    """Turn the target's ``channels`` block into SageMaker ``TrainingInput``s."""
    from sagemaker.inputs import TrainingInput

    fit_inputs = {}
    for name, spec in channels.items():
        s3_uri = str(spec["s3"]).rstrip("/")
        if not s3_uri.startswith("s3://"):
            raise SystemExit(f"channel {name!r}: `s3` must be an s3:// URI, got {s3_uri!r}")
        fit_inputs[name] = TrainingInput(
            s3_data=s3_uri,
            input_mode=spec.get("mode", "FastFile"),
            s3_data_type="S3Prefix",
            distribution=spec.get("distribution", "FullyReplicated"),
        )
    return fit_inputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        required=True,
        help="Target YAML name or path; bare names resolve under sagemaker/configs.",
    )
    p.add_argument("--account", default="robotics-old", choices=list(ACCOUNT_CONFIGS))
    # Build / image
    p.add_argument("--image-tag", default="latest", help="ECR image tag.")
    p.add_argument("--dockerfile", default="sagemaker/Dockerfile")
    p.add_argument("--skip-build", action="store_true",
                   help="Reuse the pushed ECR image (no docker build/push).")
    p.add_argument("--build-only", action="store_true",
                   help="Build + push the image and exit (for a docker host that cannot submit).")
    # Instance / queue
    p.add_argument("--instance-count", type=int, default=None,
                   help="Number of nodes; defaults to the target YAML.")
    p.add_argument("--instance-type", default=None,
                   help="Instance alias (e.g. p5en); overrides the queue's type.")
    p.add_argument("--queue", default=None, help="FSS queue alias; defaults to the target YAML.")
    p.add_argument("--priority", type=int, default=400)
    p.add_argument("--spot", action="store_true", help="Use spot instances (spot queues need this).")
    # AWS
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--profile", default=None, help="AWS profile (default: the account's).")
    # Naming + env
    p.add_argument("--name", default=None, help="Job name suffix (else a timestamp).")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                   help="Extra env var injected into the job. Repeatable.")
    p.add_argument("--dry-run", action="store_true",
                   help="Assemble everything and print a summary without submitting. "
                        "Implies --skip-build (a dry run should never kick off a docker build).")
    p.add_argument("script_args", nargs=argparse.REMAINDER,
                   help="`-- task=<name> [hydra overrides ...]` for scripts/train.py.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = load_target(args.config)

    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    # The target YAML carries a default task; the CLI may override it.
    if not any(tok.split("=", 1)[0].lstrip("+~") == "task" for tok in script_args):
        if not target.get("task"):
            raise SystemExit("No task: pass `-- task=<name>` or set `task:` in the target YAML.")
        script_args = [f"task={target['task']}", *script_args]
    task = next(t.split("=", 1)[1] for t in script_args if t.split("=", 1)[0].lstrip("+~") == "task")

    account = ACCOUNT_CONFIGS[args.account]
    profile = args.profile or account["profile"]
    user = os.environ.get("SM_USER") or os.environ.get("USER") or "user"

    image_uri = build_and_push_image(
        account=account, image_tag=args.image_tag, dockerfile=args.dockerfile,
        region=args.region, profile=profile,
        dlc_profile=os.environ.get("SM_DLC_PROFILE"),
        skip_build=(args.skip_build or args.dry_run) and not args.build_only,
    )
    if args.build_only:
        print(f"Built and pushed: {image_uri}")
        print("Submit from a host with AWS API access:")
        print(f"  python sagemaker/launch_sm.py --config {args.config} --skip-build "
              f"--image-tag {args.image_tag} -- {' '.join(script_args)}")
        return

    instance_count = args.instance_count or int(target.get("instance_count", 1))
    queue_name: str | None = None
    if args.instance_type is not None:
        instance_alias = args.instance_type
    else:
        queue_name, instance_alias = resolve_queue(args.queue or target.get("queue", "ml"))
        # Spot FSS queues reject non-spot jobs, so --spot is implied by the queue.
        if "spot" in queue_name and not args.spot:
            print(f"note: {queue_name} is a spot queue; enabling --spot")
            args.spot = True
    instance_type = INSTANCE_MAPPER[instance_alias]

    now = datetime.now()
    timestamp = f"{now.strftime('%Y%m%d-%H%M%S')}-{now.microsecond // 1000:03d}"
    job_name = make_job_name(task, user, args.name or timestamp)

    output_base = account["s3_checkpoint_base"].rstrip("/")
    checkpoint_s3 = f"{output_base}/{job_name}"

    channels = target["channels"]
    environment: dict[str, str] = {
        "SAGEMAKER_PROGRAM": ENTRY_POINT,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        f"{ENV_PREFIX}_INPUT_CHANNEL_COUNT": str(len(channels)),
        f"{ENV_PREFIX}_CHECKPOINT_S3": checkpoint_s3,
    }
    if not args.spot:
        # FSS reserved-capacity Batch queues reject the job without this — and
        # spot queues reject the job WITH it ("SM_USE_RESERVED_CAPACITY should
        # not be set for spot queues").
        environment["SM_USE_RESERVED_CAPACITY"] = "1"
    for i, name in enumerate(channels):
        environment[f"{ENV_PREFIX}_INPUT_CHANNEL_{i:03d}"] = name
    if any(str(s.get("mode", "FastFile")) == "FastFile" for s in channels.values()):
        environment["SM_FFM_WITH_MOUNTPOINT_ENABLED"] = "true"
    if target.get("pretrained_s3"):
        # Weights fetched with `aws s3 cp` in-container instead of a channel.
        environment[f"{ENV_PREFIX}_PRETRAINED_S3"] = str(target["pretrained_s3"])
    for kv in args.env:
        if "=" not in kv:
            raise SystemExit(f"--env must be KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        environment[k] = v
    for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN"):
        val = os.environ.get(key)
        if val:
            environment.setdefault(key, val)

    hyperparameters = {
        "train_argv": shlex.join(script_args),
        "sm_overrides": shlex.join([str(o) for o in target["overrides"]]),
        "sm_env": json.dumps(target.get("env", {})),
    }

    print("=== FastWAM SageMaker launch ===")
    print(f"  target:       {args.config}  task={task}")
    print(f"  account:      {args.account} ({account['account_id']})  profile={profile}")
    print(f"  image:        {image_uri}")
    print(f"  instances:    {instance_count} x {instance_type}"
          + (f"  queue={queue_name} (priority={args.priority})" if account["use_queue"] else ""))
    print(f"  job_name:     {job_name}")
    print(f"  checkpoints:  {checkpoint_s3}   (<- /opt/ml/checkpoints)")
    for name, spec in channels.items():
        print(f"  channel {name:<12} {spec['s3']}  [{spec.get('mode', 'FastFile')}]")
    print(f"  train_argv:   {hyperparameters['train_argv']}")
    print(f"  sm_overrides: {hyperparameters['sm_overrides']}")

    if args.dry_run:
        print("  (dry-run: NOT submitting)")
        return

    import boto3
    from sagemaker import Session
    from sagemaker.pytorch import PyTorch

    session = Session(
        boto_session=boto3.session.Session(region_name=args.region, profile_name=profile),
        default_bucket=account["s3_bucket"],
    )
    fit_inputs = build_training_inputs(channels)
    max_run = account["max_run"]

    estimator = PyTorch(
        entry_point=ENTRY_POINT,
        source_dir=None,  # code is baked into the image at /opt/ml/code
        image_uri=image_uri,
        role=account["arn"],
        instance_count=instance_count,
        instance_type=instance_type,
        # NOTE: no `distribution=` on purpose. SageMaker's torch_distributed
        # would torchrun the entry per GPU; FastWAM's trainer requires the
        # Accelerate DeepSpeed plugin, so entry.py runs once per host and calls
        # `accelerate launch --config_file ...` itself.
        hyperparameters=hyperparameters,
        output_path=output_base,
        checkpoint_s3_uri=checkpoint_s3,
        checkpoint_local_path="/opt/ml/checkpoints",
        code_location=output_base,
        environment=environment,
        tags=tags(),
        volume_size=account["volume_size"],
        max_run=max_run,
        max_wait=max_run if args.spot else None,
        train_use_spot_instances=args.spot,
        keep_alive_period_in_seconds=None if args.spot else 300,
        enable_sagemaker_metrics=True,
        disable_profiler=True,
        debugger_hook_config=False,
        base_job_name=job_name,
        sagemaker_session=session,
    )

    if account["use_queue"] and _HAS_QUEUE:
        from sagemaker.aws_batch.training_queue import TrainingQueue as Queue

        os.environ["AWS_DEFAULT_REGION"] = args.region
        # TrainingQueue takes no session: its submit_service_job call goes
        # through get_batch_boto_client(), i.e. boto3's DEFAULT session. That
        # session is built lazily and then cached, and the SageMaker Session
        # above already materialised it — so setting AWS_PROFILE here is too
        # late, and setdefault() would not override an ambient value anyway.
        # Submitting as robotics-old while passing the ad2 execution role failed
        # with "Cross-account pass role is not allowed". Rebuild the default
        # session explicitly so Batch is called as the target account.
        os.environ["AWS_PROFILE"] = profile
        boto3.setup_default_session(profile_name=profile, region_name=args.region)
        print(f"Submitting via Batch queue: {queue_name}")
        queued = Queue(queue_name).map(
            estimator, inputs=[fit_inputs or None], job_names=[job_name],
            priority=args.priority,
            share_identifier=os.environ.get("SM_FSS_IDENTIFIER", "default"),
            timeout={"attemptDurationSeconds": max_run},
            # Batch-level tags, distinct from the estimator's SageMaker tags
            # above. The ad2 account (385697366450) carries an Organizations SCP
            # that denies batch:SubmitServiceJob unless the *Batch* request
            # carries both tri.owner.email and tri.project — estimator tags ride
            # in the SageMaker payload and are invisible to that check, so
            # omitting this made every submission to fss-cv-wfm-p5en fail with
            # "explicit deny in a service control policy". Batch propagates
            # these onto the training job, which is why working jobs in that
            # queue show the tags on SageMaker but `payload Tags: None`.
            tags=tag_map(),
        )
        print(f"Queued: {queued}")
    else:
        print(f"Submitting training job: {job_name}")
        estimator.fit(inputs=fit_inputs or None, wait=False, job_name=job_name)

    print(f"Monitor: aws sagemaker describe-training-job --training-job-name {job_name} "
          f"--profile {profile} --region {args.region}")


if __name__ == "__main__":
    main()
