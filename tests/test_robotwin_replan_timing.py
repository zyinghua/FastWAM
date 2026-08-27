"""Policy timing regression tests; no model weights, CUDA, or simulator required.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_robotwin_replan_timing.py' -v

Only NumPy and Pillow are needed; model/configuration imports are stubbed.
"""

import importlib.util
import logging
import sys
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


def _load_policy_module():
    dependencies = {
        "torch": {"dtype": object, "Tensor": object, "no_grad": nullcontext},
        "hydra": {"compose": Mock(), "initialize_config_dir": Mock()},
        "hydra.core.global_hydra": {"GlobalHydra": Mock()},
        "hydra.utils": {"instantiate": Mock()},
        "omegaconf": {"DictConfig": dict, "OmegaConf": Mock()},
        "fastwam.datasets.lerobot.processors.fastwam_processor": {"FastWAMProcessor": Mock()},
        "fastwam.datasets.lerobot.robot_video_dataset": {"DEFAULT_PROMPT": "Task: {task}"},
        "fastwam.datasets.lerobot.utils.normalizer": {"load_dataset_stats_from_json": Mock()},
    }
    stubs = {}
    for name, attributes in dependencies.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        stubs[name] = stub

    policy_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/robotwin/fastwam_policy/deploy_policy.py"
    )
    spec = importlib.util.spec_from_file_location("_fastwam_timing_test_policy", policy_path)
    module = importlib.util.module_from_spec(spec)
    # Do not leave dependency stubs or the policy's sys.path edits in other tests.
    with patch.dict(sys.modules, stubs), patch.object(sys, "path", sys.path.copy()):
        spec.loader.exec_module(module)
    return module


class _FakeModel:
    def __init__(self):
        self.actions = np.arange(32 * 14, dtype=np.float32).reshape(32, 14)
        self.calls = []

    def load_checkpoint(self, path):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer_action(self, **kwargs):
        self.calls.append(kwargs)
        return {"action": self.actions}


class ReplanTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_policy_module()

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.model = _FakeModel()
        self.register = self.stack.enter_context(patch.object(self.module.atexit, "register"))
        self.stack.enter_context(
            patch.object(self.module, "instantiate", side_effect=[self.model, Mock()])
        )
        self.stack.enter_context(
            patch.object(self.module.OmegaConf, "create", return_value=SimpleNamespace())
        )
        self.observation = {"joint_action": {"vector": np.zeros(14, dtype=np.float32)}}

    def make_policy(self, timing_enabled=True, replan_steps=24):
        policy = self.module.WorldActionRobotWinPolicy(
            model_cfg={},
            processor_cfg={},
            checkpoint_path="unused.pt",
            dataset_stats_path=Path("unused_stats.json"),
            device="cpu",
            model_dtype="float32",
            action_horizon=32,
            replan_steps=replan_steps,
            num_inference_steps=10,
            sigma_shift=None,
            seed=42,
            text_cfg_scale=1.0,
            negative_prompt="",
            rand_device="cpu",
            tiled=False,
            timing_enabled=timing_enabled,
            num_video_frames=33,
        )
        policy._build_robotwin_image_tensor = Mock(return_value="image")
        policy._normalize_state = Mock(return_value="state")
        policy._denormalize_action = Mock(side_effect=lambda action: action[None])
        return policy

    def test_times_model_call_before_denormalization(self):
        policy = self.make_policy()
        events = []
        policy._build_robotwin_image_tensor.side_effect = lambda obs: events.append("image")
        policy._normalize_state.side_effect = lambda state: events.append("state")

        def infer_action(**kwargs):
            events.append("infer")
            return {"action": self.model.actions}

        def clock():
            events.append("clock")
            return float(events.count("clock"))

        def denormalize(action):
            events.append("denormalize")
            return action[None]

        self.model.infer_action = infer_action
        policy._denormalize_action.side_effect = denormalize
        with patch.object(self.module.time, "perf_counter", side_effect=clock):
            actions = policy._infer_action_chunk(self.observation, "lift pot")

        self.assertEqual(events, ["image", "state", "clock", "infer", "clock", "denormalize"])
        self.assertEqual(policy._replan_times, [1.0])
        self.assertEqual(policy.get_timing_rollout(), {"infer_s": 1.0, "sim_s": 0.0})
        np.testing.assert_array_equal(actions, self.model.actions)

    def test_disabled_timing_does_not_read_clock_or_register_exit_callback(self):
        policy = self.make_policy(timing_enabled=False)
        env = Mock()
        with patch.object(self.module.time, "perf_counter") as clock:
            policy.step(env, self.observation)
            policy.reset()
        clock.assert_not_called()
        self.register.assert_not_called()
        self.assertEqual(policy._replan_times, [])
        self.assertEqual(policy.get_timing_rollout(), {"infer_s": 0.0, "sim_s": 0.0})

    def test_summary_separates_first_call_and_flushes_only_once(self):
        policy = self.make_policy()
        policy._replan_times = [9.0, 1.0, 2.0, 3.0]
        with self.assertLogs(self.module.logger, level="INFO") as logs:
            policy._log_replan_timing()
        self.assertEqual(
            logs.records[0].getMessage(),
            "Replan timing | init 9.000s | steady mean 2.000s min 1.000s max 3.000s (n=3)",
        )
        self.assertEqual(policy._replan_times, [])
        with self.assertNoLogs(self.module.logger):
            policy._log_replan_timing()

    def test_single_replan_has_no_steady_samples(self):
        policy = self.make_policy()
        policy._replan_times = [1.25]
        with self.assertLogs(self.module.logger, level="INFO") as logs:
            policy._log_replan_timing()
        self.assertEqual(
            logs.records[0].getMessage(),
            "Replan timing | init 1.250s | steady mean nans min nans max nans (n=0)",
        )

    def test_reset_flushes_previous_episode_and_clears_counters(self):
        policy = self.make_policy()
        with self.assertNoLogs(self.module.logger):
            policy.reset()  # RoboTwin also resets before the very first episode.
        with patch.object(self.module.time, "perf_counter", side_effect=[1.0, 3.0]):
            policy._infer_action_chunk(self.observation, "lift pot")
        policy.pending_actions.append(self.model.actions[0])
        policy.step_count = 5
        policy._timing_rollout["sim_s"] = 10.0
        with self.assertLogs(self.module.logger, level="INFO"):
            policy.reset()
        self.assertEqual(policy.episode_count, 2)
        self.assertEqual(policy.step_count, 0)
        self.assertFalse(policy.pending_actions)
        self.assertEqual(policy._replan_times, [])
        self.assertEqual(policy.get_timing_rollout(), {"infer_s": 0.0, "sim_s": 0.0})
        with patch.object(self.module.time, "perf_counter", side_effect=[5.0, 6.5]):
            policy._infer_action_chunk(self.observation, "lift pot")
        self.assertEqual(policy._replan_times, [1.5])

    def test_exit_callback_flushes_final_episode_without_reset(self):
        policy = self.make_policy()
        self.register.assert_called_once_with(policy._log_replan_timing)
        callback = self.register.call_args.args[0]
        policy._replan_times = [4.0, 2.0]
        with self.assertLogs(self.module.logger, level="INFO") as logs:
            callback()
        self.assertIn("init 4.000s | steady mean 2.000s", logs.output[0])
        with self.assertNoLogs(self.module.logger):
            callback()

    def test_counts_replans_not_executed_actions_and_keeps_sim_time_separate(self):
        policy = self.make_policy(replan_steps=16)
        env = Mock()
        env.get_instruction.return_value = "lift pot"
        # One model call (2s) followed by 16 simulator steps (1s each).
        clock_values = [0.0, 2.0] + [value for _ in range(16) for value in (0.0, 1.0)]
        with patch.object(self.module.time, "perf_counter", side_effect=clock_values):
            policy.step(env, self.observation)
            for _ in range(15):
                self.assertFalse(policy.should_request_observation())
                policy.step(env, None)
        self.assertTrue(policy.should_request_observation())
        self.assertEqual(policy._replan_times, [2.0])
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(policy.get_timing_rollout(), {"infer_s": 2.0, "sim_s": 16.0})
        for index, call in enumerate(env.take_action.call_args_list):
            np.testing.assert_array_equal(call.args[0], self.model.actions[index])
            self.assertEqual(call.kwargs, {"action_type": "qpos"})

        with patch.object(self.module.time, "perf_counter", side_effect=[0.0, 3.0, 0.0, 1.0]):
            policy.step(env, self.observation)
        self.assertEqual(policy._replan_times, [2.0, 3.0])
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(self.model.calls[1]["action_horizon"], 32)
        self.assertEqual(self.model.calls[1]["num_inference_steps"], 10)

    def test_failed_inference_is_not_recorded_as_completed_replan(self):
        policy = self.make_policy()

        def fail(**kwargs):
            raise RuntimeError("inference failed")

        self.model.infer_action = fail
        with patch.object(self.module.time, "perf_counter", return_value=1.0):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                policy._infer_action_chunk(self.observation, "lift pot")
        self.assertEqual(policy._replan_times, [])
        self.assertEqual(policy.get_timing_rollout()["infer_s"], 0.0)

    def test_info_logging_is_visible_when_root_logger_is_critical(self):
        with patch.object(logging.getLogger(), "level", logging.CRITICAL):
            self.assertEqual(self.module.logger.getEffectiveLevel(), logging.INFO)


if __name__ == "__main__":
    unittest.main()
