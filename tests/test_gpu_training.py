"""Small contract tests; CUDA checks are skipped on a CPU-only runtime."""
from unittest import TestCase, main, mock, skipUnless
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
try:
    import torch
except ImportError:
    torch = None

CONFIG = dict(seed=123, epsilon=1.0, lr=0.001, batch_size=1000, chunk_size=128, num_envs=2)


@skipUnless(torch is not None, "PyTorch unavailable")
class ValidationTests(TestCase):
    def test_strict_config(self):
        from guandan_gpu.training import _validated_config
        self.assertEqual(_validated_config(CONFIG), CONFIG)
        for key, value in (("seed", True), ("seed", -1), ("epsilon", True),
                           ("epsilon", float("nan")), ("epsilon", 2), ("lr", 0),
                           ("num_envs", 0), ("num_envs", 9), ("num_envs", True),
                           ("batch_size", 0), ("chunk_size", False),
                           ("batch_size", 65537), ("chunk_size", 65537)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                _validated_config({**CONFIG, key: value})
        with self.assertRaises(ValueError):
            _validated_config({**CONFIG, "extra": 1})
        with self.assertRaises(ValueError):
            _validated_config({key: val for key, val in CONFIG.items() if key != "num_envs"})

    def test_development_deals(self):
        from guandan_gpu.training import _validated_deals
        good = [(100000, 2, 0), (109999, 14, 3)]
        self.assertEqual(_validated_deals(good, 2, []), good)
        for deals in ([good[0]], [good[0], good[0]], [(99999, 2, 0), good[1]],
                      [(110000, 2, 0), good[1]], [(True, 2, 0), good[1]],
                      [(100001, True, 0), good[1]], [(100001, 2, True), good[1]],
                      [(100001, 15, 0), good[1]], [(100001, 2, 4), good[1]]):
            with self.subTest(deals=deals), self.assertRaises(ValueError):
                _validated_deals(deals, 2, [])
        with self.assertRaises(ValueError):
            _validated_deals(good, 2, [100000])


@skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
class CUDATests(TestCase):
    def test_scoring_complete_candidates_and_first_tie(self):
        from guandan.env import HandEnv
        from guandan.learning.encoding import encode_action, encode_observation
        from guandan_gpu.training import GPUTrainer, score_many
        trainer = GPUTrainer(CONFIG)
        requests = []
        for seed in (100002, 100003):
            env = HandEnv()
            obs = env.reset(seed, initial_level=2, starting_player=0)
            requests.append((obs, env.legal_actions(0)))
        results = score_many(trainer.model, requests, 17)
        self.assertEqual(len(results), 2)
        for (obs, candidates), scores in zip(requests, results):
            states = torch.tensor([encode_observation(obs)] * len(candidates), device="cuda")
            actions = torch.tensor([encode_action(x) for x in candidates], device="cuda")
            with torch.no_grad():
                expected = trainer.model(states, actions).cpu()
            self.assertEqual(scores.device.type, "cpu")
            self.assertEqual(len(scores), len(candidates))
            torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            for parameter in trainer.model.parameters():
                parameter.zero_()
        for scores in score_many(trainer.model, requests, 17):
            self.assertEqual(int(torch.argmax(scores)), 0)

    def test_wave_frozen_parameters_cuda_and_failure_boundary(self):
        import guandan_gpu.training as training
        trainer = training.GPUTrainer(CONFIG)
        initial = [p.detach().clone() for p in trainer.model.parameters()]
        actual_env = training.HandEnv
        steps = []
        actual_step = actual_env.step

        def checked_step(env, *args, **kwargs):
            # This also covers research replay steps, all preceding optimization.
            self.assertTrue(all(torch.equal(old, p) for old, p in zip(initial, trainer.model.parameters())))
            self.assertEqual(trainer.updates, 0)
            steps.append(1)
            return actual_step(env, *args, **kwargs)

        with mock.patch.object(actual_env, "step", checked_step):
            result = trainer.train_wave([(100010, 2, 0), (100011, 3, 1)])
        self.assertEqual(result["wave"], 1)
        self.assertEqual(result["behavior_version"], 0)
        self.assertEqual(result["learning_version"], 1)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["samples"], sum(hand["steps"] for hand in result["hands"]))
        # serialize_replay internally validates by replaying as well.
        self.assertGreaterEqual(len(steps), result["samples"] * 2)
        self.assertEqual([hand["seed"] for hand in result["hands"]], [100010, 100011])
        self.assertTrue(all(hand["replay_verified"] for hand in result["hands"]))
        for key in ("forward_checks", "gradient_checks", "adam_tensor_checks"):
            self.assertGreater(result["device_proof"][key], 0)
        self.assertTrue(any(not torch.equal(old, p) for old, p in zip(initial, trainer.model.parameters())))
        self.assertTrue(trainer.ready_for_checkpoint)
        with self.assertRaises(ValueError):
            trainer.train_wave([(100010, 2, 0), (100012, 3, 1)])
        self.assertTrue(trainer.ready_for_checkpoint)
        with mock.patch.object(training, "encode_observation", side_effect=ValueError("bad features")):
            with self.assertRaisesRegex(ValueError, "bad features"):
                trainer.train_wave([(100012, 2, 0), (100013, 3, 1)])
        self.assertFalse(trainer.ready_for_checkpoint)
        with self.assertRaisesRegex(RuntimeError, "wave boundary"):
            trainer.train_wave([(100014, 2, 0), (100015, 3, 1)])


if __name__ == "__main__":
    main()
