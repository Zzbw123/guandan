"""DMC trainer contracts: boundary, terminal targets and real environment replay."""

from __future__ import annotations

from collections import Counter
from unittest import TestCase, main, mock, skipUnless

try:
    import torch
except ImportError:  # The base environment has no mandatory torch dependency.
    torch = None


CONFIG = {"seed": 314159, "epsilon": 1.0, "lr": 0.001,
          "batch_size": 1000, "chunk_size": 256}


@skipUnless(torch is not None, "PyTorch is not installed")
class TrainerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is not None:
            from guandan.learning.training import Trainer
            cls.Trainer = Trainer

    def test_config_and_development_seed_boundary(self):
        bad_configs = (
            {**CONFIG, "seed": True},
            {**CONFIG, "epsilon": float("nan")},
            {**CONFIG, "epsilon": 1.01},
            {**CONFIG, "lr": 0},
            {**CONFIG, "batch_size": 0},
            {**CONFIG, "chunk_size": False},
            {**CONFIG, "other": 1},
            {key: value for key, value in CONFIG.items() if key != "lr"},
        )
        for config in bad_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.Trainer(config)
        trainer = self.Trainer(CONFIG)
        self.assertTrue(trainer.ready_for_checkpoint)
        self.assertEqual(trainer.used_deal_seeds, [])
        for seed in (99999, 110000, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                trainer.train_episode(seed, 2, 0)
        self.assertTrue(trainer.ready_for_checkpoint)

    def test_complete_hand_uses_both_teams_terminal_targets_and_updates(self):
        import guandan.learning.training as training

        class TrackedEnv(training.HandEnv):
            chosen_players: list[int] = []

            def reset(self, *args, **kwargs):
                self._record_training = True
                return super().reset(*args, **kwargs)

            def step(self, player_id, action, state_version=None):
                result = super().step(player_id, action, state_version)
                if getattr(self, "_record_training", False):
                    self.chosen_players.append(player_id)
                return result

        trainer = self.Trainer(CONFIG)
        before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
        targets_seen: list[float] = []
        actual_mse = torch.nn.functional.mse_loss

        def collect_targets(predictions, targets, *args, **kwargs):
            targets_seen.extend(targets.tolist())
            return actual_mse(predictions, targets, *args, **kwargs)

        with mock.patch.object(training, "HandEnv", TrackedEnv), mock.patch.object(
            torch.nn.functional, "mse_loss", side_effect=collect_targets,
        ):
            result = trainer.train_episode(100500, 2, 0)

        self.assertEqual(result["episode"], 1)
        self.assertEqual(result["seed"], 100500)
        self.assertEqual(result["steps"], result["samples"])
        self.assertEqual(result["steps"], len(TrackedEnv.chosen_players))
        self.assertEqual(set(result["team_rewards"]), {-1, 1})
        self.assertEqual(
            Counter(targets_seen),
            Counter(result["team_rewards"][player % 2]
                    for player in TrackedEnv.chosen_players),
        )
        self.assertEqual(result["updates"], 1)
        self.assertTrue(result["replay_verified"])
        self.assertEqual(len(result["terminal_digest"]), 64)
        self.assertGreaterEqual(result["mean_loss"], 0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, trainer.model.parameters())))
        self.assertTrue(trainer.ready_for_checkpoint)
        self.assertEqual(trainer.used_deal_seeds, [100500])

    def test_failure_keeps_checkpoint_boundary_closed(self):
        import guandan.learning.training as training

        trainer = self.Trainer(CONFIG)
        with mock.patch.object(training, "encode_observation", side_effect=ValueError("bad features")):
            with self.assertRaisesRegex(ValueError, "bad features"):
                trainer.train_episode(100500, 2, 0)
        self.assertFalse(trainer.ready_for_checkpoint)
        self.assertEqual(trainer.episodes, 0)
        self.assertEqual(trainer.used_deal_seeds, [])
        with self.assertRaisesRegex(RuntimeError, "episode boundary"):
            trainer.train_episode(100501, 2, 0)


if __name__ == "__main__":
    main()
