"""Independent controller checks for leakage, persistence and schedule boundaries."""
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from guandan.env import HandEnv
from guandan.learning.encoding import encode_observation


class ControllerInformationTests(unittest.TestCase):
    def test_hidden_hand_permutation_cannot_change_features_or_legal_actions(self):
        env = HandEnv()
        env.reset(100512, initial_level=9)
        hands = list(env.state.initial_hands)
        hands[1], hands[3] = hands[3], hands[1]
        alternative = HandEnv.from_hands(hands, initial_level=9)
        self.assertNotEqual(env.state_digest(), alternative.state_digest())
        self.assertEqual(encode_observation(env.observe(0)), encode_observation(alternative.observe(0)))
        self.assertEqual(env.legal_actions(0), alternative.legal_actions(0))

    def test_rotating_seats_preserves_current_and_historical_features(self):
        env = HandEnv()
        env.reset(100513)
        env.step(0, env.legal_actions(0)[0])
        obs = env.observe(env.state.current_player)
        shift = 1
        seat = lambda p: None if p is None else (p+shift)%4
        counts = tuple(obs.remaining_counts[(p-shift)%4] for p in range(4))
        rotated = replace(obs, player_id=seat(obs.player_id), current_player=seat(obs.current_player),
                          remaining_counts=counts, finish_order=tuple(map(seat, obs.finish_order)),
                          last_player=seat(obs.last_player), passed_players=tuple(map(seat, obs.passed_players)),
                          history=tuple(replace(e, player=seat(e.player), finished=tuple(map(seat,e.finished)))
                                        for e in obs.history))
        self.assertEqual(encode_observation(obs), encode_observation(rotated))
        self.assertEqual(encode_observation(obs), encode_observation(replace(obs, state_version=999999)))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional CPU PyTorch unavailable")
class ControllerCheckpointTests(unittest.TestCase):
    def setUp(self):
        from guandan.learning.training import Trainer
        self.trainer = Trainer(dict(seed=19, epsilon=.1, lr=.001, batch_size=64, chunk_size=256))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"checkpoint"

    def test_load_preserves_all_parameters_rng_and_decision(self):
        import torch
        from guandan.learning.checkpoint import save_checkpoint, load_checkpoint
        from guandan.learning.model import DMCAgent
        env = HandEnv()
        obs = env.reset(100511)
        legal = env.legal_actions(0)
        action = DMCAgent(self.trainer.model).act(obs, legal)
        receipt = save_checkpoint(self.trainer, self.path)
        restored = load_checkpoint(self.path, receipt["sha256"])
        self.assertEqual(action, DMCAgent(restored.model).act(obs, legal))
        self.assertEqual(self.trainer.rng.getstate(), restored.rng.getstate())
        for a, b in zip(self.trainer.model.parameters(), restored.model.parameters()):
            self.assertTrue(torch.equal(a, b))
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.trainer, self.path)
        with self.assertRaisesRegex(ValueError, "pinned"):
            load_checkpoint(self.path, "0"*64)

    def test_corrupt_bytes_rejected_before_deserialization(self):
        from guandan.learning.checkpoint import save_checkpoint, load_checkpoint
        save_checkpoint(self.trainer, self.path)
        file = self.path/"checkpoint.pt"
        data = bytearray(file.read_bytes()); data[len(data)//2] ^= 1
        file.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "hash"):
            load_checkpoint(self.path)

    def test_rehashed_wrong_version_and_invalid_seed_rejected(self):
        import torch
        from guandan.learning.checkpoint import save_checkpoint, load_checkpoint
        save_checkpoint(self.trainer, self.path)
        file = self.path/"checkpoint.pt"
        original = torch.load(file, weights_only=True)
        original_receipt = json.loads((self.path/"manifest.json").read_text())
        for modification in ("version", "seed"):
            payload = dict(original)
            receipt = dict(original_receipt)
            if modification == "version":
                payload["versions"] = {**payload["versions"], "feature":"invalid"}
                receipt["versions"] = payload["versions"]
            else:
                payload["episodes"] = receipt["episodes"] = 1
                payload["used_deal_seeds"] = receipt["used_deal_seeds"] = [9000000]
            torch.save(payload, file)
            data = file.read_bytes()
            receipt.update(sha256=sha256(data).hexdigest(), bytes=len(data))
            (self.path/"manifest.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "versions" if modification == "version" else "seed"):
                load_checkpoint(self.path)

    def test_high_branching_fixture_scores_every_candidate(self):
        import torch
        from guandan.learning.model import score_actions
        from guandan.rules.cards import card_id
        hand = tuple(card_id(rank, suit, copy) for rank in (2, 3, 4)
                     for suit in range(4) for copy in range(2)) + (18, 72, 7)
        rest = [c for c in range(108) if c not in hand]
        env = HandEnv.from_hands((hand, tuple(rest[:27]), tuple(rest[27:54]), tuple(rest[54:])), 7)
        actions = env.legal_actions(0)
        self.assertEqual(8769, len(actions))
        calls = []
        handle = self.trainer.model.action_fc.register_forward_hook(lambda module, args, output: calls.append(len(args[0])))
        try:
            scores = score_actions(self.trainer.model, env.observe(0), actions, 256)
        finally:
            handle.remove()
        self.assertEqual(8769, sum(calls))
        self.assertLessEqual(max(calls), 256)
        self.assertEqual(8769, len(scores))
        self.assertTrue(torch.isfinite(scores).all())

    def test_failed_episode_checkpoint_is_refused(self):
        from guandan.learning.checkpoint import save_checkpoint
        self.trainer.ready_for_checkpoint = False
        with self.assertRaisesRegex(ValueError, "boundary"):
            save_checkpoint(self.trainer, self.path)
        self.assertFalse(self.path.exists())

    def test_fixed_schedule_covers_both_teams_and_all_seats(self):
        from guandan.learning.evaluation import smoke_schedule, summarize
        trials = smoke_schedule()
        self.assertEqual(48, len(trials))
        self.assertEqual(48, len({t.trial_id for t in trials}))
        self.assertEqual({101000,101001}, {t.deal_seed for t in trials})
        for opponent in ("greedy", "random", "team"):
            selected = [t for t in trials if t.matchup_id.endswith(opponent)]
            self.assertEqual(16, len(selected))
            self.assertEqual({0,1,2,3}, {t.focal_seat for t in selected})
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
