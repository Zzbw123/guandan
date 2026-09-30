"""CPU DMC network and full-candidate scoring contracts."""

import importlib.util
import unittest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch
    from guandan.learning.encoding import ACTION_DIM, STATE_DIM, encode_action, encode_observation
    from guandan.learning.model import DMCNetwork, DMCAgent, NETWORK_VERSION, score_actions
    from guandan.env.hand_env import HandEnv


@unittest.skipUnless(_HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.obs = HandEnv().reset(seed=100500)
        self.model = DMCNetwork()

    def test_architecture_and_forward_shape(self):
        self.assertEqual("gd-dmc-network-v1", NETWORK_VERSION)
        self.assertEqual((128, STATE_DIM), tuple(self.model.state_fc.weight.shape))
        self.assertEqual((128, ACTION_DIM), tuple(self.model.action_fc.weight.shape))
        self.assertIsNone(self.model.action_fc.bias)
        self.assertEqual((64, 128), tuple(self.model.hidden_fc.weight.shape))
        self.assertEqual((1, 64), tuple(self.model.output_fc.weight.shape))
        result = self.model(torch.zeros((3, STATE_DIM)), torch.zeros((3, ACTION_DIM)))
        self.assertEqual((3,), tuple(result.shape))
        self.assertTrue(torch.all(result.abs() <= 1).item())

    def test_score_all_candidates_in_chunks_matches_direct_network(self):
        env = HandEnv()
        env.reset(seed=100500)
        candidates = env.legal_actions(0)
        self.assertGreater(len(candidates), 256)
        calls = []
        hook = self.model.state_fc.register_forward_hook(lambda *_: calls.append(1))
        try:
            scores = score_actions(self.model, self.obs, candidates, chunk_size=37)
        finally:
            hook.remove()
        self.assertEqual([1], calls)
        self.assertEqual((len(candidates),), tuple(scores.shape))
        self.assertEqual("cpu", scores.device.type)
        states = torch.tensor((encode_observation(self.obs),) * len(candidates), dtype=torch.float32)
        encoded_actions = torch.tensor(tuple(encode_action(a) for a in candidates), dtype=torch.float32)
        with torch.no_grad():
            expected = self.model(states, encoded_actions)
        self.assertTrue(torch.allclose(scores, expected, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(
            scores, score_actions(self.model, self.obs, candidates, 1), rtol=1e-6, atol=1e-7,
        ))

    def test_first_candidate_wins_ties_and_bad_inputs_fail(self):
        env = HandEnv()
        env.reset(seed=100500)
        candidates = env.legal_actions(0)[:2]
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.zero_()
        self.assertIs(candidates[0], DMCAgent(self.model, chunk_size=1).act(self.obs, candidates))
        for bad in (0, -1, 1.5, True):
            with self.subTest(chunk_size=bad), self.assertRaises(ValueError):
                score_actions(self.model, self.obs, candidates, bad)
        with self.assertRaises(ValueError):
            score_actions(self.model, self.obs, [])
        with torch.no_grad():
            self.model.output_fc.bias.fill_(float("nan"))
        with self.assertRaises(ValueError):
            score_actions(self.model, self.obs, candidates, 1)


if __name__ == "__main__":
    unittest.main()
