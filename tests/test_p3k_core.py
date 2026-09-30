"""Independent full-candidate CUDA checks for the read-only P3k probe."""
import math
import os
from pathlib import Path
import sys
import unittest

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from experiments.p3k_core import geometry, gram_matrix, oracle, production
from guandan.agents import GreedyAgent
from guandan.env import HandEnv
from guandan.learning.encoding import encode_action, encode_observation
from guandan.types import Action
from guandan_gpu.training import GPUTrainer


class GeometryTests(unittest.TestCase):
    def test_parallel_antiparallel_and_weighted_descent(self):
        # g_D=(2,0), g_A=(6,0), g_C=(-40,0).
        result = geometry([[4., 12., -80.], [12., 36., -240.],
                           [-80., -240., 1600.]])
        self.assertEqual(result['dmc_norm'], 2.)
        for name, norm, cosine, ratio, factor, conflict in (
            ('absolute', 6., 1., .3, 1.3, False),
            ('centered', 40., -1., 2., -1., True),
        ):
            with self.subTest(name=name):
                row = result[name]
                self.assertEqual(row['norm'], norm)
                self.assertEqual(row['cosine'], cosine)
                self.assertAlmostEqual(row['weighted_norm_ratio'], ratio)
                self.assertAlmostEqual(row['dmc_descent_factor'], factor)
                self.assertEqual(row['conflict'], conflict)

    def test_zero_auxiliary_and_zero_dmc_null_semantics(self):
        result = geometry([[4., 0., 0.], [0., 0., 0.], [0., 0., 9.]])
        self.assertIsNone(result['absolute']['cosine'])
        self.assertIsNone(result['absolute']['conflict'])
        self.assertEqual(result['absolute']['weighted_norm_ratio'], 0.)
        self.assertEqual(result['absolute']['dmc_descent_factor'], 1.)
        self.assertEqual(result['centered']['cosine'], 0.)
        self.assertAlmostEqual(result['centered']['weighted_norm_ratio'], .15)
        self.assertEqual(result['centered']['dmc_descent_factor'], 1.)
        zero_dmc = geometry([[0., 0., 0.], [0., 4., 0.], [0., 0., 0.]])
        for name in ('absolute', 'centered'):
            for key in ('cosine', 'conflict', 'weighted_norm_ratio', 'dmc_descent_factor'):
                self.assertIsNone(zero_dmc[name][key])

    def test_gram_matrix_matches_hand_dot_products(self):
        gradients = [torch.tensor([2., 0.]), torch.tensor([6., 0.]),
                     torch.tensor([-40., 0.])]
        self.assertEqual(gram_matrix(gradients),
                         [[4., 12., -80.], [12., 36., -240.], [-80., -240., 1600.]])

    def test_invalid_gram_rejected(self):
        for matrix in ([], [[1.]], [[1., 2., 3.]] * 2,
                       [[math.nan, 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                       [[math.inf, 0., 0.], [0., 0., 0.], [0., 0., 0.]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                geometry(matrix)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fixture selection never removes actions from HandEnv's legal set.
        env = HandEnv()
        env.reset(109090)
        agent = GreedyAgent()
        cls.multiple = cls.singleton = None
        while not env.state.terminal:
            obs = env.observe(env.state.current_player)
            legal = env.legal_actions(obs.player_id)
            if cls.multiple is None and 8 <= len(legal) <= 60 and len(legal) % 7:
                cls.multiple = (obs, legal)
            if cls.singleton is None and len(legal) == 1:
                cls.singleton = (obs, legal)
            if cls.multiple is not None and cls.singleton is not None:
                break
            env.step(obs.player_id, agent.act(obs, legal), state_version=obs.state_version)
        if cls.multiple is None or cls.singleton is None:
            raise AssertionError('required complete legal fixtures were not reached')

    def setUp(self):
        self.model = GPUTrainer(dict(seed=314510, epsilon=.1, lr=.001,
                                    batch_size=256, chunk_size=1024, num_envs=4)).model
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(next(self.model.parameters()).device.type, 'cuda')

    def dense_reference(self, obs, legal, executed, reward):
        # Independently normalize raw teacher scores, then use dense autograd.
        state = torch.tensor([encode_observation(obs)] * len(legal), device='cuda')
        actions = torch.tensor([encode_action(a) for a in legal], device='cuda')
        raw = [float(GreedyAgent().score(obs, a)) for a in legal]
        lo, hi = min(raw), max(raw)
        target = torch.tensor([0. if lo == hi else 1.6 * (v-lo)/(hi-lo)-.8
                               for v in raw], device='cuda')
        q = self.model(state, actions)
        losses = [(q[legal.index(executed)]-reward).square(),
                  (q-target).square().mean(),
                  ((q-q.mean())-(target-target.mean())).square().mean()]
        gradients = []
        for i, loss in enumerate(losses):
            values = torch.autograd.grad(loss, tuple(self.model.parameters()), retain_graph=i < 2)
            gradients.append(torch.cat([value.flatten() for value in values]).detach())
        return gradients, [loss.item() for loss in losses], q.detach(), target

    def assert_probe_close(self, actual, expected):
        ag, al, aq, at = actual
        eg, el, eq, et = expected
        self.assertEqual(len(ag), 3)
        for i, (a, e) in enumerate(zip(ag, eg)):
            with self.subTest(objective=i):
                torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)
                self.assertAlmostEqual(al[i], el[i], places=6)
        torch.testing.assert_close(aq, eq, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(at, et, atol=2e-6, rtol=2e-5)

    def test_production_and_layer_oracle_match_independent_dense_both_rewards(self):
        obs, legal = self.multiple
        self.assertGreater(len(legal), 7)
        self.assertNotEqual(len(legal) % 7, 0)
        # Include a nonzero executed index to detect accidentally selecting q[0].
        executed = legal[-1]
        for reward in (-1, 1):
            with self.subTest(reward=reward):
                reference = self.dense_reference(obs, legal, executed, reward)
                self.assert_probe_close(production(self.model, obs, legal, executed, reward, 7), reference)
                self.assert_probe_close(oracle(self.model, obs, legal, executed, reward), reference)

    def test_chunk_sizes_preserve_complete_candidate_objectives(self):
        obs, legal = self.multiple
        reference = self.dense_reference(obs, legal, legal[-1], -1)
        for chunk in (1, 7, len(legal) + 1):
            with self.subTest(chunk=chunk):
                output = production(self.model, obs, legal, legal[-1], -1, chunk)
                self.assertEqual(output[2].numel(), len(legal))
                self.assertEqual(output[3].numel(), len(legal))
                self.assert_probe_close(output, reference)

    def test_singleton_centered_zero_absolute_nonzero(self):
        obs, legal = self.singleton
        self.assertEqual(len(legal), 1)
        output = production(self.model, obs, legal, legal[0], 1, 1)
        self.assert_probe_close(output, self.dense_reference(obs, legal, legal[0], 1))
        self.assertEqual(output[1][2], 0.)
        self.assertEqual(output[0][2].count_nonzero().item(), 0)
        self.assertGreater(torch.linalg.vector_norm(output[0][1]).item(), 0.)

    def test_probe_preserves_parameters_and_clears_existing_gradients(self):
        obs, legal = self.multiple
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, .25)
        production(self.model, obs, legal, legal[-1], 1, 7)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_illegal_reward_and_executed_action_rejected(self):
        obs, legal = self.multiple
        for reward in (0, 2, -2, True, False, math.nan, math.inf, '1', None):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                production(self.model, obs, legal, legal[0], reward, 7)
        for action in (Action('invalid'), None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                production(self.model, obs, legal, action, 1, 7)
        with self.assertRaises(ValueError):
            production(self.model, obs, [], legal[0], 1, 7)


if __name__ == '__main__':
    unittest.main()
