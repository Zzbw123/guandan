"""Independent algebra and dense CUDA autograd oracles for P3i."""
import copy
import math
import unittest

from experiments.p3j_objective import centered_backward
from experiments.p3i_objective import decompose
try:
    import torch
except ImportError:
    torch = None


class AlgebraTests(unittest.TestCase):
    def test_offset_and_relative_error_separate(self):
        result = decompose([3., 5., 4.], [0., 2., 1.])
        self.assertEqual(result['offset_mse'], 9.)
        self.assertEqual(result['centered_mse'], 0.)
        result = decompose([1., -1., 0.], [-1., 1., 0.])
        self.assertEqual(result['offset_mse'], 0.)
        self.assertAlmostEqual(result['centered_mse'], 8/3)
        self.assertLess(abs(result['identity_residual']), 1e-12)

    def test_translation_and_singleton(self):
        q, t = [.1, .6, -.2], [-.8, .8, .3]
        a = decompose(q, t)
        b = decompose([x+4 for x in q], [x-3 for x in t])
        self.assertAlmostEqual(a['centered_mse'], b['centered_mse'])
        self.assertEqual(decompose([.7], [-.8])['centered_mse'], 0.)

    def test_bad_arrays(self):
        for q, t in (([], []), ([1], []), ([math.nan], [0]), ([0], [math.inf])):
            with self.assertRaises(ValueError):
                decompose(q, t)

    def test_bad_api_arguments(self):
        for w in (-1, True, math.nan, math.inf):
            with self.assertRaises(ValueError): centered_backward(None, [], w)
        for c in (0, -1, False, 1.5, 65537):
            with self.assertRaises(ValueError): centered_backward(None, [], .1, c)
        with self.assertRaises(ValueError): centered_backward(None, [])


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA required')
class CUDATests(unittest.TestCase):
    def setUp(self):
        from guandan_gpu.training import GPUTrainer
        self.model = GPUTrainer(dict(seed=314410, epsilon=.1, lr=.001,
                                    batch_size=256, chunk_size=1024, num_envs=4)).model

    def requests(self):
        from experiments.inspect_p3h import states
        from guandan.env import HandEnv
        from guandan.agents import GreedyAgent
        chosen = []
        for _, obs, legal in states():
            if len(legal) <= 45 and len(legal) not in [len(x[1]) for x in chosen]:
                chosen.append((obs, legal))
                if len(chosen) == 3: break
        # Reach a real forced-single-action state rather than truncate a list.
        env = HandEnv(); env.reset(109090)
        while not env.state.terminal:
            obs = env.observe(env.state.current_player)
            legal = env.legal_actions(obs.player_id)
            if len(legal) == 1:
                chosen.append((obs, legal)); break
            env.step(obs.player_id, GreedyAgent().act(obs, legal), state_version=obs.state_version)
        self.assertEqual(len(chosen), 4)
        return chosen

    def dense(self, model, req):
        from guandan.agents import GreedyAgent
        from guandan.learning.encoding import encode_observation, encode_action
        terms, selected = [], []
        for obs, legal in req:
            q = model(torch.tensor([encode_observation(obs)]*len(legal), device='cuda'),
                      torch.tensor([encode_action(a) for a in legal], device='cuda'))
            raw = [float(GreedyAgent().score(obs, a)) for a in legal]
            lo, hi = min(raw), max(raw)
            target = torch.tensor([0. if lo == hi else 1.6*(v-lo)/(hi-lo)-.8 for v in raw], device='cuda')
            terms.append(((q-q.mean())-(target-target.mean())).square().mean())
            selected.append((q[0]-1.)**2)
        return torch.stack(terms).mean(), torch.stack(selected).mean()

    def test_dense_gradients_chunks_and_one_adam_step(self):
        req = self.requests()
        for chunk in (1, 7, 1024):
            with self.subTest(chunk=chunk):
                a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
                _, dmc = self.dense(a, req); dmc.backward()
                result = centered_backward(a, req, .1, chunk)
                aux, dmc = self.dense(b, req); (dmc+.1*aux).backward()
                self.assertAlmostEqual(result['loss'], aux.item(), places=6)
                self.assertEqual(result['candidates'], sum(len(l) for _, l in req))
                self.assertEqual(result['requests'], 4)
                for x, y in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
                oa = torch.optim.Adam(a.parameters(), lr=.001, capturable=True)
                ob = torch.optim.Adam(b.parameters(), lr=.001, capturable=True)
                oa.step(); ob.step()
                for x, y in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)

    def test_singleton_zero_and_zero_weight_preserves_gradients(self):
        req = self.requests()[-1:]
        out = centered_backward(self.model, req)
        self.assertEqual(out['loss'], 0.)
        for p in self.model.parameters(): self.assertEqual(p.grad.count_nonzero().item(), 0)
        for p in self.model.parameters(): p.grad.fill_(.25)
        old = [p.grad.clone() for p in self.model.parameters()]
        centered_backward(self.model, self.requests(), 0., 7)
        for p, g in zip(self.model.parameters(), old): self.assertTrue(torch.equal(p.grad, g))

    def test_output_gradient_translation_and_global_not_chunk_mean(self):
        q = torch.tensor([.1, .5, -.9, .3, .8], device='cuda', requires_grad=True)
        t = torch.tensor([-.8, .2, .8, -.3, .7], device='cuda')
        r = q-t; loss = (r-r.mean()).square().mean()
        g, = torch.autograd.grad(loss, q)
        self.assertAlmostEqual(g.sum().item(), 0., places=6)
        v = (q.detach()+2.).requires_grad_(); u = t-3.
        residual = v-u; translated = (residual-residual.mean()).square().mean()
        h, = torch.autograd.grad(translated, v)
        torch.testing.assert_close(loss, translated, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(g, h, atol=2e-6, rtol=2e-5)
        wrong = sum((r[i:i+2]-r[i:i+2].mean()).square().sum() for i in range(0, 5, 2))/5
        self.assertGreater(abs(loss.item()-wrong.item()), .01)

    def test_high_candidates_dense_gradient(self):
        from benchmark_learning_runtime import high_branch_fixture
        req = [high_branch_fixture()]
        a, b = self.model, copy.deepcopy(self.model)
        result = centered_backward(a, req, .1, 1024)
        aux, _ = self.dense(b, req); (.1*aux).backward()
        self.assertEqual(result['candidates'], 8769)
        self.assertAlmostEqual(result['loss'], aux.item(), places=6)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)

    def test_parameters_unchanged_and_nonfinite_rejected(self):
        before = copy.deepcopy(self.model.state_dict())
        centered_backward(self.model, self.requests(), .1, 7)
        for k, v in self.model.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))
        with torch.no_grad(): next(self.model.parameters()).flatten()[0] = math.nan
        with self.assertRaises(ValueError): centered_backward(self.model, self.requests())

    def test_invalid_late_request_does_not_add_gradients(self):
        req = self.requests()+[(self.requests()[0][0], [])]
        with self.assertRaises(ValueError): centered_backward(self.model, req)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
