"""Independent dense references for complete-candidate P5f objectives."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import torch

from experiments.p5f_objective import fit_batch, raw_scores, score_requests, teacher_labels
from guandan.agents import GreedyAgent
from guandan.learning.encoding import encode_action, encode_observation
from guandan.learning.model import DMCNetwork
from guandan.types import Action, PlayerObservation


def small_request():
    obs = PlayerObservation(0, (0, 1, 54), 2, 0, (3, 3, 3, 3), (),
                            None, None, (), (), 0, False, None)
    actions = [Action("single", (0,), 2), Action("single", (54,), 2),
               Action("single", (1,), 2), Action("pair", (0, 54), 2)]
    return obs, actions


def dense(model, requests, objective):
    terms = []
    for obs, legal in requests:
        device = next(model.parameters()).device
        state = torch.tensor([encode_observation(obs)] * len(legal), device=device)
        actions = torch.tensor([encode_action(a) for a in legal], device=device)
        logits = raw_scores(model, state, actions)
        labels = teacher_labels(obs, legal)
        if objective == "regression":
            target = torch.tensor(labels["targets"], device=device)
            terms.append((torch.tanh(logits) - target).square().mean())
        else:
            optimal = torch.tensor(labels["optimal"], dtype=torch.bool, device=device)
            terms.append(torch.logsumexp(logits, 0) - torch.logsumexp(logits[optimal], 0))
    return torch.stack(terms).mean()


class LabelAndCPUScoreTests(unittest.TestCase):
    def test_exact_ties_duplicates_and_normalization(self):
        obs, legal = small_request()
        labels = teacher_labels(obs, legal)
        self.assertEqual(labels["scores"][0], labels["scores"][1])
        self.assertEqual(labels["targets"][0], labels["targets"][1])
        self.assertEqual(labels["optimal"], [False, False, False, True])
        self.assertEqual(min(labels["targets"]), -.8)
        self.assertEqual(max(labels["targets"]), .8)
        tied = teacher_labels(obs, legal[:2])
        self.assertEqual(tied["targets"], [0., 0.])
        self.assertEqual(tied["optimal"], [True, True])

    def test_score_requests_matches_direct_logits_and_boundaries(self):
        torch.manual_seed(413)
        model = DMCNetwork()
        request = small_request()
        state = torch.tensor([encode_observation(request[0])] * len(request[1]))
        actions = torch.tensor([encode_action(a) for a in request[1]])
        with torch.no_grad():
            expected = raw_scores(model, state, actions)
        for size in (1, 3, 1024):
            got = score_requests(model, [request, (request[0], request[1][:1])], size)
            torch.testing.assert_close(got[0], expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(len(got[1]), 1)
        model.requires_grad_(False).eval()
        torch.testing.assert_close(score_requests(model, [request], 2)[0],
                                   expected, rtol=1e-6, atol=1e-7)

    def test_invalid_inputs(self):
        obs, legal = small_request()
        with self.assertRaises(ValueError): teacher_labels(obs, [])
        with self.assertRaises(ValueError): teacher_labels(obs, [Action("single", (9,), 3)])
        with patch.object(GreedyAgent, "score", side_effect=[1., 2.]):
            with self.assertRaisesRegex(ValueError, "identically encoded"):
                teacher_labels(obs, legal[:2])
        model = DMCNetwork()
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): score_requests(model, [(obs, legal)], size)
        with torch.no_grad(): next(model.parameters()).view(-1)[0] = math.inf
        with self.assertRaises(ValueError): score_requests(model, [(obs, legal)])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required; controller schedules GPU tests")
class CUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(414)
        self.model = DMCNetwork().cuda()
        obs, legal = small_request()
        self.requests = [(obs, legal), (obs, legal[:2]), (obs, legal[:1])]

    def test_dense_loss_gradient_and_one_adam_each_objective(self):
        for objective in ("regression", "ranking"):
            for size in (1, 3, 1024):
                with self.subTest(objective=objective, chunk=size):
                    a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
                    oa = torch.optim.Adam(a.parameters(), lr=.001, eps=1e-4, capturable=True)
                    ob = torch.optim.Adam(b.parameters(), lr=.001, eps=1e-4, capturable=True)
                    expected = dense(b, self.requests, objective)
                    expected.backward()
                    result = fit_batch(a, oa, self.requests, objective, size)
                    self.assertAlmostEqual(result["loss"], expected.item(), delta=2e-6)
                    self.assertEqual(result["per_request_sizes"], [4, 2, 1])
                    self.assertEqual(result["scored_candidates"], 7)
                    self.assertEqual(result["chunks"], math.ceil(7 / size))
                    for x, y in zip(a.parameters(), b.parameters()):
                        torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)
                    ob.step()
                    for x, y in zip(a.parameters(), b.parameters()):
                        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
                    self.assertTrue(all(state["step"].item() == 1 for state in oa.state.values()))

    def test_ranking_translation_extreme_and_zero_gradient(self):
        logits = torch.tensor([10000., 9999., -10000.], device="cuda", requires_grad=True)
        mask = torch.tensor([True, False, False], device="cuda")
        loss = torch.logsumexp(logits, 0) - torch.logsumexp(logits[mask], 0)
        shifted = logits + 1234.
        other = torch.logsumexp(shifted, 0) - torch.logsumexp(shifted[mask], 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), other.item(), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        obs, legal = small_request()
        for request in ((obs, legal[:1]), (obs, legal[:2])):
            model = copy.deepcopy(self.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-4, capturable=True)
            result = fit_batch(model, optimizer, [request], "ranking", 1)
            self.assertEqual(result["loss"], 0.)
            self.assertTrue(all(p.grad.count_nonzero().item() == 0 for p in model.parameters()))

    def test_large_complete_and_cross_chunk_top_ties(self):
        from benchmark_learning_runtime import high_branch_fixture
        obs, legal = small_request()
        crossed = [(obs, [legal[0], legal[2], legal[3], legal[0], legal[2], legal[3], legal[0]])]
        labels = teacher_labels(*crossed[0])
        self.assertEqual([i for i, flag in enumerate(labels["optimal"]) if flag], [2, 5])
        a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
        expected = dense(b, crossed, "ranking")
        expected.backward()
        result = fit_batch(a, torch.optim.Adam(a.parameters(), lr=.001, eps=1e-4, capturable=True),
                           crossed, "ranking", 3)
        self.assertAlmostEqual(result["loss"], expected.item(), delta=2e-6)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-5)

        request = high_branch_fixture()
        labels = teacher_labels(*request)
        self.assertEqual(len(labels["optimal"]), 8769)
        self.assertTrue(any(labels["optimal"]))
        for size in (37, 1024):
            model = copy.deepcopy(self.model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-4, capturable=True)
            result = fit_batch(model, optimizer, [request], "ranking", size)
            self.assertEqual(result["scored_candidates"], 8769)
            self.assertEqual(result["per_request_sizes"], [8769])
            self.assertTrue(math.isfinite(result["loss"]))

    def test_extreme_model_logits_stay_finite(self):
        model = copy.deepcopy(self.model)
        with torch.no_grad():
            model.output_fc.weight.zero_()
            model.output_fc.bias.fill_(10000.)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-4, capturable=True)
        result = fit_batch(model, optimizer, [self.requests[0]], "ranking", 2)
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_rejections_leave_parameters_unchanged(self):
        model = copy.deepcopy(self.model)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-4, capturable=True)
        for kwargs in ({"objective": "bad"}, {"objective": "ranking", "chunk_size": 0}):
            with self.assertRaises(ValueError): fit_batch(model, optimizer, self.requests, **kwargs)
        with self.assertRaises(ValueError): fit_batch(model, optimizer, self.requests + [(self.requests[0][0], [])], "ranking")
        other = DMCNetwork().cuda()
        with self.assertRaises(ValueError):
            fit_batch(model, torch.optim.Adam(other.parameters(), eps=1e-4, capturable=True), self.requests, "ranking")
        with self.assertRaises(TypeError):
            fit_batch(model, torch.optim.SGD(model.parameters(), lr=.001), self.requests, "ranking")
        with self.assertRaises(ValueError):
            fit_batch(model, torch.optim.Adam(model.parameters(), lr=.001, eps=1e-4), self.requests, "ranking")
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
