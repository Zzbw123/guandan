"""CPU target contracts; opt in to GPU objective diagnostics with RUN_P3F_CUDA=1."""
import copy
import os
import unittest
from unittest import mock

from experiments.p3f_teacher import fit_batch, teacher_targets
from guandan.agents import GreedyAgent
from guandan.types import Action, PlayerObservation


def observation(hand=(0, 54, 1)):
    return PlayerObservation(0, hand, 2, 0, (len(hand), 3, 3, 3), (),
                             None, None, (), (), 0, False, None)


class TargetTests(unittest.TestCase):
    def test_normalization_order_and_every_candidate(self):
        obs = observation()
        legal = [Action("single", (0,), 2), Action("pair", (0, 54), 2),
                 Action("single", (1,), 3)]
        raw = [GreedyAgent().score(obs, action) for action in legal]
        expected = [1.6 * (value - min(raw)) / (max(raw) - min(raw)) - .8 for value in raw]
        with mock.patch.object(GreedyAgent, "score", autospec=True,
                               side_effect=lambda self, o, a: raw[legal.index(a)]) as scoring:
            self.assertEqual(teacher_targets(obs, legal), expected)
            self.assertEqual(scoring.call_count, len(legal))
        self.assertEqual(teacher_targets(obs, list(reversed(legal))), list(reversed(expected)))
        self.assertEqual(min(expected), -.8)
        self.assertEqual(max(expected), .8)

    def test_equal_singleton_and_real_finish(self):
        obs = observation()
        self.assertEqual(teacher_targets(obs, [Action("single", (0,), 2)]), [0.0])
        self.assertEqual(teacher_targets(obs, [Action("single", (0,), 2),
                                              Action("single", (54,), 2)]), [0.0, 0.0])
        obs = observation((0, 54))
        legal = [Action("single", (0,), 2), Action("pair", (0, 54), 2)]
        self.assertEqual(GreedyAgent().score(obs, legal[1]), 1000.0)
        self.assertEqual(teacher_targets(obs, legal), [-.8, .8])

    def test_copy_equivalent_targets(self):
        obs = observation()
        legal = [Action("single", (0,), 2), Action("single", (1,), 3),
                 Action("pair", (0, 54), 2)]
        swapped = observation((54, 0, 55))
        equivalents = [Action("single", (54,), 2), Action("single", (55,), 3),
                      Action("pair", (54, 0), 2)]
        self.assertEqual(teacher_targets(obs, legal), teacher_targets(swapped, equivalents))
        self.assertEqual(teacher_targets(obs, legal), teacher_targets(copy.deepcopy(obs), copy.deepcopy(legal)))

    def test_bad_targets_and_fit_arguments_before_torch_import(self):
        obs = observation()
        for legal in ([], (), None, [Action("bad", (0,), 2)],
                      [Action("single", (107,), 17)]):
            with self.subTest(legal=legal), self.assertRaises(ValueError):
                teacher_targets(obs, legal)
        with self.assertRaises(TypeError):
            teacher_targets(object(), [Action("single", (0,), 2)])
        with self.assertRaises(TypeError):
            teacher_targets(obs, [object()])
        with mock.patch.object(GreedyAgent, "score", return_value=float("nan")):
            with self.assertRaises(ValueError):
                teacher_targets(obs, [Action("single", (0,), 2)])
        for chunk in (0, -1, True, 1.5, None):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                fit_batch(None, None, [], chunk)
        with self.assertRaises(ValueError):
            fit_batch(None, None, [])


@unittest.skipUnless(os.environ.get("RUN_P3F_CUDA") == "1", "CUDA diagnostic requires explicit opt-in")
class CUDAObjectiveTests(unittest.TestCase):
    def test_chunked_matches_independent_observation_mean_objective(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from guandan.learning.encoding import encode_action, encode_observation
        from guandan.learning.model import DMCNetwork
        requests = [
            (observation((0, 54)), [Action("single", (0,), 2),
                                     Action("single", (54,), 2), Action("pair", (0, 54), 2)]),
            (observation((1,)), [Action("single", (1,), 3)]),
        ]
        model = DMCNetwork().to(device="cuda", dtype=torch.float32)
        dense = DMCNetwork().to(device="cuda", dtype=torch.float32)
        dense.load_state_dict(copy.deepcopy(model.state_dict()))
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, capturable=True)
        dense_optimizer = torch.optim.Adam(dense.parameters(), lr=.001, capturable=True)
        dense_optimizer.zero_grad(set_to_none=True)
        observation_losses = []
        for obs, legal in requests:
            state = torch.tensor([encode_observation(obs)] * len(legal), device="cuda")
            action = torch.tensor([encode_action(value) for value in legal], device="cuda")
            raw = [GreedyAgent().score(obs, value) for value in legal]
            lo, hi = min(raw), max(raw)
            target = [0.0] * len(raw) if lo == hi else [1.6 * (value - lo) / (hi - lo) - .8 for value in raw]
            scores = dense(state, action)
            observation_losses.append((scores - torch.tensor(target, device="cuda")).square().mean())
        objective = torch.stack(observation_losses).mean()
        objective.backward()
        dense_gradients = [parameter.grad.detach().clone() for parameter in dense.parameters()]
        dense_optimizer.step()
        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with mock.patch("experiments.p3f_teacher.encode_observation", wraps=encode_observation) as encode:
                result = fit_batch(model, optimizer, requests, chunk_size=2)
                self.assertEqual(encode.call_count, len(requests))
            self.assertEqual(step.call_count, 1)
        self.assertAlmostEqual(result["loss"], objective.item(), places=6)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["scored_candidates"], 4)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["per_request_sizes"], [3, 1])
        for p, q, gradient in zip(model.parameters(), dense.parameters(), dense_gradients):
            torch.testing.assert_close(p.grad, gradient, atol=1e-6, rtol=1e-4)
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-4)
            self.assertTrue(torch.isfinite(p).all().item())
        self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before, model.parameters())))
        for key in ("forward_checks", "gradient_checks", "adam_tensor_checks", "parameter_checks"):
            self.assertGreater(result["device_proof"][key], 0)
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device.type, "cuda")
                    self.assertTrue(torch.isfinite(value).all().item())


if __name__ == "__main__":
    unittest.main()
