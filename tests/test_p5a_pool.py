"""Role, isolation, rejection and exact-compatibility contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.p5a_training import GPUTrainer, lineup, weight_digest, _validated_config
from experiments.p5a_checkpoint import capture, save_checkpoint, load_checkpoint, _validate, META


def config(mode='mixed'):
    return dict(seed=314500, epsilon=.1, lr=.001, batch_size=256, chunk_size=1024, num_envs=4, mode=mode)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class ScheduleTests(unittest.TestCase):
    def test_all_roles_and_seat_balance(self):
        expected = [('current','greedy','current','greedy'), ('current','team','greedy','team'),
                    ('current','frozen','team','frozen'), ('current','greedy','frozen','team')]
        for block, row in enumerate(expected):
            for focal in range(4):
                actual = lineup('mixed', block*4+focal)
                self.assertEqual(tuple(actual[(focal+s)%4] for s in range(4)), row)
                self.assertEqual(actual, lineup('mixed', block*4+focal+16))
        self.assertEqual(lineup('selfplay', 5), ['current']*4)

    def test_invalid_schedule(self):
        for mode, n in [('x', 0), ('mixed', -1), ('mixed', True), ('mixed', 1.5)]:
            with self.assertRaises(ValueError): lineup(mode, n)

    def test_strict_config_and_json_roundtrip(self):
        self.assertEqual(_validated_config(json.loads(json.dumps(config()))), config())
        for c in [{**config(), 'mode': 'x'}, {**config(), 'extra': True}, {**config(), 'epsilon': float('nan')},
                  {k:v for k,v in config().items() if k != 'mode'}]:
            with self.assertRaises(ValueError): _validated_config(c)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GPUTests(unittest.TestCase):
    def test_selfplay_matches_old_two_waves(self):
        from guandan_gpu.training import GPUTrainer as Old
        from guandan_gpu.checkpoint import capture as old_capture
        cfg = config('selfplay')
        old = Old({k:v for k,v in cfg.items() if k != 'mode'})
        schedule = [[(108700+w*4+i, 2+i, i) for i in range(4)] for w in range(2)]
        expected = [old.train_wave(d) for d in schedule]
        snapshot = old_capture(old)
        new = GPUTrainer(cfg)
        actual = [new.train_wave(d) for d in schedule]
        other = capture(new)
        for key in ('model', 'optimizer', 'policy_rng', 'torch_rng', 'cuda_rng', 'episodes', 'updates', 'waves', 'used_deal_seeds'):
            self.assertTrue(equal(snapshot[key], other[key]), key)
        for x, y in zip(expected, actual):
            for key in ('samples','mean_loss','updates','scored_candidates','score_requests','score_batches'):
                self.assertEqual(x[key],y[key], key)

    def test_policy_boundary_and_learner_only_samples(self):
        from guandan.agents.baselines import GreedyAgent
        from guandan.types import PlayerObservation, Action
        actual = GreedyAgent.act
        seen = []
        def guarded(agent, obs, candidates):
            self.assertIs(type(obs), PlayerObservation)
            self.assertIs(type(candidates), tuple)
            self.assertTrue(all(type(x) is Action for x in candidates))
            self.assertFalse(hasattr(obs, 'seed'))
            seen.append(obs.player_id)
            return actual(agent, obs, candidates)
        t = GPUTrainer(config()); before = weight_digest(t.frozen)
        with patch.object(GreedyAgent, 'act', guarded):
            row = t.train_wave([(108720+i,2+i,i) for i in range(4)])
        self.assertTrue(seen)
        self.assertEqual(before, weight_digest(t.frozen))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in t.frozen.parameters()))
        self.assertEqual(row['samples'], sum(d['sample'] for h in row['hands'] for d in h['decisions']))
        self.assertLess(row['samples'], sum(h['steps'] for h in row['hands']))

    def test_failure_requires_reload(self):
        t = GPUTrainer(config())
        with patch('experiments.p5a_training.score_many', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                t.train_wave([(108740+i,2+i,i) for i in range(4)])
        with self.assertRaises(RuntimeError): t.train_wave([(108744+i,2+i,i) for i in range(4)])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): save_checkpoint(t, Path(d)/'bad')

    def test_illegal_baseline_rejected(self):
        from guandan.types import Action
        t = GPUTrainer(config())
        with patch.object(t.baselines['greedy'], 'act', return_value=Action('invalid')):
            with self.assertRaisesRegex(ValueError, 'illegal'):
                t.train_wave([(108760+i,2+i,i) for i in range(4)])
        self.assertFalse(t.ready_for_checkpoint)

    def test_frozen_mutation_rejected(self):
        t = GPUTrainer(config())
        with torch.no_grad(): next(t.frozen.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'pool changed'):
            t.train_wave([(108760+i,2+i,i) for i in range(4)])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): save_checkpoint(t, Path(d)/'bad')

    def test_checkpoint_roundtrip_no_overwrite_and_hash(self):
        t = GPUTrainer(config())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'good'; manifest = save_checkpoint(t,p)
            restored = load_checkpoint(p, manifest['sha256'])
            self.assertTrue(equal(capture(t), capture(restored)))
            with self.assertRaises(FileExistsError): save_checkpoint(t,p)
            with self.assertRaises(ValueError): load_checkpoint(p,'0'*64)

    def test_checkpoint_semantic_tampering(self):
        t = GPUTrainer(config()); original = capture(t)
        def frozen(p): p['frozen'][next(iter(p['frozen']))].view(-1)[0] += 1
        mutations = [lambda p:p['versions'].update(pool='bad'), lambda p:p.update(pool_sha256='0'*64), frozen,
                     lambda p:p['config'].update(mode='bad'), lambda p:p.update(episodes=4),
                     lambda p:p.update(used_deal_seeds=[9000000]), lambda p:p.update(sources={}),
                     lambda p:p['runtime'].update(torch='bad'), lambda p:p.update(policy_rng=('invalid',))]
        for mutation in mutations:
            p = copy.deepcopy(original); mutation(p)
            m = {k:p[k] for k in META}
            with self.assertRaises((ValueError, TypeError, IndexError)):
                _validate(p,m)

    def test_seed_boundary_rejects_validation_and_reserved(self):
        t = GPUTrainer(config())
        for seed in (200000,206000,9000000):
            with self.assertRaises(ValueError): t.train_wave([(seed+i,2+i,i) for i in range(4)])
        self.assertTrue(t.ready_for_checkpoint)

    def test_complete_high_branch_current_and_frozen(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from benchmark_learning_runtime import high_branch_fixture
        from guandan_gpu.training import score_many
        from guandan.learning.encoding import encode_action, encode_observation
        t = GPUTrainer(config())
        obs, actions = high_branch_fixture()
        self.assertEqual(len(actions), 8769)
        states = torch.tensor([encode_observation(obs)]*len(actions), device='cuda')
        encoded = torch.tensor([encode_action(a) for a in actions], device='cuda')
        for model in (t.model,t.frozen):
            with torch.no_grad(): expected = model(states,encoded).cpu()
            for chunk in (37,1024):
                actual = score_many(model,[(obs,actions)],chunk)[0]
                self.assertEqual(len(actual),8769)
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                self.assertEqual(int(actual.argmax()),int(expected.argmax()))

    def test_mid_update_failure_reload_repeats_exactly(self):
        t = GPUTrainer(config())
        deals = [(108780+i,2+i,i) for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            initial = Path(d)/'initial'; save_checkpoint(t,initial)
            reference = t.train_wave(deals); expected = capture(t)
            damaged = load_checkpoint(initial)
            original = damaged.optimizer.step
            def fail_after_update(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError('after Adam')
            with patch.object(damaged.optimizer,'step',fail_after_update):
                with self.assertRaisesRegex(RuntimeError,'after Adam'): damaged.train_wave(deals)
            self.assertFalse(damaged.ready_for_checkpoint)
            with self.assertRaises(ValueError): save_checkpoint(damaged,Path(d)/'failed')
            restored = load_checkpoint(initial)
            self.assertEqual(reference,restored.train_wave(deals))
            self.assertTrue(equal(expected,capture(restored)))

    def test_disk_rejection_preserves_global_rng(self):
        from hashlib import sha256
        t = GPUTrainer(config())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad'; save_checkpoint(t,p)
            payload=torch.load(p/'checkpoint.pt',weights_only=True)
            payload['frozen'][next(iter(payload['frozen']))].view(-1)[0] += 1
            torch.save(payload,p/'checkpoint.pt')
            manifest=json.loads((p/'manifest.json').read_text('utf-8'))
            data=(p/'checkpoint.pt').read_bytes()
            manifest.update(sha256=sha256(data).hexdigest(),bytes=len(data))
            (p/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            torch.manual_seed(888); torch.cuda.manual_seed_all(889)
            before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state_all())
            with self.assertRaisesRegex(ValueError,'frozen pool tensor'): load_checkpoint(p)
            self.assertTrue(equal(before,(torch.get_rng_state(),torch.cuda.get_rng_state_all())))

    def test_fresh_process_runner_resume(self):
        import subprocess
        root=Path(__file__).resolve().parents[1]
        t=GPUTrainer(config())
        schedule=[[(108800+n,2+n%13,n%4) for n in range(w*4,w*4+4)] for w in range(4)]
        for d in schedule[:2]: t.train_wave(d)
        with tempfile.TemporaryDirectory() as d:
            mid=Path(d)/'mid'; save_checkpoint(t,mid)
            expected_rows=[t.train_wave(x) for x in schedule[2:]]
            expected=capture(t); out=Path(d)/'resumed'
            r=subprocess.run([sys.executable,str(root/'scripts/p5a_run.py'),str(out),'--resume',str(mid)],
                             capture_output=True,timeout=120)
            self.assertEqual(r.returncode,0,r.stderr.decode('utf-8',errors='replace'))
            for w,row in zip((3,4),expected_rows):
                self.assertEqual(json.loads(json.dumps(row)),json.loads((out/f'wave-{w}.json').read_text('utf-8')))
            self.assertTrue(equal(expected,capture(load_checkpoint(out/'wave-4'))))


if __name__ == '__main__': unittest.main()
