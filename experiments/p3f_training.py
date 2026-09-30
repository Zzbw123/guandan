"""Frozen two-phase P3f training jobs; importing this module starts no training."""
from __future__ import annotations

import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from unittest.mock import patch

from guandan.agents import GreedyAgent
from guandan.env import HandEnv
from experiments.p3f_teacher import fit_batch


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_json(value) + "\n")


def model_digest(model_or_state) -> str:
    """Hash sorted tensor names, dtype, shape and contiguous CPU NumPy bytes."""
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    digest = sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = _json([name, str(tensor.dtype), list(tensor.shape)]).encode("utf-8")
        data = tensor.numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def save_phase1(trainer, directory: Path, arm: str, seed: int, updates: int,
                hands: int = 200) -> dict:
    """Raw artifact, intentionally separate from the wave-boundary checkpoint."""
    import torch
    directory = Path(directory)
    payload = dict(model=_cpu(trainer.model.state_dict()),
                   optimizer=_cpu(trainer.optimizer.state_dict()),
                   phase1_updates=updates, config=dict(trainer.config), hands=hands,
                   arm=arm, seed=seed)
    buffer = BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    with (directory / "raw.pt").open("xb") as handle:
        handle.write(data)
    manifest = dict(sha256=sha256(data).hexdigest(), bytes=len(data),
                    model_sha256=model_digest(payload["model"]),
                    phase1_updates=updates, config=payload["config"], hands=hands,
                    arm=arm, seed=seed)
    _write(directory / "manifest.json", manifest)
    return manifest


def reset_phase2(directory: Path, config: dict):
    """Verify disk payload, construct fresh trainer, then load only model weights."""
    import torch
    from guandan_gpu.training import GPUTrainer
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data = (directory / "raw.pt").read_bytes()
    if len(data) != manifest["bytes"] or sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("phase1 file SHA/size mismatch")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    for key in ("config", "hands", "arm", "seed", "phase1_updates"):
        if payload[key] != manifest[key]:
            raise ValueError(f"phase1 manifest mismatch: {key}")
    if payload["config"] != config or payload["seed"] != config["seed"]:
        raise ValueError("phase1 configuration mismatch")
    if model_digest(payload["model"]) != manifest["model_sha256"]:
        raise ValueError("phase1 model hash mismatch")
    trainer = GPUTrainer(config)
    trainer.model.load_state_dict(payload["model"], strict=True)
    initial_hash = model_digest(trainer.model)
    if initial_hash != manifest["model_sha256"] or trainer.optimizer.state:
        raise ValueError("phase2 model alignment / fresh optimizer failed")
    counters = dict(episodes=trainer.episodes, waves=trainer.waves, updates=trainer.updates)
    if any(counters.values()) or trainer.used_deal_seeds:
        raise ValueError("phase2 counters are not zero")
    reset = dict(phase1_file=str((directory / "raw.pt").resolve()),
                 phase1_file_sha256=manifest["sha256"],
                 phase1_model_sha256=manifest["model_sha256"],
                 phase2_initial_model_sha256=initial_hash,
                 optimizer_empty=True, initial_counters=counters, used_deal_seeds=[],
                 policy_rng_repr_sha256=sha256(repr(trainer.rng.getstate()).encode()).hexdigest(),
                 torch_rng_sha256=sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                 cuda_rng_sha256=[sha256(state.cpu().numpy().tobytes()).hexdigest()
                                  for state in torch.cuda.get_rng_state_all()])
    return trainer, reset


def _deal(index):
    return (100000 + index, 2 + index % 13, index % 4)


def _replay_row(env, deal, candidates, samples):
    replay = env.serialize_replay()
    verified = HandEnv.replay(replay)
    if not verified.state.terminal or verified.state_digest() != env.state_digest():
        raise ValueError("hand replay verification failed")
    return dict(seed=deal[0], level=deal[1], starting_player=deal[2],
                steps=len(replay["steps"]), samples=samples,
                legal_candidates=candidates, terminal_digest=env.state_digest(),
                replay_verified=True, replay=replay)


def _dmc_phase(trainer, start, stop, waves_path, replays_path):
    """Instrument environment IO only; preserve GPUTrainer's learning algorithm."""
    rows = []
    started = time.perf_counter()
    with gzip.open(replays_path, "xt", encoding="utf-8", newline="\n") as replay_file:
        class RecordedHandEnv(HandEnv):
            def reset(self, seed, rules_config=None, initial_level=2, starting_player=0):
                obs = super().reset(seed, rules_config=rules_config,
                                    initial_level=initial_level, starting_player=starting_player)
                self.recorded_deal = (seed, initial_level, starting_player)
                self.recorded_candidates = 0
                self.recorded_observations = 0
                return obs

            def legal_actions(self, player_id):
                legal = super().legal_actions(player_id)
                if hasattr(self, "recorded_deal"):
                    self.recorded_candidates += len(legal)
                    self.recorded_observations += 1
                return legal

            def serialize_replay(self):
                replay = super().serialize_replay()
                verified = HandEnv.replay(replay)
                if not verified.state.terminal or verified.state_digest() != self.state_digest():
                    raise ValueError("DMC replay verification failed")
                row = dict(seed=self.recorded_deal[0], level=self.recorded_deal[1],
                           starting_player=self.recorded_deal[2],
                           steps=len(replay["steps"]), samples=self.recorded_observations,
                           legal_candidates=self.recorded_candidates,
                           terminal_digest=self.state_digest(), replay_verified=True)
                replay_file.write(_json({**row, "replay": replay}) + "\n")
                replay_file.flush()
                rows.append(row)
                return replay

        with waves_path.open("x", encoding="utf-8", newline="\n") as waves_file:
            with patch("guandan_gpu.training.HandEnv", RecordedHandEnv):
                for index in range(start, stop, 4):
                    wave_started = time.perf_counter()
                    result = trainer.train_wave([_deal(i) for i in range(index, index + 4)])
                    result["elapsed_seconds"] = time.perf_counter() - wave_started
                    waves_file.write(_json(result) + "\n")
                    waves_file.flush()
                    if trainer.waves % 50 == 0:
                        print(_json(dict(progress="DMC", hands=trainer.episodes,
                                         wave=trainer.waves, updates=trainer.updates)), flush=True)
    return dict(hands=len(rows), samples=sum(row["samples"] for row in rows),
                legal_candidates=sum(row["legal_candidates"] for row in rows),
                updates=trainer.updates, seconds=time.perf_counter() - started,
                hand_rows=rows)


def _teacher_phase(trainer, directory):
    pending, hand_rows = [], []
    observations = candidates = updates = fitted = 0
    teacher = GreedyAgent()
    started = time.perf_counter()
    with (directory / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as batch_file:
        def fit_pending():
            nonlocal updates, fitted
            batch_started = time.perf_counter()
            result = fit_batch(trainer.model, trainer.optimizer, pending, chunk_size=1024)
            updates += 1
            row = dict(result, update=updates, observation_start=fitted,
                       observation_stop=fitted + len(pending),
                       elapsed_seconds=time.perf_counter() - batch_started)
            batch_file.write(_json(row) + "\n")
            batch_file.flush()
            fitted += len(pending)
            pending.clear()

        with gzip.open(directory / "replays.jsonl.gz", "xt", encoding="utf-8", newline="\n") as replay_file:
            for index in range(200):
                deal = _deal(index)
                env = HandEnv()
                obs = env.reset(deal[0], initial_level=deal[1], starting_player=deal[2])
                hand_candidates = steps = 0
                while True:
                    if steps >= 1000:
                        raise RuntimeError("teacher hand exceeded 1000 steps")
                    legal = env.legal_actions(obs.player_id)
                    action = teacher.act(obs, legal)
                    pending.append((obs, legal))
                    observations += 1
                    candidates += len(legal)
                    hand_candidates += len(legal)
                    steps += 1
                    if len(pending) == 64:
                        fit_pending()
                    result = env.step(obs.player_id, action, state_version=obs.state_version)
                    if result.terminal:
                        break
                    obs = env.observe(result.next_player)
                row = _replay_row(env, deal, hand_candidates, steps)
                replay_file.write(_json(row) + "\n")
                replay_file.flush()
                hand_rows.append({key: value for key, value in row.items() if key != "replay"})
                if (index + 1) % 50 == 0:
                    print(_json(dict(progress="teacher", hands=index + 1,
                                     observations=observations, updates=updates)), flush=True)
        if pending:
            fit_pending()
    if fitted != observations:
        raise ValueError("teacher FIFO observations not fitted exactly once")
    return dict(hands=200, samples=observations, legal_candidates=candidates,
                updates=updates, seconds=time.perf_counter() - started, hand_rows=hand_rows)


def train(root: Path, job: str) -> dict:
    """Run one explicitly selected 800-hand job. Caller owns diagnostic gating."""
    allowed = {f"{arm}-{seed}" for arm in ("control", "teacher")
               for seed in (314380, 314381, 314382)}
    if job not in allowed:
        raise ValueError("unknown P3f job")
    from guandan_gpu.training import GPUTrainer
    from guandan_gpu.checkpoint import runtime, save_checkpoint
    arm, seed_text = job.split("-")
    seed = int(seed_text)
    config = dict(seed=seed, epsilon=.1, lr=.001, batch_size=256,
                  chunk_size=1024, num_envs=4)
    output = Path(root).resolve() / "training" / job
    output.mkdir(parents=True, exist_ok=False)
    phase1_dir = output / "phase1"
    phase1_dir.mkdir()
    started = time.perf_counter()
    trainer = GPUTrainer(config)
    phase1_initial_model_sha256 = model_digest(trainer.model)
    phase1 = (_teacher_phase(trainer, phase1_dir) if arm == "teacher" else
              _dmc_phase(trainer, 0, 200, phase1_dir / "waves.jsonl",
                         phase1_dir / "replays.jsonl.gz"))
    manifest = save_phase1(trainer, phase1_dir, arm, seed, phase1["updates"])
    del trainer
    trainer, reset = reset_phase2(phase1_dir, config)
    _write(output / "reset.json", reset)
    phase2 = _dmc_phase(trainer, 200, 800, output / "phase2-waves.jsonl",
                        output / "phase2-replays.jsonl.gz")
    if trainer.episodes != 600 or trainer.waves != 150:
        raise ValueError("phase2 episode/wave accounting failed")
    final_manifest = save_checkpoint(trainer, output / "final")
    report = dict(status="PASS", runtime=runtime(), job=job, arm=arm, seed=seed, config=config,
                  phase1_initial_model_sha256=phase1_initial_model_sha256,
                  phase1=phase1, phase2=phase2, total_hands=800,
                  total_samples=phase1["samples"] + phase2["samples"],
                  total_legal_candidates=phase1["legal_candidates"] + phase2["legal_candidates"],
                  total_updates=phase1["updates"] + phase2["updates"],
                  phase1_manifest=manifest, final_manifest=final_manifest,
                  seconds=time.perf_counter() - started)
    _write(output / "report.json", report)
    return report
