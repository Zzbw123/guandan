"""Frozen P3g teacher-fit diagnostic. Importing starts neither training nor scoring.

Selected states are purposive diagnostic states, not an unbiased population.
Research replay and deal information stay on the reconstruction side; every
model receives only PlayerObservation and the complete legal Action list.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/evaluations/p3f-teacher-v1"
SEEDS = (314380, 314381, 314382)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=True)


def canonical_digest(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest(path):
    value = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized_targets(teacher_scores):
    values = [float(v) for v in teacher_scores]
    check(values and all(math.isfinite(v) for v in values),
          "teacher scores must be nonempty and finite")
    lo, hi = min(values), max(values)
    return ([0.] * len(values) if lo == hi else
            [1.6 * ((v - lo) / (hi - lo)) - .8 for v in values])


def summarize(scores, teacher_scores, finish_indexes):
    """Pure full-candidate metrics; exact ties choose the first model maximum."""
    values, teacher = [float(v) for v in scores], [float(v) for v in teacher_scores]
    check(values and len(values) == len(teacher), "nonempty matching score arrays required")
    check(all(math.isfinite(v) for v in values + teacher), "nonfinite scores")
    finish = list(finish_indexes)
    check(all(type(i) is int and 0 <= i < len(values) for i in finish)
          and len(set(finish)) == len(finish), "invalid finish indexes")
    index = max(range(len(values)), key=values.__getitem__)
    target = normalized_targets(teacher)
    lo, hi = min(teacher), max(teacher)
    return dict(candidate_count=len(values), argmax_index=index,
                teacher_top1=teacher[index] == hi,
                normalized_regret=(0. if lo == hi else (hi - teacher[index]) / (hi - lo)),
                candidate_mse=sum((v - t) ** 2 for v, t in zip(values, target)) / len(values),
                min=min(values), max=max(values), spread=max(values) - min(values),
                saturated_count=sum(abs(v) >= .99 for v in values),
                finish_opportunity=bool(finish),
                finish_miss=(index not in finish if finish else None))


class Aggregate:
    """Observation means, candidate-weight saturation, opportunity-weight misses."""
    def __init__(self):
        self.count = 0
        self.sums = Counter()

    def add(self, metrics):
        self.count += 1
        for key in ("teacher_top1", "normalized_regret", "candidate_mse", "spread",
                    "candidate_count", "saturated_count", "finish_opportunity"):
            self.sums[key] += metrics[key]
        self.sums["finish_miss"] += int(metrics["finish_miss"] is True)

    def result(self):
        mean = lambda key: self.sums[key] / self.count if self.count else None
        return dict(observations=self.count,
                    mean_teacher_top1=mean("teacher_top1"),
                    mean_normalized_regret=mean("normalized_regret"),
                    mean_candidate_mse=mean("candidate_mse"), mean_spread=mean("spread"),
                    candidates=self.sums["candidate_count"],
                    saturated_candidates=self.sums["saturated_count"],
                    saturation_rate=(self.sums["saturated_count"] / self.sums["candidate_count"]
                                     if self.sums["candidate_count"] else None),
                    finish_opportunities=self.sums["finish_opportunity"],
                    finish_misses=self.sums["finish_miss"],
                    finish_miss_rate=(self.sums["finish_miss"] / self.sums["finish_opportunity"]
                                      if self.sums["finish_opportunity"] else None))


def source_specs(root):
    """Fixed corpus order: common demonstrations, final development, phase1 development."""
    return [("training", BASE / "training/teacher-314380/phase1/replays.jsonl.gz", 200)] + [
        (f"final-{seed}", BASE / f"evaluations/teacher-{seed}/replays.jsonl.gz", 208)
        for seed in SEEDS] + [
        (f"phase1-{seed}", Path(root) / f"evaluations/teacher-phase1-{seed}/replays.jsonl.gz", 208)
        for seed in SEEDS]


def iter_selected(source, path, expected_games, counts):
    """Re-enumerate every decision; yield all finishes and first four other choices/game."""
    from guandan.env import HandEnv
    from guandan.types import Action
    from guandan.evaluation.schedule import deal_hands
    from experiments.p3f_protocol import schedule
    trials = {trial.trial_id: trial for trial in schedule(False)}
    seen = set()
    per_game = []
    for game_index, item in enumerate(lines(path)):
        replay = item["replay"]
        if source == "training":
            game_id = str(item["seed"])
            check((item["seed"], item["level"], item["starting_player"]) ==
                  (100000 + game_index, 2 + game_index % 13, game_index % 4),
                  "training schedule mismatch")
            env = HandEnv()
            env.reset(item["seed"], initial_level=item["level"], starting_player=item["starting_player"])
            focal = None
        else:
            game_id = item["trial_id"]
            check(game_id in trials, "unknown development trial")
            trial = trials[game_id]
            env = HandEnv.from_hands(deal_hands(trial), trial.level, trial.starting_player)
            focal = trial.focal_team
        check(game_id not in seen, "duplicate source game")
        seen.add(game_id)
        check([list(h) for h in env.state.initial_hands] == replay["initial_hands"] and
              env.state_digest() == replay["initial_digest"], "scheduled initial replay mismatch")
        verified = HandEnv.replay(replay)
        check(verified.state.terminal and verified.state_digest() == replay["final_digest"],
              "source replay is not verified terminal")
        if "terminal_digest" in item:
            check(item["terminal_digest"] == replay["final_digest"], "row final digest mismatch")
        selected, other_selected, opportunities = 0, 0, 0
        for step_index, step in enumerate(replay["steps"]):
            player = step["player"]
            obs, legal = env.observe(player), env.legal_actions(player)
            chosen = Action.from_dict(step["action"])
            check(chosen in legal, "recorded action missing from complete candidates")
            counts["all_decisions_reenumerated"] += 1
            if focal is None or player % 2 == focal:
                counts["eligible_decisions"] += 1
                finish = [i for i, a in enumerate(legal)
                          if a.kind != "pass" and len(a.cards) == len(obs.hand)]
                opportunities += bool(finish)
                counts["finish_opportunities"] += bool(finish)
                other = not finish and len(legal) > 1
                counts["other_choice_opportunities"] += bool(other)
                if finish or (other and other_selected < 4):
                    selected += 1
                    other_selected += bool(other)
                    counts["selected_states"] += 1
                    counts["selected_candidates"] += len(legal)
                    counts["finish_states_selected"] += bool(finish)
                    counts["other_choices_selected"] += bool(other)
                    yield (dict(source=source, game_id=game_id, step=step_index,
                                player=player, hand_size=len(obs.hand), finish_indexes=finish,
                                selection_reason="finish" if finish else "first4_other_choice",
                                candidate_sha256=canonical_digest([a.to_dict() for a in legal]),
                                observation_sha256=canonical_digest(asdict(obs))), obs, legal)
            env.step(player, chosen, state_version=step["state_version"])
            check(env.state_digest() == step["digest"], "replay step digest mismatch")
        check(env.state_digest() == replay["final_digest"], "replay final digest mismatch")
        counts["games"] += 1
        per_game.append(dict(game_id=game_id, selected_states=selected,
                             finish_opportunities=opportunities, other_choices_selected=other_selected))
    check(len(seen) == expected_games, "source game count mismatch")
    if source != "training":
        check(seen == set(trials), "source trial coverage mismatch")
    counts["per_game"] = per_game


def verify_hashes(mapping):
    check(isinstance(mapping, dict) and mapping, "empty hash map")
    for name, expected in mapping.items():
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT), "hash path escapes repository")
        check(digest(path) == expected, f"pinned SHA mismatch: {name}")


def verify_evaluation_files(folder, report):
    """Verify generated evidence's internal binding and every declared artifact."""
    check(report["binding_sha256"] == digest(folder / "binding.json"),
          "evaluation binding SHA mismatch")
    hashes = report["artifact_sha256"]
    check(isinstance(hashes, dict) and "replays.jsonl.gz" in hashes,
          "evaluation artifacts must bind replay")
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        check(path.is_relative_to(folder.resolve()), "evaluation artifact escapes folder")
        check(digest(path) == expected, "evaluation artifact SHA mismatch: " + name)


def verify_phase1_evaluation_binding(folder, report, prereg, prereg_hash, seed):
    """Bind the new development corpus to exactly the frozen raw model."""
    binding = read(folder / "binding.json")
    checkpoint = BASE / f"training/teacher-{seed}/phase1"
    raw_pin = prereg["input_sha256"][(checkpoint / "raw.pt").relative_to(ROOT).as_posix()]
    manifest_pin = prereg["input_sha256"][(checkpoint / "manifest.json").relative_to(ROOT).as_posix()]
    manifest = read(checkpoint / "manifest.json")
    check(digest(checkpoint / "raw.pt") == manifest["sha256"] == raw_pin and
          digest(checkpoint / "manifest.json") == manifest_pin, "phase1 input pins mismatch")
    check(Path(binding["checkpoint"]).resolve() == checkpoint.resolve(),
          "phase1 evaluation checkpoint selection mismatch")
    check(binding["manifest_sha256"] == manifest_pin,
          "phase1 evaluation manifest pin mismatch")
    for evidence in (binding, report):
        check(evidence["candidate_sha256"] == raw_pin and
              evidence["preregistration_sha256"] == prereg_hash and
              evidence["phase1_model_sha256"] == manifest["model_sha256"],
              "phase1 evaluation candidate/preregistration/model binding mismatch")
        check(evidence["job"] == f"teacher-phase1-{seed}" and evidence["seed"] == seed and
              evidence["arm"] == "teacher" and evidence["games"] == 208,
              "phase1 evaluation job identity mismatch")


def execute(root):
    root = Path(root).resolve()
    prereg = read(root / "preregistration.json")
    for key in ("source_sha256", "input_sha256"):
        verify_hashes(prereg[key])
    for source, path, _ in source_specs(root):
        if not source.startswith("phase1-"):
            check(path.relative_to(ROOT).as_posix() in prereg["input_sha256"],
                  "fixed corpus replay absent from preregistration")
    prereg_hash = digest(root / "preregistration.json")
    new_hashes = {}
    for seed in SEEDS:
        folder = root / f"evaluations/teacher-phase1-{seed}"
        report = read(folder / "report.json")
        check(report["status"] == "PASS" and report["games"] == 208,
              "phase1 development report must pass with 208 games")
        verify_evaluation_files(folder, report)
        verify_phase1_evaluation_binding(folder, report, prereg, prereg_hash, seed)
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                new_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    verify_hashes(new_hashes)
    output = root / "scores"
    output.mkdir(exist_ok=False)
    write(output / "input-freeze.json", dict(preregistration_sha256=prereg_hash,
                                             new_input_sha256=new_hashes))
    from experiments.p3g_guard import load_phase1
    from experiments.p3f_protocol import config
    from experiments.p3f_training import model_digest
    from guandan_gpu.training import GPUTrainer, score_many
    from guandan_gpu.checkpoint import load_checkpoint
    from guandan.agents import GreedyAgent
    import torch
    check(torch.cuda.is_available(), "CUDA unavailable")
    started = time.perf_counter()
    models, metadata = {}, {}
    for phase in ("phase1", "final"):
        for seed in SEEDS:
            folder = BASE / f"training/teacher-{seed}/{phase}"
            filename = "raw.pt" if phase == "phase1" else "checkpoint.pt"
            sha = prereg["input_sha256"][(folder / filename).relative_to(ROOT).as_posix()]
            if phase == "phase1":
                manifest_sha = prereg["input_sha256"][(folder / "manifest.json").relative_to(ROOT).as_posix()]
                trainer, receipt = load_phase1(folder, sha, seed, expected_manifest_sha256=manifest_sha)
            else:
                runtime = GPUTrainer(config(seed))
                del runtime
                trainer = load_checkpoint(folder, expected_sha256=sha)
                del trainer.optimizer
            model = trainer.model
            model.eval()
            label = f"{phase}-{seed}"
            models[label] = model
            metadata[label] = dict(file=(folder / filename).relative_to(ROOT).as_posix(),
                                   file_sha256=sha, model_sha256=model_digest(model),
                                   device=str(next(model.parameters()).device))
            check(next(model.parameters()).device.type == "cuda", "model must be CUDA")
            del trainer
    teacher = GreedyAgent()
    sources, aggregates = {}, {}
    with gzip.open(output / "scores.jsonl.gz", "xt", encoding="utf-8", newline="\n") as handle:
        for source, path, games in source_specs(root):
            counts = Counter()
            pairs = {label: {group: Aggregate() for group in ("all", "multi_choice")}
                     for label in models}
            for row, obs, legal in iter_selected(source, path, games, counts):
                teacher_scores = [float(teacher.score(obs, action)) for action in legal]
                row.update(teacher_scores=teacher_scores, targets=normalized_targets(teacher_scores), models={})
                for label, model in models.items():
                    scores = score_many(model, [(obs, legal)], 1024)[0].tolist()
                    metrics = summarize(scores, teacher_scores, row["finish_indexes"])
                    row["models"][label] = dict(scores=scores, metrics=metrics)
                    pairs[label]["all"].add(metrics)
                    if len(legal) > 1:
                        pairs[label]["multi_choice"].add(metrics)
                handle.write(canonical_json(row) + "\n")
            sources[source] = dict(counts)
            aggregates[source] = {label: {group: acc.result() for group, acc in pair.items()}
                                  for label, pair in pairs.items()}
            print(canonical_json(dict(source=source, games=counts["games"],
                                      selected_states=counts["selected_states"])), flush=True)
    for label, model in models.items():
        check(model_digest(model) == metadata[label]["model_sha256"], "model changed during scoring")
    for key in ("source_sha256", "input_sha256"):
        verify_hashes(prereg[key])
    verify_hashes(new_hashes)
    check(digest(root / "preregistration.json") == prereg_hash, "preregistration changed")
    report = dict(status="PASS", sources=sources, models=metadata, aggregates=aggregates,
                  scores_sha256=digest(output / "scores.jsonl.gz"),
                  input_freeze_sha256=digest(output / "input-freeze.json"),
                  preregistration_sha256=prereg_hash, chunk_size=1024,
                  scope="purposive diagnostic states; not an unbiased population or strength evaluation",
                  model_promoted=False, training_executed=False, validation_read=False,
                  seconds=time.perf_counter() - started,
                  runtime=dict(executable=sys.executable, torch=torch.__version__,
                               cuda=torch.version.cuda, device=torch.cuda.get_device_name(0)))
    write(output / "report.json", report)
    return report


def main():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    execute(parser.parse_args().root)


if __name__ == "__main__":
    main()
