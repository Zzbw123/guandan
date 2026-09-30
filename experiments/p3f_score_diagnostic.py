"""Read-only CUDA score diagnostic of six frozen P3e development jobs.

Research replays remain environment-side. Scorers receive only observations and
the complete legal candidate lists. No training or new deals are executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/evaluations/p3e-scale-v2"
JOBS = tuple(f"{seed}-{budget}" for seed in (314370, 314371, 314372)
             for budget in (400, 1600))
NUMERIC_TOLERANCE = 1e-6
CHOICE_GAP_LIMIT = 2e-6


def summarize_scores(scores, legal, hand_size, chosen_index, teacher_index):
    """Pure score summary; preserve first-index argmax and physical candidates."""
    values = [float(value) for value in scores]
    if not values or len(values) != len(legal):
        raise ValueError("scores must cover the nonempty complete candidate list")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")
    if type(hand_size) is not int or hand_size <= 0:
        raise ValueError("hand_size must be a positive integer")
    for index in (chosen_index, teacher_index):
        if type(index) is not int or not 0 <= index < len(values):
            raise ValueError("chosen and teacher indices must be in candidate range")
    finishing = [i for i, action in enumerate(legal)
                 if action.kind != "pass" and len(action.cards) == hand_size]
    best = max(range(len(values)), key=values.__getitem__)
    top = values[best]
    return dict(candidate_count=len(values), selected_index=chosen_index,
                teacher_index=teacher_index, argmax_index=best,
                finish_indexes=finishing, chosen_finishes=chosen_index in finishing,
                min=min(values), max=top, spread=top-min(values),
                saturated_count=sum(abs(value) >= .99 for value in values),
                chosen_score_gap=top-values[chosen_index],
                finish_score_gap=(top-max(values[i] for i in finishing)
                                  if finishing else None),
                near_top_indexes=[i for i, value in enumerate(values)
                                  if top-value <= NUMERIC_TOLERANCE])


def compare_scores(reference, alternative):
    """Numerical fidelity and rank sensitivity are separate diagnostics."""
    a, b = [float(v) for v in reference], [float(v) for v in alternative]
    if not a or len(a) != len(b) or not all(math.isfinite(v) for v in a+b):
        raise ValueError("comparison needs matching nonempty finite score arrays")
    first = max(range(len(a)), key=a.__getitem__)
    second = max(range(len(b)), key=b.__getitem__)
    error = max(abs(x-y) for x, y in zip(a, b))
    gap = max(a[first]-a[second], b[second]-b[first])
    changed = first != second
    return dict(max_error=error, reference_argmax=first, alternative_argmax=second,
                argmax_changed=changed, choice_gap=gap,
                rank_sensitive=changed and gap <= CHOICE_GAP_LIMIT,
                near_tie=changed and gap <= NUMERIC_TOLERANCE,
                failed=error > NUMERIC_TOLERANCE or
                       (changed and gap > CHOICE_GAP_LIMIT))


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def indexed_metadata(path):
    rows = {}
    for row in lines(path):
        tid = row["trial_id"]
        check(tid not in rows, f"duplicate metadata trial: {path.name}/{tid}")
        rows[tid] = row
    return rows


def check(condition, message):
    if not condition:
        raise ValueError(message)


def candidate_digest(legal):
    raw = json.dumps([a.to_dict() for a in legal], sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def snapshot():
    sources = sorted((ROOT/"src").rglob("*.py")) + [Path(__file__),
        ROOT/"tests/test_p3f_score_diagnostic.py",
        ROOT/"experiments/p3e_protocol.py", ROOT/"experiments/p3e_common.py",
        ROOT/"experiments/p3e_metrics.py", ROOT/"scripts/run_gpu.ps1",
        ROOT/"docs/P3F_DIAGNOSTIC_PROTOCOL.md"]
    inputs = [BASE/"preregistration.json"]
    bindings = {}
    for job in JOBS:
        path = BASE/"evaluations"/job
        binding = read(path/"binding.json")
        seed, budget = job.split("-")
        checkpoint = BASE/"training"/seed/f"wave-{int(budget)//4}"
        check((ROOT/binding["checkpoint"]).resolve() == checkpoint.resolve(),
              f"{job}: fixed development checkpoint")
        manifest = read(checkpoint/"manifest.json")
        check(binding["job"] == job and binding["candidate_sha256"] == manifest["sha256"]
              == digest(checkpoint/"checkpoint.pt"), f"{job}: pinned candidate hash")
        check(binding["preregistration_sha256"] == digest(BASE/"preregistration.json"),
              f"{job}: preregistration binding")
        bindings[job] = binding
        inputs.extend(path/name for name in ("binding.json", "report.json", "results.jsonl",
                      "measurements.jsonl.gz", "replays.jsonl.gz"))
        inputs.extend(checkpoint/name for name in ("checkpoint.pt", "manifest.json"))
    hashes = lambda paths: {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    return dict(source_hashes=hashes(sources), input_hashes=hashes(inputs), bindings=bindings)


def direct_scores(model, obs, legal, torch, encode_observation, encode_action):
    device = next(model.parameters()).device
    check(device.type == "cuda", "direct scoring must use CUDA")
    state = encode_observation(obs)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(legal), 1024):
            candidates = legal[start:start+1024]
            states = torch.tensor([state]*len(candidates), dtype=torch.float32, device=device)
            actions = torch.tensor([encode_action(a) for a in candidates],
                                   dtype=torch.float32, device=device)
            values = model.forward(states, actions)
            check(values.device.type == "cuda", "direct forward output must be CUDA")
            pieces.append(values)
        return torch.cat(pieces).cpu().tolist()


def execute(output):
    # Path creation and freezing precede importing Torch or scoring any state.
    output.mkdir(parents=True, exist_ok=False)
    frozen = snapshot()
    frozen.update(version="gd-p3f-score-diagnostic-v1", utc=datetime.now(timezone.utc).isoformat(),
        jobs=list(JOBS), scope="existing P3e development research replays only",
        selection="all focal immediate-finish states plus first 128 other focal choice states per job, stored replay and step order",
        production_chunk=1024, alternative_chunk=37, direct_chunk=1024,
        numeric_tolerance=NUMERIC_TOLERANCE, choice_gap_limit=CHOICE_GAP_LIMIT,
        boundary="model receives only PlayerObservation and complete legal Action list",
        model_promoted=False, new_hands_executed=False, validation_read=False)
    write(output/"specification.json", frozen)
    sys.path.insert(0, str(ROOT/"src"))
    sys.path.insert(0, str(ROOT/"experiments"))
    import torch
    from guandan.env import HandEnv
    from guandan.types import Action
    from guandan.agents.baselines import GreedyAgent
    from guandan.evaluation.schedule import deal_hands
    from guandan.learning.encoding import encode_observation, encode_action
    from guandan_gpu.checkpoint import load_checkpoint
    from guandan_gpu.training import GPUTrainer, score_many
    from p3e_protocol import schedule
    check(sys.version_info[:2] == (3, 14), "diagnostic requires original Python314 runtime")
    check(torch.cuda.is_available(), "CUDA unavailable")
    started = time.perf_counter()
    totals, job_reports = Counter(), {}
    failures = []
    teacher = GreedyAgent()
    trials = {t.trial_id: t for t in schedule()}
    with (output/"scores.jsonl").open("x", encoding="utf-8") as target:
        for job in JOBS:
            path = BASE/"evaluations"/job
            binding = frozen["bindings"][job]
            checkpoint = ROOT/binding["checkpoint"]
            # Establish the frozen deterministic runtime before the loader's
            # runtime gate; loader then independently constructs and restores.
            trainer = GPUTrainer(read(checkpoint/"manifest.json")["config"])
            trainer = load_checkpoint(checkpoint, expected_sha256=binding["candidate_sha256"])
            trainer.model.eval()
            results = indexed_metadata(path/"results.jsonl")
            measurements = indexed_metadata(path/"measurements.jsonl.gz")
            report = read(path/"report.json")
            check(report["status"] == "PASS" and report["candidate_sha256"] == binding["candidate_sha256"],
                  f"{job}: report binding")
            check(set(results) == set(measurements) == set(trials), f"{job}: metadata coverage")
            counts, seen = Counter(), set()
            for item in lines(path/"replays.jsonl.gz"):
                tid, replay = item["trial_id"], item["replay"]
                check(tid in trials and tid not in seen, f"{job}: replay coverage")
                seen.add(tid)
                trial, result, measurement = trials[tid], results[tid], measurements[tid]
                verified = HandEnv.replay(replay)
                settlement = verified.state.settlement
                check(verified.state.terminal and settlement is not None and
                      verified.state_digest() == result["terminal_digest"] and
                      len(replay["steps"]) == result["steps"] and
                      list(settlement.finish_order) == result["finish_order"] and
                      int(settlement.winner_team == trial.focal_team) == result["win"] and
                      settlement.team_rewards[trial.focal_team] == result["team_reward"],
                      f"{job}/{tid}: replay result metadata")
                check(result["deal_seed"] == trial.deal_seed and result["level"] == trial.level and
                      result["focal_team"] == trial.focal_team and result["status"] == "ok",
                      f"{job}/{tid}: scheduled result metadata")
                n = result["steps"]
                check(measurement["replay_verified"] and
                      all(len(measurement[k]) == n for k in ("candidate_counts", "enumeration_ms", "decision_ms")),
                      f"{job}/{tid}: measurement coverage")
                neural = {v["step"]: v for v in measurement["neural"]}
                check(len(neural) == len(measurement["neural"]) and set(neural) == {i for i,s in enumerate(replay["steps"])
                      if s["player"] % 2 == trial.focal_team}, f"{job}/{tid}: neural coverage")
                env = HandEnv.from_hands(deal_hands(trial), trial.level, trial.starting_player)
                check([list(h) for h in env.state.initial_hands] == replay["initial_hands"] and
                      env.state_digest() == replay["initial_digest"], f"{job}/{tid}: scheduled initial state")
                counts["replays_verified"] += 1
                for step_index, step in enumerate(replay["steps"]):
                    seat = step["player"]
                    obs, legal = env.observe(seat), env.legal_actions(seat)
                    chosen = Action.from_dict(step["action"])
                    check(chosen in legal and len(legal) == measurement["candidate_counts"][step_index],
                          f"{job}/{tid}/{step_index}: recorded action/candidate coverage")
                    if seat % 2 == trial.focal_team:
                        counts["focal_decisions"] += 1
                        finish = any(a.kind != "pass" and len(a.cards) == len(obs.hand) for a in legal)
                        counts["finish_opportunities"] += int(finish)
                        counts["other_choice_opportunities"] += int(not finish and len(legal)>1)
                        other = not finish and len(legal)>1 and counts["other_choices_selected"]<128
                        if finish or other:
                            counts["finish_states_selected"] += int(finish)
                            counts["other_choices_selected"] += int(other)
                            counts["states_scored"] += 1
                            check(neural[step_index]["scored_candidates"] == len(legal) and
                                  neural[step_index]["device"]["type"] == "cuda", "source CUDA candidate coverage")
                            scores = score_many(trainer.model, [(obs, legal)], 1024)[0].tolist()
                            small = score_many(trainer.model, [(obs, legal)], 37)[0].tolist()
                            direct = direct_scores(trainer.model, obs, legal, torch, encode_observation, encode_action)
                            summary = summarize_scores(scores, legal, len(obs.hand), legal.index(chosen),
                                                       legal.index(teacher.act(obs, legal)))
                            chunk_cmp, forward_cmp = compare_scores(scores, small), compare_scores(scores, direct)
                            mismatch = summary["selected_index"] != summary["argmax_index"]
                            tolerance_case = mismatch and summary["chosen_score_gap"] <= CHOICE_GAP_LIMIT
                            collision = bool(not summary["chosen_finishes"] and any(
                                encode_action(chosen) == encode_action(legal[i]) for i in summary["finish_indexes"]))
                            errors = []
                            if chunk_cmp["failed"] or forward_cmp["failed"]: errors.append("numeric_or_rank_fidelity")
                            if mismatch and not tolerance_case: errors.append("recorded_action_mismatch")
                            if collision: errors.append("chosen_nonfinish_finish_feature_collision")
                            row = dict(job=job, trial_id=tid, step=step_index, player=seat,
                                source_candidate_sha256=binding["candidate_sha256"],
                                candidate_sha256=candidate_digest(legal), selection_reason="finish" if finish else "first128_other_choice",
                                **summary, scores=scores, scores_chunk37=small, scores_direct_forward=direct,
                                max_chunk_error=chunk_cmp["max_error"], max_forward_error=forward_cmp["max_error"],
                                chunk_comparison=chunk_cmp, forward_comparison=forward_cmp,
                                recorded_action_exact_mismatch=mismatch, recorded_action_tolerance_case=tolerance_case,
                                chosen_finish_encoding_collision=collision, scoring_device=str(next(trainer.model.parameters()).device),
                                status="FAIL" if errors else "PASS", failures=errors)
                            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                            for name, value in (("exact_action_mismatches", mismatch), ("action_tolerance_cases", tolerance_case),
                                  ("feature_collisions", collision), ("chunk_rank_sensitive", chunk_cmp["rank_sensitive"]),
                                  ("forward_rank_sensitive", forward_cmp["rank_sensitive"]), ("failed_states", bool(errors))):
                                counts[name] += int(value)
                            counts["candidates_scored"] += len(legal)
                    env.step(seat, chosen, state_version=step["state_version"])
                    check(env.state_digest() == step["digest"], "reconstructed replay step digest")
                check(env.state_digest() == replay["final_digest"], "reconstructed replay final digest")
            check(seen == set(trials) and len(seen) == report["games"] == binding["games"], f"{job}: full replay coverage")
            check(counts["finish_states_selected"] == counts["finish_opportunities"] and
                  counts["other_choices_selected"] == min(128, counts["other_choice_opportunities"]),
                  f"{job}: deterministic selected coverage")
            job_reports[job] = dict(counts)
            totals.update(counts)
            if counts["failed_states"]: failures.append(f"{job}: failed states")
            print(json.dumps(dict(job=job, counts=dict(counts))), flush=True)
            del trainer
    for group in ("source_hashes", "input_hashes"):
        for name, expected in frozen[group].items():
            check(digest(ROOT/name) == expected, f"{group}: changed during diagnostic: {name}")
    runtime = dict(executable=sys.executable, python=platform.python_version(), torch=torch.__version__,
                   cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                   seconds=time.perf_counter()-started, utc=datetime.now(timezone.utc).isoformat())
    write(output/"runtime.json", runtime)
    result = dict(status="FAIL" if failures else "PASS", jobs=job_reports, totals=dict(totals), failures=failures,
                  specification_sha256=digest(output/"specification.json"), scores_sha256=digest(output/"scores.jsonl"),
                  runtime_sha256=digest(output/"runtime.json"), model_promoted=False,
                  scope="score fidelity and deterministic replay diagnostic; no strength claim")
    write(output/"report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new nonexistent output directory")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): parser.error("--output must not already exist")
    started = time.perf_counter()
    try:
        result = execute(output)
    except Exception as error:
        if output.is_dir():
            failure = dict(status="FAIL", error=type(error).__name__, message=str(error),
                           scope="diagnostic aborted; completed score rows remain available for inspection",
                           model_promoted=False)
            write(output/"failure.json", failure)
            if not (output/"runtime.json").exists():
                write(output/"runtime.json", dict(executable=sys.executable,
                      python=platform.python_version(), seconds=time.perf_counter()-started,
                      utc=datetime.now(timezone.utc).isoformat(), completed=False))
            if not (output/"report.json").exists():
                write(output/"report.json", failure)
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
