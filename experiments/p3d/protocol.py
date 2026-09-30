"""Controller-frozen P3d schedule and analysis, with no adaptive choices."""
from dataclasses import asdict
from guandan.evaluation.contracts import Trial
from guandan.evaluation.statistics import summarize_matchup

VERSION = "gd-p3d-validation-v1"
CONFIG = dict(seed=314360, epsilon=.1, lr=.001, batch_size=256, chunk_size=1024, num_envs=4)
TRAIN_SEEDS = list(range(104000,104400))
VALIDATION_SEEDS = list(range(201000,201065))
OPPONENTS = ("greedy", "random", "team")
POLICY_SEED = 271828
BOOTSTRAP_SEED = 314361
REPLICATES = 5000


def schedule():
    result = []
    for opponent in OPPONENTS:
        for i, seed in enumerate(VALIDATION_SEEDS):
            for rotation in range(4):
                for swap in range(2):
                    focal = (rotation+swap)%4
                    policies = tuple("dmc" if seat%2 == focal%2 else opponent for seat in range(4))
                    result.append(Trial(f"dmc|dmc|{opponent}", i, seed, 2+i%13,
                                        rotation, swap, 0, focal, policies, 0))
    return result


def specification():
    return dict(protocol=VERSION, config=CONFIG, training_seeds=TRAIN_SEEDS,
                waves=100, candidate_wave=100, validation_seeds=VALIDATION_SEEDS,
                trials=[{**asdict(t),"policies":list(t.policies)} for t in schedule()], policy_seed=POLICY_SEED,
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=REPLICATES,
                primary="greedy", point_threshold=.55, ci_lower_threshold=.5,
                startup_timeout_s=60., decision_timeout_s=2., max_steps=1000,
                training_process_timeout_s=1800, evaluation_process_timeout_s=1800,
                reserved_test_executed=False)


def summarize(rows):
    trials = schedule()
    if len(rows) != len(trials) or {r["trial_id"] for r in rows} != {t.trial_id for t in trials}:
        raise ValueError("incomplete/duplicate evaluation schedule")
    result = {}
    for opponent in OPPONENTS:
        key = f"dmc|dmc|{opponent}"
        result[opponent] = summarize_matchup([r for r in rows if r["matchup_id"] == key],
            [t for t in trials if t.matchup_id == key], REPLICATES, BOOTSTRAP_SEED)
    primary = result["greedy"]["metrics"]["win_rate"]
    passed = (primary["estimate"] >= .55 and primary["ci95"] is not None
              and primary["ci95"][0] > .5 and not primary["degenerate"])
    return dict(results=result, validation_gate="VALIDATION_GATE_PASSED" if passed else "NOT_ESTABLISHED",
                main_baseline="greedy", reserved_test_executed=False, model_promoted=False)
