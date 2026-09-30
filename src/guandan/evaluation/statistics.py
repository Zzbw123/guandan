"""Deal-clustered summaries for the frozen P2 baseline evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite, sqrt
from random import Random
from statistics import median

from .contracts import Trial


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number, not bool")
    return float(value)


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[min(lower + 1, len(sorted_values) - 1)] * fraction


def cluster_bootstrap(
    values: list[float], strata: list[int], repetitions: int, seed: int,
    confidence: float = 0.95,
) -> dict:
    """Resample original deals within strata, preserving every stratum's size."""
    if len(values) != len(strata) or not values:
        raise ValueError("values and strata must have the same nonzero length")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(confidence) not in (float, int) or not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    grouped: dict[int, list[float]] = defaultdict(list)
    for index, (value, stratum) in enumerate(zip(values, strata, strict=True)):
        if type(stratum) is not int:
            raise ValueError(f"strata[{index}] must be an integer")
        grouped[stratum].append(_number(value, f"values[{index}]"))
    estimate = sum(sum(group) for group in grouped.values()) / len(values)
    degenerate = all(len(set(group)) == 1 for group in grouped.values())
    result = {
        "estimate": estimate,
        "ci95": None,
        "method": "stratified_deal_cluster_bootstrap_percentile_linear",
        "degenerate": degenerate,
        "reason": None,
    }
    if any(len(group) < 2 for group in grouped.values()):
        result["reason"] = "fewer_than_two_clusters_in_stratum"
        return result
    rng = Random(seed)
    groups = [grouped[key] for key in sorted(grouped)]
    samples: list[float] = []
    denominator = len(values)
    for _ in range(repetitions):
        total = 0.0
        for group in groups:
            total += sum(group[rng.randrange(len(group))] for _ in group)
        samples.append(total / denominator)
    samples.sort()
    tail = (1 - confidence) / 2
    result["ci95"] = [_percentile(samples, tail), _percentile(samples, 1 - tail)]
    if samples[0] == samples[-1]:
        result["degenerate"] = True
        result["reason"] = "bootstrap_distribution_degenerate"
    return result


_TRIAL_FIELDS = (
    "matchup_id", "deal_index", "deal_seed", "level", "rotation", "swap",
    "flip", "focal_seat", "focal_team", "policies", "starting_player",
)
_ROW_FIELDS = frozenset((
    "trial_id", *_TRIAL_FIELDS, "status", "win", "team_reward",
    "focal_level_gain", "opponent_level_gain", "finish_order",
    "terminal_digest", "steps", "illegal_actions", "timeouts", "error",
))
_METRICS = ("win_rate", "team_reward", "focal_level_gain", "opponent_level_gain")


def _require_int(value: object, name: str, lower: int, upper: int | None = None) -> int:
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _validate_result(row: dict, trial: Trial) -> dict[str, float]:
    if row.get("status") != "ok" or type(row.get("status")) is not str:
        raise ValueError(f"trial {trial.trial_id}: unsuccessful status")
    if type(row.get("win")) is not int or row["win"] not in (0, 1):
        raise ValueError(f"trial {trial.trial_id}: win must be 0 or 1")
    win = bool(row["win"])
    reward = row.get("team_reward")
    if type(reward) is not int or reward != (1 if win else -1):
        raise ValueError(f"trial {trial.trial_id}: invalid team_reward")
    focal_gain = _require_int(row.get("focal_level_gain"), "focal_level_gain", 0, 3)
    opponent_gain = _require_int(row.get("opponent_level_gain"), "opponent_level_gain", 0, 3)
    if (focal_gain > 0) != win or (opponent_gain > 0) == win or focal_gain + opponent_gain not in (1, 2, 3):
        raise ValueError(f"trial {trial.trial_id}: invalid level gains")
    order = row.get("finish_order")
    if type(order) is not list or len(order) not in (2, 4):
        raise ValueError(f"trial {trial.trial_id}: invalid finish_order")
    if any(type(seat) is not int or seat not in range(4) for seat in order) or len(set(order)) != len(order):
        raise ValueError(f"trial {trial.trial_id}: invalid finish_order seats")
    winning_team = trial.focal_team if win else 1 - trial.focal_team
    if order[0] % 2 != winning_team:
        raise ValueError(f"trial {trial.trial_id}: finish_order winner mismatch")
    if len(order) == 2:
        if order[1] % 2 != winning_team or focal_gain + opponent_gain != 3:
            raise ValueError(f"trial {trial.trial_id}: invalid double-down settlement")
    else:
        if order[1] % 2 == winning_team:
            raise ValueError(f"trial {trial.trial_id}: double-down order must have two seats")
        teammate = (order[0] + 2) % 4
        expected_gain = {2: 3, 3: 2, 4: 1}[order.index(teammate) + 1]
        if focal_gain + opponent_gain != expected_gain:
            raise ValueError(f"trial {trial.trial_id}: finish_order level gain mismatch")
    _require_int(row.get("steps"), "steps", 1, 1000)
    if row.get("illegal_actions") != 0 or type(row.get("illegal_actions")) is not int:
        raise ValueError(f"trial {trial.trial_id}: illegal actions")
    if row.get("timeouts") != 0 or type(row.get("timeouts")) is not int:
        raise ValueError(f"trial {trial.trial_id}: timeouts")
    if row.get("error") is not None:
        raise ValueError(f"trial {trial.trial_id}: error is present")
    digest = row.get("terminal_digest")
    if type(digest) is not str or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"trial {trial.trial_id}: invalid terminal_digest")
    return {
        "win_rate": float(win), "team_reward": float(reward),
        "focal_level_gain": float(focal_gain), "opponent_level_gain": float(opponent_gain),
    }


def summarize_matchup(
    rows: list[dict], expected_trials: list[Trial], bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict:
    """Reject incomplete scenarios and summarize paired rotations by original deal."""
    if type(rows) is not list:
        raise ValueError("rows must be a list")
    if type(expected_trials) is not list or not expected_trials or any(type(trial) is not Trial for trial in expected_trials):
        raise ValueError("expected_trials must be a nonempty sequence of Trials")
    expected: dict[str, Trial] = {}
    matchups: set[str] = set()
    deal_trials: dict[int, list[Trial]] = defaultdict(list)
    for trial in expected_trials:
        if trial.trial_id in expected:
            raise ValueError(f"duplicate expected trial_id: {trial.trial_id}")
        expected[trial.trial_id] = trial
        matchups.add(trial.matchup_id)
        deal_trials[trial.deal_index].append(trial)
    if len(matchups) != 1:
        raise ValueError("expected_trials span multiple matchups")
    if len({trials[0].deal_seed for trials in deal_trials.values()}) != len(deal_trials):
        raise ValueError("deal_seed reused across deal clusters")
    for deal_index, trials in deal_trials.items():
        first = trials[0]
        if any((t.deal_seed, t.level) != (first.deal_seed, first.level) for t in trials):
            raise ValueError(f"deal {deal_index}: inconsistent seed or level")
        if len(trials) != (8 if first.policies[first.focal_seat] == first.policies[(first.focal_seat + 2) % 4] else 16):
            raise ValueError(f"deal {deal_index}: incomplete trial group")
    observed: dict[str, dict] = {}
    for row in rows:
        if type(row) is not dict or type(row.get("trial_id")) is not str:
            raise ValueError("each row needs a string trial_id")
        trial_id = row["trial_id"]
        if trial_id not in expected:
            raise ValueError(f"unexpected trial_id: {trial_id}")
        if trial_id in observed:
            raise ValueError(f"duplicate trial_id: {trial_id}")
        observed[trial_id] = row
    if set(observed) != set(expected):
        raise ValueError(f"missing trial_ids: {sorted(set(expected) - set(observed))[:3]}")
    deal_values: dict[int, list[dict[str, float]]] = defaultdict(list)
    for trial_id, trial in expected.items():
        row = observed[trial_id]
        if set(row) != _ROW_FIELDS:
            missing = sorted(_ROW_FIELDS - set(row))
            extra = sorted(set(row) - _ROW_FIELDS)
            raise ValueError(f"trial {trial_id}: row fields mismatch; missing={missing}, extra={extra}")
        for field in _TRIAL_FIELDS:
            actual = row.get(field)
            wanted = getattr(trial, field)
            if field == "policies":
                wanted = list(wanted)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"trial {trial_id}: {field} does not match schedule")
            if field == "policies" and any(type(policy) is not str for policy in actual):
                raise ValueError(f"trial {trial_id}: policy names must be strings")
        deal_values[trial.deal_index].append(_validate_result(row, trial))
    clusters = []
    for deal_index in sorted(deal_values):
        values = deal_values[deal_index]
        trial = deal_trials[deal_index][0]
        cluster = {
            "deal_index": deal_index, "deal_seed": trial.deal_seed,
            "level": trial.level, "games": len(values),
        }
        cluster.update({metric: sum(v[metric] for v in values) / len(values) for metric in _METRICS})
        clusters.append(cluster)
    strata = [cluster["level"] for cluster in clusters]
    metrics = {
        metric: cluster_bootstrap(
            [cluster[metric] for cluster in clusters], strata, bootstrap_replicates,
            bootstrap_seed,
        ) for metric in _METRICS
    }
    win_values = [cluster["win_rate"] for cluster in clusters]
    win_mean = sum(win_values) / len(win_values)
    win_sd = sqrt(sum((value - win_mean) ** 2 for value in win_values) / (len(win_values) - 1)) if len(win_values) > 1 else None
    return {
        "matchup_id": next(iter(matchups)),
        "games": len(rows),
        "independent_deals": len(clusters),
        "strata_counts": dict(sorted(Counter(strata).items())),
        "metrics": metrics,
        "effect_vs_50pp": 100 * (metrics["win_rate"]["estimate"] - 0.5),
        "clusters": clusters,
        "quality": {
            "cluster_win_rate": {
                "mean": win_mean, "sd": win_sd, "median": median(win_values),
                "min": min(win_values), "max": max(win_values),
                "distribution": dict(sorted(Counter(win_values).items())),
            },
            "degenerate": {metric: metrics[metric]["degenerate"] for metric in _METRICS},
        },
    }
