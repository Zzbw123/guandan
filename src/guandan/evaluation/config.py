"""Strict experiment configuration and disjoint split boundaries."""
from itertools import product
import math

from guandan.evaluation.contracts import EVALUATION_VERSION, POLICIES, Matchup
from guandan.types import RULES_VERSION


CONFIG_KEYS = frozenset(("evaluation_version", "rules_version", "split", "seed_start", "deal_count",
                         "policies", "primary_matchup", "primary_metric", "bootstrap_replicates",
                         "bootstrap_seed", "policy_seed", "workers", "max_steps", "decision_timeout_ms"))


def validate_config(config: dict, splits: dict) -> tuple[list[int], list[Matchup]]:
    if type(config) is not dict or set(config) != CONFIG_KEYS:
        raise ValueError("Evaluation config must contain exactly the documented fields")
    if config["evaluation_version"] != EVALUATION_VERSION or config["rules_version"] != RULES_VERSION:
        raise ValueError("Incompatible evaluation/rules version")
    if type(splits) is not dict or set(splits) != {
        "version", "development", "validation", "reserved_test", "stop_is_exclusive", "reserved_test_enabled"
    }:
        raise ValueError("Invalid seed split schema")
    if splits["version"] != "gd-seed-splits-v1" or splits["stop_is_exclusive"] is not True or splits["reserved_test_enabled"] is not False:
        raise ValueError("Reserved test must stay disabled; split stops must be exclusive")
    ranges = {}
    for name in ("development", "validation", "reserved_test"):
        interval = splits[name]
        if type(interval) is not dict or set(interval) != {"start", "stop"}:
            raise ValueError("Each split must have start and stop")
        start, stop = interval["start"], interval["stop"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
            raise ValueError("Invalid seed interval")
        ranges[name] = (start, stop)
    for a, b in (("development", "validation"), ("development", "reserved_test"), ("validation", "reserved_test")):
        if max(ranges[a][0], ranges[b][0]) < min(ranges[a][1], ranges[b][1]):
            raise ValueError("Seed splits must not overlap")
    if type(config["split"]) is not str or config["split"] not in ("development", "validation"):
        raise ValueError("P2 may run only development/validation; reserved test is sealed")
    for key in ("seed_start", "deal_count", "bootstrap_replicates", "bootstrap_seed", "policy_seed", "workers", "max_steps"):
        if type(config[key]) is not int or config[key] < (0 if key.endswith("seed") or key == "seed_start" else 1):
            raise ValueError(f"Invalid positive integer parameter: {key}")
    if config["deal_count"] < 26 or config["deal_count"] % 13:
        raise ValueError("Need equal coverage of 13 levels and at least two deals per level")
    if config["bootstrap_replicates"] < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    threshold = config["decision_timeout_ms"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("decision_timeout_ms must be positive and finite")
    if type(config["policies"]) is not list or config["policies"] != list(POLICIES):
        raise ValueError("P2 reports the complete frozen three-policy matrix")
    if config["primary_metric"] != "win_rate" or config["primary_matchup"] != "team|team|greedy":
        raise ValueError("The preregistered primary comparison cannot be changed")
    seeds = list(range(config["seed_start"], config["seed_start"] + config["deal_count"]))
    lower, upper = ranges[config["split"]]
    if seeds[0] < lower or seeds[-1] >= upper:
        raise ValueError("Requested seeds leave the selected split")
    return seeds, [Matchup(*names) for names in product(POLICIES, repeat=3)]
