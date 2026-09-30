"""Run one paired trial while keeping research state outside policy inputs."""

from __future__ import annotations

from math import isfinite
from time import perf_counter_ns

from guandan.agents.baselines import make_agent
from guandan.env import HandEnv
from guandan.evaluation.contracts import Trial
from guandan.evaluation.schedule import deal_hands, policy_rng_seed
from guandan.types import Action


class IllegalActionError(ValueError):
    """A policy returned a value outside the enumerated legal actions."""


class DecisionTimeoutError(TimeoutError):
    """A returned policy call exceeded the fixed soft time limit."""


class StepLimitExceeded(RuntimeError):
    """A hand did not settle within its action budget."""


def run_trial(
    trial: Trial, policy_seed: int, max_steps: int = 1000,
    decision_timeout_ms: float = 1000.0,
) -> tuple[dict, dict]:
    """Return a JSON-ready trial row and unaggregated per-turn measurements.

    The decision threshold is checked after ``act`` returns. It cannot stop a
    policy which never returns; callers must use a separate process watchdog
    when evaluating untrusted or potentially hanging policies.
    """
    if type(trial) is not Trial:
        raise TypeError("trial must be Trial")
    if type(policy_seed) is not int:
        raise TypeError("policy_seed must be an integer")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if (type(decision_timeout_ms) not in (int, float)
            or not isfinite(decision_timeout_ms) or decision_timeout_ms <= 0):
        raise ValueError("decision_timeout_ms must be finite and positive")

    row = {
        "trial_id": trial.trial_id,
        "matchup_id": trial.matchup_id,
        "deal_index": trial.deal_index,
        "deal_seed": trial.deal_seed,
        "level": trial.level,
        "rotation": trial.rotation,
        "swap": trial.swap,
        "flip": trial.flip,
        "focal_seat": trial.focal_seat,
        "focal_team": trial.focal_team,
        "policies": list(trial.policies),
        "starting_player": trial.starting_player,
        "status": "error",
        "win": None,
        "team_reward": None,
        "focal_level_gain": None,
        "opponent_level_gain": None,
        "finish_order": None,
        "terminal_digest": None,
        "steps": 0,
        "illegal_actions": 0,
        "timeouts": 0,
        "error": None,
    }
    measurements = {"enumeration_ms": [], "decision_ms": [], "candidate_counts": []}

    try:
        env = HandEnv.from_hands(deal_hands(trial), trial.level,
                                 trial.starting_player)
        agents = tuple(make_agent(name, policy_rng_seed(policy_seed, trial, seat))
                       for seat, name in enumerate(trial.policies))

        while not env.state.terminal:
            if row["steps"] >= max_steps:
                raise StepLimitExceeded(f"hand exceeded {max_steps} steps")
            seat = env.state.current_player
            observation = env.observe(seat)

            started = perf_counter_ns()
            try:
                legal = tuple(env.legal_actions(seat))
            finally:
                measurements["enumeration_ms"].append(
                    (perf_counter_ns() - started) / 1_000_000)
            measurements["candidate_counts"].append(len(legal))

            started = perf_counter_ns()
            try:
                action = agents[seat].act(observation, legal)
            finally:
                elapsed_ms = (perf_counter_ns() - started) / 1_000_000
                measurements["decision_ms"].append(elapsed_ms)
                row["timeouts"] += int(elapsed_ms > decision_timeout_ms)

            # A soft deadline also records illegal output if both occur.
            illegal = type(action) is not Action or action not in legal
            timed_out = elapsed_ms > decision_timeout_ms
            row["illegal_actions"] += int(illegal)
            if timed_out:
                raise DecisionTimeoutError(
                    f"seat {seat} decision took {elapsed_ms:.3f} ms "
                    f"(limit {decision_timeout_ms:g} ms)")
            if illegal:
                raise IllegalActionError(f"seat {seat} returned a nonlegal action")

            try:
                env.step(seat, action, state_version=observation.state_version)
            except (TypeError, ValueError):
                row["illegal_actions"] += 1
                raise
            row["steps"] += 1

        settlement = env.state.settlement
        if settlement is None:
            raise RuntimeError("terminal hand has no settlement")
        focal_won = settlement.winner_team == trial.focal_team
        row.update({
            "status": "ok",
            "win": int(focal_won),
            "team_reward": settlement.team_rewards[trial.focal_team],
            "focal_level_gain": settlement.level_gain if focal_won else 0,
            "opponent_level_gain": 0 if focal_won else settlement.level_gain,
            "finish_order": list(settlement.finish_order),
            "terminal_digest": env.state_digest(),
        })
    except Exception as exc:
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}

    return row, measurements
