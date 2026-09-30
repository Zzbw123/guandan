"""Predetermined paired deals and seat-preserving symmetry schedule."""

from __future__ import annotations

from hashlib import sha256

from guandan.env import HandEnv
from guandan.evaluation.contracts import Matchup, Trial


def build_trials(matchup: Matchup, seeds: list[int] | tuple[int, ...]) -> list[Trial]:
    """Expand one matchup into all predeclared symmetries for each deal.

    The caller owns the 27-matchup matrix. ``deal_index`` is the position in
    the fixed seed list, so a level occurs ten times in the full 130-deal run.
    """
    if type(matchup) is not Matchup:
        raise TypeError("matchup must be Matchup")
    if not isinstance(seeds, (list, tuple)):
        raise TypeError("seeds must be a list or tuple")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate deal seeds")

    trials = []
    for deal_index, deal_seed in enumerate(seeds):
        level = 2 + deal_index % 13
        for rotation in range(4):
            for swap in range(2):
                for flip in range(1 if matchup.focal == matchup.teammate else 2):
                    focal_seat = (rotation + swap + 2 * flip) % 4
                    teammate_seat = (focal_seat + 2) % 4
                    policies = [matchup.opponent] * 4
                    policies[focal_seat] = matchup.focal
                    policies[teammate_seat] = matchup.teammate
                    trials.append(Trial(
                        matchup.key, deal_index, deal_seed, level,
                        rotation, swap, flip, focal_seat, tuple(policies), 0,
                    ))
    return trials


def deal_hands(trial: Trial) -> tuple[tuple[int, ...], ...]:
    """Rotate complete original hands without changing clockwise order."""
    if type(trial) is not Trial:
        raise TypeError("trial must be Trial")
    env = HandEnv()
    env.reset(trial.deal_seed, initial_level=trial.level)
    original = env.state.initial_hands
    return tuple(original[(seat - trial.rotation) % 4] for seat in range(4))


def policy_rng_seed(policy_seed: int, trial: Trial, seat: int) -> int:
    """Derive a stable agent seed without using a deal seed or hidden cards."""
    if type(policy_seed) is not int:
        raise TypeError("policy_seed must be an integer")
    if type(trial) is not Trial:
        raise TypeError("trial must be Trial")
    if type(seat) is not int or seat not in range(4):
        raise ValueError("seat must be in 0..3")
    original_seat = (seat - trial.rotation) % 4
    payload = f"{policy_seed}:{trial.deal_index}:{trial.rotation}:{original_seat}"
    return int.from_bytes(sha256(payload.encode("ascii")).digest(), "big")
