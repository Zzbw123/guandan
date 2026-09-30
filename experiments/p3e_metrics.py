"""Observation-only diagnostic counts; these counts do not establish strength."""
from collections.abc import Iterable, Mapping

from guandan.types import Action, PlayerObservation


def decision_metrics(
    obs: PlayerObservation, legal: list[Action], chosen: Action
) -> dict[str, int]:
    """Count one active player's decision over the complete legal action list."""
    if obs.terminal:
        raise ValueError("Cannot count a terminal observation")
    if obs.player_id not in range(4) or obs.current_player != obs.player_id:
        raise ValueError("Observation must belong to the current player")
    if not legal:
        raise ValueError("Legal actions must be nonempty")
    if chosen not in legal:
        raise ValueError("Chosen action must belong to the legal action list")

    nonpass = [action for action in legal if action.kind != "pass"]
    optional = any(action.kind == "pass" for action in legal) and bool(nonpass)
    chosen_pass = chosen.kind == "pass"
    teammate = (
        obs.last_player is not None
        and obs.last_player % 2 == obs.player_id % 2
        and bool(nonpass)
        and obs.last_action is not None
    )
    finish = any(len(action.cards) == len(obs.hand) for action in nonpass)
    chosen_finish = not chosen_pass and len(chosen.cards) == len(obs.hand)
    endgame = len(obs.hand) <= 5
    return {
        "decisions": 1,
        "choices": int(len(legal) > 1),
        "forced_pass": int(not nonpass),
        "optional_pass_opportunities": int(optional),
        "optional_passes": int(optional and chosen_pass),
        "teammate_response_opportunities": int(teammate),
        "teammate_overtakes": int(teammate and not chosen_pass),
        "finish_opportunities": int(finish),
        "missed_finishes": int(finish and not chosen_finish),
        "endgame_decisions": int(endgame),
        "endgame_optional_pass_opportunities": int(endgame and optional),
        "endgame_optional_passes": int(endgame and optional and chosen_pass),
        "lead_decisions": int(obs.last_action is None),
        "cards_played": len(chosen.cards),
        "legal_candidates": len(legal),
        f"action_{chosen.kind}": 1,
    }


def sum_metrics(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Sum nonnegative integer counts, retaining every encountered metric key."""
    totals: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Metric {key!r} must be a nonnegative integer")
            totals[key] = totals.get(key, 0) + value
    return totals
