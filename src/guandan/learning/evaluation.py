"""Fixed development-only integration schedule; no promotion statistics."""
from time import perf_counter

from guandan.agents.baselines import make_agent
from guandan.env import HandEnv
from guandan.evaluation.contracts import Trial
from guandan.evaluation.schedule import deal_hands, policy_rng_seed
from guandan.learning.model import DMCAgent


def smoke_schedule():
    trials = []
    for opponent in ("greedy", "random", "team"):
        for i, seed in enumerate((101000, 101001)):
            for rotation in range(4):
                for swap in range(2):
                    seat = (rotation + swap) % 4
                    policies = tuple("dmc" if p % 2 == seat % 2 else opponent for p in range(4))
                    trials.append(Trial(f"dmc|dmc|{opponent}", i, seed, 2+i,
                                        rotation, swap, 0, seat, policies))
    return trials


def evaluate_trial(model, trial, chunk_size=256):
    env = HandEnv.from_hands(deal_hands(trial), trial.level, trial.starting_player)
    actors = [DMCAgent(model, chunk_size=chunk_size) if p == "dmc"
              else make_agent(p, policy_rng_seed(271828, trial, seat))
              for seat, p in enumerate(trial.policies)]
    steps = 0
    decision_ms, candidate_counts = [], []
    while not env.state.terminal:
        if steps >= 1000:
            raise RuntimeError("Evaluation exceeded 1000-step guard")
        seat = env.state.current_player
        obs = env.observe(seat)
        legal = env.legal_actions(seat)
        started = perf_counter()
        action = actors[seat].act(obs, legal)
        elapsed = (perf_counter() - started) * 1000
        if trial.policies[seat] == "dmc":
            decision_ms.append(elapsed)
            candidate_counts.append(len(legal))
        if action not in legal:
            raise RuntimeError("Policy returned an action outside complete legal set")
        env.step(seat, action, state_version=obs.state_version)
        steps += 1
    replay = HandEnv.replay(env.serialize_replay())
    if replay.state_digest() != env.state_digest():
        raise RuntimeError("Evaluation replay mismatch")
    settlement = env.state.settlement
    return dict(trial_id=trial.trial_id, opponent=trial.policies[(trial.focal_seat+1)%4],
                deal_seed=trial.deal_seed, level=trial.level, rotation=trial.rotation,
                swap=trial.swap, focal_team=trial.focal_team,
                win=int(settlement.winner_team == trial.focal_team),
                reward=settlement.team_rewards[trial.focal_team], steps=steps,
                terminal_digest=env.state_digest(), replay_verified=True,
                decision_ms=decision_ms, candidate_counts=candidate_counts)


def summarize(rows):
    expected = {t.trial_id for t in smoke_schedule()}
    if len(rows) != len(expected) or {r["trial_id"] for r in rows} != expected:
        raise ValueError("Incomplete or duplicate evaluation trials")
    result = {}
    for opponent in ("greedy", "random", "team"):
        selected = [r for r in rows if r["opponent"] == opponent]
        groups = [sum(r["win"] for r in selected if r["deal_seed"] == seed) / 8
                  for seed in (101000, 101001)]
        result[opponent] = dict(games=len(selected), independent_deals=2,
                                group_win_rates=groups, win_rate=sum(groups)/2,
                                confidence_interval=None, conclusion="INTEGRATION_ONLY")
    return result
