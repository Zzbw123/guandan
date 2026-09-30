"""Deterministic, audited greedy demonstrations for the P5f research loader.

Replay and seeds stay in corpus management. Consumers receive only observations
and the complete legal action lists reconstructed from the environment.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

from guandan.agents import GreedyAgent
from guandan.env.hand_env import HandEnv
from guandan.learning.encoding import FEATURE_VERSION
from guandan.types import ACTION_VERSION, OBSERVATION_VERSION, RULES_VERSION


CORPUS_VERSION = "gd-p5f-corpus-v1"
REPLAY_NAME = "replays.jsonl.gz"
MANIFEST_NAME = "manifest.json"
MAX_STEPS = 1000


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deal(value: object) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("deal must be (seed, level, starting_player)")
    seed, level, starting_player = value
    if type(seed) is not int or not 100000 <= seed <= 109999:
        raise ValueError("seed must be an integer in development range 100000..109999")
    if type(level) is not int or not 2 <= level <= 14:
        raise ValueError("level must be an integer in 2..14")
    if type(starting_player) is not int or not 0 <= starting_player <= 3:
        raise ValueError("starting_player must be an integer in 0..3")
    return seed, level, starting_player


def _deals(value: object) -> list[tuple[int, int, int]]:
    if type(value) is not list or not value:
        raise ValueError("deals must be a nonempty list")
    deals = [_deal(row) for row in value]
    seeds = [row[0] for row in deals]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate corpus seed")
    return deals


def _play(seed: int, level: int, starting_player: int,
          collect: bool = False, replay: dict | None = None) -> tuple[HandEnv, list, int]:
    env = HandEnv()
    env.reset(seed, initial_level=level, starting_player=starting_player)
    teacher = GreedyAgent()
    requests = []
    candidates = 0
    if replay is not None:
        if type(replay) is not dict or type(replay.get("steps")) is not list:
            raise ValueError("invalid saved replay")
        initial = {
            "visibility": "research_full", "rules_version": RULES_VERSION,
            "action_version": ACTION_VERSION,
            "initial_hands": [list(hand) for hand in env.state.initial_hands],
            "initial_level": level, "starting_player": starting_player,
            "research_fixture": False, "initial_digest": env.state_digest(),
        }
        if any(key not in replay or _canonical(replay[key]) != _canonical(value)
               for key, value in initial.items()):
            raise ValueError("saved replay initial state differs from reset")
    while not env.state.terminal:
        if env.state.version >= MAX_STEPS:
            raise ValueError("greedy hand exceeded 1000 steps")
        player = env.state.current_player
        obs = env.observe(player)
        legal = env.legal_actions(player)
        if not legal:
            raise ValueError("nonterminal state has no legal action")
        candidates += len(legal)
        if collect:
            requests.append((obs, legal))
        action = teacher.act(obs, legal)
        result = env.step(player, action, state_version=obs.state_version)
        if replay is not None:
            index = env.state.version - 1
            if index >= len(replay["steps"]):
                raise ValueError("saved replay ended before greedy trajectory")
            settlement = asdict(result.settlement) if result.settlement is not None else None
            actual = {
                "player": player, "action": asdict(action),
                "state_version": obs.state_version,
                "event": asdict(result.events[0]), "digest": env.state_digest(),
                "terminal": result.terminal, "settlement": settlement,
            }
            if _canonical(actual) != _canonical(replay["steps"][index]):
                raise ValueError("saved replay step differs from deterministic greedy trajectory")
    if replay is not None:
        settlement = asdict(env.state.settlement) if env.state.settlement is not None else None
        expected = {
            **initial, "final_digest": env.state_digest(),
            "final_settlement": settlement, "final_terminal": env.state.terminal,
        }
        if (len(replay["steps"]) != env.state.version
                or _canonical({key: value for key, value in replay.items() if key != "steps"})
                != _canonical(expected)):
            raise ValueError("saved replay final state differs from deterministic greedy trajectory")
    return env, requests, candidates


def build_corpus(directory: Path, deals: list[tuple[int, int, int]]) -> dict:
    """Create an exclusive corpus directory and return its frozen manifest."""
    validated = _deals(deals)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    observations = candidates = 0
    logical_hash = hashlib.sha256()
    replay_path = directory / REPLAY_NAME
    with replay_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for index, (seed, level, start) in enumerate(validated):
                env, _, count = _play(seed, level, start)
                replay = env.serialize_replay()
                checked = HandEnv.replay(replay)
                if not checked.state.terminal or checked.state_digest() != env.state_digest():
                    raise ValueError("generated replay failed verification")
                row = dict(index=index, seed=seed, level=level,
                           starting_player=start, steps=env.state.version,
                           legal_candidates=count, replay=replay)
                line = _canonical(row) + b"\n"
                zipped.write(line)
                logical_hash.update(line)
                observations += row["steps"]
                candidates += count
    manifest = {
        "version": CORPUS_VERSION,
        "rules_version": RULES_VERSION,
        "action_version": ACTION_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "feature_version": FEATURE_VERSION,
        "teacher_version": GreedyAgent.version,
        "deals": [list(deal) for deal in validated],
        "hands": len(validated),
        "observations": observations,
        "candidates": candidates,
        "replays_sha256": _file_digest(replay_path),
        "logical_sha256": logical_hash.hexdigest(),
    }
    (directory / MANIFEST_NAME).write_bytes(_canonical(manifest) + b"\n")
    return manifest


def _manifest(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = {"version", "rules_version", "action_version",
                "observation_version", "feature_version", "teacher_version",
                "deals", "hands", "observations", "candidates",
                "replays_sha256", "logical_sha256"}
    if type(manifest) is not dict or set(manifest) != expected:
        raise ValueError("invalid corpus manifest fields")
    for name, value in (("version", CORPUS_VERSION), ("rules_version", RULES_VERSION),
                        ("action_version", ACTION_VERSION),
                        ("observation_version", OBSERVATION_VERSION),
                        ("feature_version", FEATURE_VERSION),
                        ("teacher_version", GreedyAgent.version)):
        if manifest[name] != value:
            raise ValueError(f"unsupported corpus {name}")
    deals = _deals(manifest["deals"])
    if (any(type(row) is not list for row in manifest["deals"])
            or type(manifest["hands"]) is not int
            or manifest["hands"] != len(deals)
            or type(manifest["observations"]) is not int
            or not 0 < manifest["observations"] <= MAX_STEPS * len(deals)
            or type(manifest["candidates"]) is not int
            or manifest["candidates"] < manifest["observations"]):
        raise ValueError("invalid corpus manifest counts")
    for key in ("replays_sha256", "logical_sha256"):
        value = manifest[key]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {key}")
    return manifest


def _read_rows(directory: Path, manifest: dict) -> Iterator[dict]:
    logical_hash = hashlib.sha256()
    observations = candidates = hands = 0
    with gzip.open(directory / REPLAY_NAME, "rb") as source:
        for raw_line in source:
            if not raw_line.endswith(b"\n"):
                raise ValueError("unterminated replay row")
            row = json.loads(raw_line)
            if raw_line != _canonical(row) + b"\n":
                raise ValueError("noncanonical replay row")
            if type(row) is not dict or set(row) != {
                    "index", "seed", "level", "starting_player", "steps",
                    "legal_candidates", "replay"}:
                raise ValueError("invalid replay row fields")
            if hands >= manifest["hands"] or type(row["index"]) is not int or row["index"] != hands:
                raise ValueError("unexpected replay row index")
            seed, level, start = manifest["deals"][hands]
            if (type(row["seed"]) is not int or type(row["level"]) is not int
                    or type(row["starting_player"]) is not int
                    or (row["seed"], row["level"], row["starting_player"]) != (seed, level, start)):
                raise ValueError("replay row deal mismatch")
            if (type(row["steps"]) is not int or not 0 < row["steps"] <= MAX_STEPS
                    or type(row["legal_candidates"]) is not int
                    or row["legal_candidates"] < row["steps"]):
                raise ValueError("invalid replay row counts")
            logical_hash.update(raw_line)
            observations += row["steps"]
            candidates += row["legal_candidates"]
            hands += 1
            yield row
    if (hands != manifest["hands"] or observations != manifest["observations"]
            or candidates != manifest["candidates"]
            or logical_hash.hexdigest() != manifest["logical_sha256"]):
        raise ValueError("corpus tail count or logical hash mismatch")


def verify_corpus(directory: Path) -> dict:
    """Validate manifest, both file hashes, and every stored row/count.

    Full policy replay is done by ``iter_hands`` once during consumption.
    """
    directory = Path(directory)
    manifest = _manifest(directory)
    if _file_digest(directory / REPLAY_NAME) != manifest["replays_sha256"]:
        raise ValueError("replay file hash mismatch")
    for _ in _read_rows(directory, manifest):
        pass
    return manifest


def iter_hands(directory: Path) -> Iterator[tuple[dict, list]]:
    """Replay every saved hand under current rules and teacher, then yield requests."""
    directory = Path(directory)
    manifest = verify_corpus(directory)
    for row in _read_rows(directory, manifest):
        env, requests, candidates = _play(row["seed"], row["level"],
                                          row["starting_player"], collect=True,
                                          replay=row["replay"])
        if row["steps"] != env.state.version or row["legal_candidates"] != candidates:
            raise ValueError("saved replay counts differ from deterministic greedy trajectory")
        metadata = {key: value for key, value in row.items() if key != "replay"}
        yield metadata, requests


def iter_batches(directory: Path, batch_size: int = 64,
                 start_batch: int = 0) -> Iterator[list]:
    """Yield FIFO observation batches, including the final partial batch."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if type(start_batch) is not int or start_batch < 0:
        raise ValueError("start_batch must be nonnegative")
    pending = []
    batch_index = 0
    for _, requests in iter_hands(directory):
        for request in requests:
            pending.append(request)
            if len(pending) == batch_size:
                if batch_index >= start_batch:
                    yield pending
                pending = []
                batch_index += 1
    if pending:
        if batch_index >= start_batch:
            yield pending
        batch_index += 1
    if start_batch > batch_index:
        raise ValueError("start_batch exceeds corpus batch count")
