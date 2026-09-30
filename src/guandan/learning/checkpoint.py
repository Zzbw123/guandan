"""Controller-owned CPU episode-boundary checkpoints, never pickle objects."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform

import torch

from guandan.types import ACTION_VERSION, OBSERVATION_VERSION, RULES_VERSION
from guandan.learning.encoding import FEATURE_VERSION, STATE_DIM, ACTION_DIM
from guandan.learning.model import NETWORK_VERSION
from guandan.learning.training import Trainer

CHECKPOINT_VERSION = "gd-dmc-checkpoint-v1"


def versions():
    return dict(checkpoint=CHECKPOINT_VERSION, rules=RULES_VERSION,
                observation=OBSERVATION_VERSION, action=ACTION_VERSION,
                feature=FEATURE_VERSION, network=NETWORK_VERSION,
                state_dim=STATE_DIM, action_dim=ACTION_DIM)


def runtime():
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                device="cpu", threads=1, deterministic=True)


def learning_sources():
    root = Path(__file__).parents[1]
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def _finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all().item():
            raise ValueError("Nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for x in value.values():
            _finite(x)
    elif isinstance(value, (list, tuple)):
        for x in value:
            _finite(x)


def save_checkpoint(trainer, directory):
    if not trainer.ready_for_checkpoint:
        raise ValueError("Checkpoint requires a successfully completed episode boundary")
    if torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Checkpoint requires the frozen deterministic single-thread runtime")
    payload = dict(versions=versions(), runtime=runtime(), sources=learning_sources(),
                   boundary="episode_boundary", config=trainer.config,
                   episodes=trainer.episodes, updates=trainer.updates,
                   used_deal_seeds=trainer.used_deal_seeds,
                   model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                   policy_rng=trainer.rng.getstate(), torch_rng=torch.get_rng_state())
    _finite(payload)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    buffer = BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    with (target / "checkpoint.pt").open("xb") as f:
        f.write(data)
    receipt = dict(sha256=sha256(data).hexdigest(), bytes=len(data),
                   versions=versions(), runtime=runtime(), sources=payload["sources"],
                   episodes=trainer.episodes, updates=trainer.updates,
                   boundary="episode_boundary", config=trainer.config,
                   used_deal_seeds=trainer.used_deal_seeds)
    with (target / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
    return receipt


def load_checkpoint(directory, expected_sha256=None):
    target = Path(directory)
    receipt = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    data = (target / "checkpoint.pt").read_bytes()
    digest = sha256(data).hexdigest()
    if digest != receipt["sha256"] or len(data) != receipt["bytes"]:
        raise ValueError("Checkpoint hash/size mismatch")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Checkpoint does not match pinned candidate hash")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    for key, expected in (("versions", versions()), ("runtime", runtime()),
                          ("sources", learning_sources()), ("boundary", "episode_boundary")):
        if payload.get(key) != expected or receipt.get(key) != expected:
            raise ValueError(f"Checkpoint incompatible {key}")
    for key in ("config", "episodes", "updates", "used_deal_seeds"):
        if receipt.get(key) != payload.get(key):
            raise ValueError(f"Checkpoint manifest disagrees on {key}")
    if any(type(payload[k]) is not int or payload[k] < 0 for k in ("episodes", "updates")):
        raise ValueError("Invalid checkpoint counters")
    seeds = payload["used_deal_seeds"]
    if (type(seeds) is not list or len(seeds) != payload["episodes"]
            or any(type(s) is not int or not 100000 <= s < 110000 for s in seeds)):
        raise ValueError("Invalid training seed history")
    _finite(payload)
    trainer = Trainer(payload["config"])
    trainer.model.load_state_dict(payload["model"], strict=True)
    optimizer = payload["optimizer"]
    if optimizer["param_groups"] != trainer.optimizer.state_dict()["param_groups"]:
        raise ValueError("Optimizer configuration mismatch")
    parameters = list(trainer.model.parameters())
    expected_ids = set(range(len(parameters))) if payload["updates"] else set()
    if set(optimizer["state"]) != expected_ids:
        raise ValueError("Optimizer state is incomplete")
    if (payload["episodes"] == 0) != (payload["updates"] == 0) or payload["updates"] < payload["episodes"]:
        raise ValueError("Optimizer counters disagree with completed episodes")
    for i, state in optimizer["state"].items():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or state["step"].numel() != 1 or state["step"].item() != payload["updates"]
                or any(state[k].shape != parameters[i].shape or state[k].dtype != parameters[i].dtype
                       for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("Optimizer tensor shape or step mismatch")
    trainer.optimizer.load_state_dict(payload["optimizer"])
    # Adam defaults restored by load_state_dict must agree with the frozen config.
    if (len(trainer.optimizer.param_groups) != 1
            or trainer.optimizer.param_groups[0]["lr"] != trainer.config["lr"]):
        raise ValueError("Optimizer configuration mismatch")
    trainer.rng.setstate(payload["policy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    trainer.episodes, trainer.updates = payload["episodes"], payload["updates"]
    trainer.used_deal_seeds = list(seeds)
    return trainer
