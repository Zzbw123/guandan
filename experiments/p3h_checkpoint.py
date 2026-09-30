"""Atomic, strict GPU wave-boundary checkpoints; trusted local artifacts only."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import random
from uuid import uuid4

import torch

from guandan.learning.checkpoint import _finite
from guandan.learning.encoding import FEATURE_VERSION, STATE_DIM, ACTION_DIM
from guandan.learning.model import NETWORK_VERSION
from guandan.types import ACTION_VERSION, OBSERVATION_VERSION, RULES_VERSION
from experiments.p3h_training import GPUTrainer


def versions():
    return dict(checkpoint="gd-p3h-checkpoint-v1", training="gd-p3h-wave-v1",
                rules=RULES_VERSION, action=ACTION_VERSION, observation=OBSERVATION_VERSION,
                feature=FEATURE_VERSION, network=NETWORK_VERSION,
                state_dim=STATE_DIM, action_dim=ACTION_DIM)


def sources():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root/'src').rglob('*.py')) + sorted((root/'experiments').glob('p3h*.py'))
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}


def runtime():
    if not torch.cuda.is_available():
        raise ValueError("CUDA runtime required")
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
                device_count=torch.cuda.device_count(), platform=platform.platform(),
                threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def _runtime_gate():
    value = runtime()
    if (value["threads"] != 1 or not value["deterministic"] or value["tf32_matmul"]
            or value["tf32_cudnn"] or value["cublas"] != ":4096:8"
            or torch.cuda.current_device() != 0):
        raise ValueError("incompatible deterministic CUDA runtime")
    return value


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value


def capture(trainer):
    if not trainer.ready_for_checkpoint:
        raise ValueError("checkpoint requires successful wave boundary")
    return dict(versions=versions(), runtime=_runtime_gate(), sources=sources(),
                boundary="wave_boundary", config=dict(trainer.config),
                episodes=trainer.episodes, updates=trainer.updates, waves=trainer.waves,
                used_deal_seeds=list(trainer.used_deal_seeds),
                model=_cpu(trainer.model.state_dict()), optimizer=_cpu(trainer.optimizer.state_dict()),
                policy_rng=trainer.rng.getstate(), torch_rng=torch.get_rng_state().clone(),
                cuda_rng=[r.clone() for r in torch.cuda.get_rng_state_all()])


META = ("versions", "runtime", "sources", "boundary", "config", "episodes", "updates", "waves", "used_deal_seeds")


def save_checkpoint(trainer, directory):
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError(target)
    payload = capture(trainer)
    _finite(payload)
    buffer = BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    manifest = {k: payload[k] for k in META}
    manifest.update(sha256=sha256(data).hexdigest(), bytes=len(data))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name + ".incomplete-" + uuid4().hex)
    stage.mkdir()
    for name, content in (("checkpoint.pt", data), ("manifest.json", (json.dumps(
            manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))):
        with (stage / name).open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    # Same-filesystem rename. On Windows this refuses any existing destination.
    if target.exists():
        raise FileExistsError(target)
    os.rename(stage, target)
    return manifest


def _validate(payload, manifest):
    if set(payload) != set(META) | {"model", "optimizer", "policy_rng", "torch_rng", "cuda_rng"}:
        raise ValueError("checkpoint payload keys")
    for key in META:
        if manifest.get(key) != payload[key]:
            raise ValueError(f"manifest mismatch {key}")
    for key, expected in (("versions", versions()), ("runtime", _runtime_gate()),
                          ("sources", sources()), ("boundary", "wave_boundary")):
        if payload[key] != expected:
            raise ValueError(f"checkpoint incompatible {key}")
    for key in ("episodes", "updates", "waves"):
        if type(payload[key]) is not int or payload[key] < 0:
            raise ValueError("invalid counters")
    from experiments.p3h_training import GPUTrainer
    # Preserve global RNG even if construction or a later validation fails.
    trainer = GPUTrainer(payload["config"])
    if payload["episodes"] != payload["waves"] * trainer.config["num_envs"]:
        raise ValueError("wave/episode counters")
    if (payload["waves"] == 0) != (payload["updates"] == 0) or payload["updates"] < payload["waves"]:
        raise ValueError("update counters")
    seeds = payload["used_deal_seeds"]
    if (type(seeds) is not list or len(seeds) != payload["episodes"]
            or any(type(s) is not int or not 100000 <= s < 110000 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("invalid development seed history")
    _finite(payload)
    expected_model = trainer.model.state_dict()
    if set(payload["model"]) != set(expected_model):
        raise ValueError("model keys")
    for key, ref in expected_model.items():
        value = payload["model"][key]
        if not isinstance(value, torch.Tensor) or value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError("model tensor shape/dtype")
    optimizer = payload["optimizer"]
    if set(optimizer) != {"state", "param_groups"} or optimizer["param_groups"] != trainer.optimizer.state_dict()["param_groups"]:
        raise ValueError("optimizer configuration")
    parameters = list(trainer.model.parameters())
    if set(optimizer["state"]) != (set(range(len(parameters))) if payload["updates"] else set()):
        raise ValueError("optimizer state incomplete")
    for i, value in optimizer["state"].items():
        if set(value) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("optimizer state keys")
        step = value["step"]
        if not isinstance(step, torch.Tensor) or step.shape != torch.Size([]) or step.dtype != torch.float32 or step.item() != payload["updates"]:
            raise ValueError("optimizer step")
        for key in ("exp_avg", "exp_avg_sq"):
            if value[key].shape != parameters[i].shape or value[key].dtype != parameters[i].dtype:
                raise ValueError("optimizer tensor shape/dtype")
    random.Random().setstate(payload["policy_rng"])
    rng = payload["torch_rng"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.get_rng_state().shape:
        raise ValueError("CPU RNG shape/dtype")
    torch.Generator(device="cpu").set_state(rng)
    cuda_rng = payload["cuda_rng"]
    if type(cuda_rng) is not list or len(cuda_rng) != torch.cuda.device_count():
        raise ValueError("CUDA RNG count")
    for i, rng in enumerate(cuda_rng):
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.shape != torch.cuda.get_rng_state(i).shape:
            raise ValueError("CUDA RNG shape/dtype")
        torch.Generator(device=f"cuda:{i}").set_state(rng)
    trainer.model.load_state_dict(payload["model"], strict=True)
    trainer.optimizer.load_state_dict(optimizer)
    trainer.rng.setstate(payload["policy_rng"])
    for key in ("episodes", "updates", "waves", "used_deal_seeds"):
        setattr(trainer, key, payload[key])
    return trainer


def load_checkpoint(directory, expected_sha256=None):
    target = Path(directory)
    if ".incomplete-" in target.name:
        raise ValueError("uncommitted checkpoint")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    data = (target / "checkpoint.pt").read_bytes()
    digest = sha256(data).hexdigest()
    if digest != manifest["sha256"] or len(data) != manifest["bytes"]:
        raise ValueError("checkpoint hash/size")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("pinned checkpoint hash")
    payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
    cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        trainer = _validate(payload, manifest)
    except BaseException:
        torch.set_rng_state(cpu_before)
        torch.cuda.set_rng_state_all(cuda_before)
        raise
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return trainer
