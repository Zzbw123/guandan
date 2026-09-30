"""Complete-candidate CUDA scoring and frozen-actor synchronous waves."""
from __future__ import annotations

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import math
import copy
from hashlib import sha256
from guandan.agents.baselines import GreedyAgent, TeamHeuristicAgent
import random
import torch

from guandan.env import HandEnv
from guandan.learning.encoding import ACTION_DIM, STATE_DIM, encode_action, encode_observation
from guandan.learning.model import DMCNetwork
from guandan.types import Action, PlayerObservation

_CONFIG_KEYS = frozenset({"seed", "epsilon", "lr", "batch_size", "chunk_size", "num_envs", "mode"})
_MAX_STEPS = 1000


def _validated_config(config: dict) -> dict:
    if type(config) is not dict or set(config) != _CONFIG_KEYS:
        raise ValueError(f"config must contain exactly {sorted(_CONFIG_KEYS)}")
    for key in ("seed", "batch_size", "chunk_size", "num_envs"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if config["seed"] < 0 or min(config["batch_size"], config["chunk_size"]) <= 0:
        raise ValueError("seed must be nonnegative and batch/chunk sizes positive")
    if max(config["batch_size"], config["chunk_size"]) > 65536:
        raise ValueError("batch_size and chunk_size must be at most 65536")
    if not 1 <= config["num_envs"] <= 8:
        raise ValueError("num_envs must be in 1..8")
    for key in ("epsilon", "lr"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= config["epsilon"] <= 1 or config["lr"] <= 0:
        raise ValueError("epsilon must be in [0, 1] and lr positive")
    if config["mode"] not in ("selfplay", "mixed"):
        raise ValueError("unsupported pool mode")
    return dict(config)


def _validated_deals(deals, num_envs: int, used_deal_seeds) -> list:
    if type(deals) is not list or len(deals) != num_envs:
        raise ValueError("deals must be a list of length num_envs")
    seen = set(used_deal_seeds)
    for deal in deals:
        if type(deal) is not tuple or len(deal) != 3:
            raise ValueError("each deal must be (seed, level, starting_player)")
        seed, level, player = deal
        if type(seed) is not int or not 100000 <= seed <= 109999:
            raise ValueError("deal seed must be in development range 100000..109999")
        if seed in seen:
            raise ValueError("deal seeds must be unique and unused")
        if type(level) is not int or not 2 <= level <= 14:
            raise ValueError("level must be an integer in 2..14")
        if type(player) is not int or not 0 <= player <= 3:
            raise ValueError("starting_player must be an integer in 0..3")
        seen.add(seed)
    return list(deals)


from guandan_gpu.training import _cuda_float32, _model_device, score_many

POOL_VERSION = "gd-p5a-pool-v1"

def weight_digest(model):
    h = sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def lineup(mode, index):
    if type(index) is not int or index < 0 or mode not in ("selfplay", "mixed"):
        raise ValueError("invalid schedule input")
    if mode == "selfplay":
        return ["current"] * 4
    rows = (("current", "greedy", "current", "greedy"),
            ("current", "team", "greedy", "team"),
            ("current", "frozen", "team", "frozen"),
            ("current", "greedy", "frozen", "team"))
    relative = rows[(index // 4) % 4]
    return [relative[(seat - index % 4) % 4] for seat in range(4)]

class GPUTrainer:
    """One fixed behavior model per wave; updates start after all replays pass."""
    def __init__(self, config):
        self.config = _validated_config(config)
        if not torch.cuda.is_available():
            raise RuntimeError("GPUTrainer requires available CUDA")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed_all(self.config["seed"])
        self.model = DMCNetwork().to(device="cuda", dtype=torch.float32)
        self.frozen = copy.deepcopy(self.model).eval()
        self.frozen.requires_grad_(False)
        self.pool_sha256 = weight_digest(self.frozen)
        self.baselines = {"greedy": GreedyAgent(), "team": TeamHeuristicAgent()}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], capturable=True)
        self.rng = random.Random(self.config["seed"])
        self.episodes = self.updates = self.waves = 0
        self.used_deal_seeds = []
        self.ready_for_checkpoint = True

    @property
    def model_version(self):
        return self.waves

    def train_wave(self, deals):
        if not self.ready_for_checkpoint:
            raise RuntimeError("trainer is not at a successful wave boundary")
        deals = _validated_deals(deals, self.config["num_envs"], self.used_deal_seeds)
        if weight_digest(self.frozen) != self.pool_sha256:
            self.ready_for_checkpoint = False
            raise ValueError("frozen pool changed")
        self.ready_for_checkpoint = False
        device = _model_device(self.model)
        behavior_version = self.model_version
        runs = []
        for hand_index, (seed, level, player) in enumerate(deals):
            env = HandEnv()
            obs = env.reset(seed, initial_level=level, starting_player=player)
            runs.append(dict(env=env, obs=obs, trajectory=[], settlement=None, steps=0,
                             roles=lineup(self.config["mode"], self.episodes + hand_index), decisions=[]))
        scored_candidates = score_requests = score_batches = 0
        frozen_candidates = frozen_requests = 0
        while any(run["settlement"] is None for run in runs):
            pending, requests, frozen_reqs = [], [], []
            for run in runs:
                if run["settlement"] is not None:
                    continue
                if run["steps"] >= _MAX_STEPS:
                    raise RuntimeError(f"hand exceeded {_MAX_STEPS} steps")
                obs = run["obs"]
                candidates = run["env"].legal_actions(obs.player_id)
                if not candidates:
                    raise RuntimeError("current player has no legal actions")
                role = run["roles"][obs.player_id]
                features = encode_observation(obs) if role == "current" else None
                selected, explored = None, False
                if role == "current":
                    if self.rng.random() < self.config["epsilon"]:
                        selected = self.rng.randrange(len(candidates))
                        explored = True
                    else:
                        requests.append((obs, candidates))
                elif role == "frozen":
                    frozen_reqs.append((obs, candidates))
                else:
                    chosen = self.baselines[role].act(obs, tuple(candidates))
                    if chosen not in candidates:
                        raise ValueError("baseline returned illegal action")
                    selected = candidates.index(chosen)
                pending.append((run, candidates, features, selected, role, explored))
            values = score_many(self.model, requests, self.config["chunk_size"]) if requests else []
            score_requests += len(requests)
            count = sum(len(request[1]) for request in requests)
            scored_candidates += count
            score_batches += math.ceil(count / self.config["chunk_size"])
            fixed_values = score_many(self.frozen, frozen_reqs, self.config["chunk_size"]) if frozen_reqs else []
            frozen_requests += len(frozen_reqs)
            frozen_candidates += sum(len(r[1]) for r in frozen_reqs)
            cursor = fixed_cursor = 0
            for run, candidates, features, selected, role, explored in pending:
                if selected is None:
                    if role == "current":
                        selected = int(torch.argmax(values[cursor]).item())
                        cursor += 1
                    else:
                        selected = int(torch.argmax(fixed_values[fixed_cursor]).item())
                        fixed_cursor += 1
                obs = run["obs"]
                action = candidates[selected]
                if role == "current":
                    run["trajectory"].append((features, encode_action(action), obs.player_id % 2))
                run["decisions"].append(dict(player=obs.player_id, role=role, candidates=len(candidates),
                                             selected=selected, explored=explored, sample=role == "current"))
                result = run["env"].step(obs.player_id, action, state_version=obs.state_version)
                run["steps"] += 1
                if result.terminal:
                    run["settlement"] = result.settlement
                else:
                    if result.next_player is None:
                        raise RuntimeError("missing next player")
                    run["obs"] = run["env"].observe(result.next_player)
        samples, hands = [], []
        for deal, run in zip(deals, runs):
            env, settlement = run["env"], run["settlement"]
            if settlement is None or settlement.team_rewards not in ((1, -1), (-1, 1)):
                raise RuntimeError("invalid terminal team rewards")
            digest = env.state_digest()
            replay = env.serialize_replay()
            verified = HandEnv.replay(replay)
            if (not verified.state.terminal or verified.state.settlement != settlement
                    or verified.state_digest() != digest or replay["final_digest"] != digest
                    or len(replay["steps"]) != run["steps"]):
                raise RuntimeError("terminal research replay verification failed")
            samples.extend((state, action, settlement.team_rewards[team])
                           for state, action, team in run["trajectory"])
            hands.append(dict(seed=deal[0], level=deal[1], starting_player=deal[2],
                              steps=run["steps"], samples=len(run["trajectory"]),
                              terminal_digest=digest, team_rewards=list(settlement.team_rewards),
                              replay_verified=True, roles=run["roles"], decisions=run["decisions"], replay=replay))
        if not samples:
            raise ValueError("no learner samples")
        self.rng.shuffle(samples)
        weighted_loss = 0.0
        forward_count = gradient_count = adam_count = 0
        for start in range(0, len(samples), self.config["batch_size"]):
            batch = samples[start:start + self.config["batch_size"]]
            states = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(device)
            actions = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(device)
            targets = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(device)
            for label, tensor in (("training states", states), ("training actions", actions), ("targets", targets)):
                _cuda_float32(tensor, label, device)
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(states, actions)
            _cuda_float32(predictions, "training forward", device)
            forward_count += 1
            if predictions.shape != targets.shape or not torch.isfinite(predictions).all().item():
                raise ValueError("invalid training predictions")
            loss = torch.nn.functional.mse_loss(predictions, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite training loss")
            loss.backward()
            for parameter in self.model.parameters():
                if parameter.grad is None:
                    raise ValueError("missing gradient")
                _cuda_float32(parameter.grad, "gradient", device)
                if not torch.isfinite(parameter.grad).all().item():
                    raise ValueError("nonfinite gradient")
                gradient_count += 1
            self.optimizer.step()
            for parameter in self.model.parameters():
                _cuda_float32(parameter, "updated parameter", device)
                if not torch.isfinite(parameter).all().item():
                    raise ValueError("nonfinite updated parameter")
                for tensor in self.optimizer.state[parameter].values():
                    if isinstance(tensor, torch.Tensor):
                        _cuda_float32(tensor, "Adam state", device)
                        if not torch.isfinite(tensor).all().item():
                            raise ValueError("nonfinite Adam state")
                        adam_count += 1
            weighted_loss += float(loss.item()) * len(batch)
            self.updates += 1
        mean_loss = weighted_loss / len(samples)
        if not math.isfinite(mean_loss):
            raise ValueError("nonfinite mean loss")
        if weight_digest(self.frozen) != self.pool_sha256:
            raise ValueError("frozen pool changed during wave")
        self.episodes += len(deals)
        self.waves += 1
        self.used_deal_seeds.extend(deal[0] for deal in deals)
        self.ready_for_checkpoint = True
        return dict(wave=self.waves, behavior_version=behavior_version,
                    learning_version=self.model_version, episodes=self.episodes,
                    updates=self.updates, samples=len(samples), mean_loss=mean_loss,
                    hands=hands, scored_candidates=scored_candidates,
                    score_requests=score_requests, score_batches=score_batches,
                    frozen_requests=frozen_requests, frozen_candidates=frozen_candidates,
                    pool_sha256=self.pool_sha256, pool_version=POOL_VERSION,
                    device_proof=dict(device=str(device), dtype=str(torch.float32),
                                      forward_checks=forward_count,
                                      gradient_checks=gradient_count, adam_tensor_checks=adam_count))
