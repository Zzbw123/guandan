"""P3h isolated DMC plus complete-candidate auxiliary regression.

Sampling/update order derives from frozen GPUTrainer; zero weight is tested
bitwise against it. Only observation and complete legal actions enter targets.
"""
import math
import torch
import guandan_gpu.training as base
from guandan_gpu.training import (GPUTrainer as BaseTrainer, _validated_deals,
    _model_device, _cuda_float32, _MAX_STEPS, score_many)
from guandan.env import HandEnv
from guandan.learning.encoding import encode_observation, encode_action
from experiments.p3h_objective import auxiliary_backward

class GPUTrainer(BaseTrainer):
    def __init__(self, config):
        config = dict(config)
        weight = config.pop('auxiliary_weight', None)
        if type(weight) not in (int, float) or weight not in (0., .1):
            raise ValueError('P3h auxiliary_weight must be exactly 0 or 0.1')
        super().__init__(config)
        self.config['auxiliary_weight'] = float(weight)

    def train_wave(self, deals):
        if not self.ready_for_checkpoint:
            raise RuntimeError("trainer is not at a successful wave boundary")
        deals = _validated_deals(deals, self.config["num_envs"], self.used_deal_seeds)
        self.ready_for_checkpoint = False
        device = _model_device(self.model)
        behavior_version = self.model_version
        runs = []
        for seed, level, player in deals:
            env = base.HandEnv()
            obs = env.reset(seed, initial_level=level, starting_player=player)
            runs.append(dict(env=env, obs=obs, trajectory=[], settlement=None, steps=0))
        scored_candidates = score_requests = score_batches = 0
        while any(run["settlement"] is None for run in runs):
            pending, requests = [], []
            for run in runs:
                if run["settlement"] is not None:
                    continue
                if run["steps"] >= _MAX_STEPS:
                    raise RuntimeError(f"hand exceeded {_MAX_STEPS} steps")
                obs = run["obs"]
                candidates = run["env"].legal_actions(obs.player_id)
                if not candidates:
                    raise RuntimeError("current player has no legal actions")
                features = encode_observation(obs)
                selected = None
                if self.rng.random() < self.config["epsilon"]:
                    selected = self.rng.randrange(len(candidates))
                else:
                    requests.append((obs, candidates))
                pending.append((run, candidates, features, selected))
            values = score_many(self.model, requests, self.config["chunk_size"]) if requests else []
            score_requests += len(requests)
            count = sum(len(request[1]) for request in requests)
            scored_candidates += count
            score_batches += math.ceil(count / self.config["chunk_size"])
            cursor = 0
            for run, candidates, features, selected in pending:
                if selected is None:
                    selected = int(torch.argmax(values[cursor]).item())
                    cursor += 1
                obs = run["obs"]
                action = candidates[selected]
                run["trajectory"].append((features, encode_action(action), obs.player_id % 2, (obs, candidates)))
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
            samples.extend((state, action, settlement.team_rewards[team], request)
                           for state, action, team, request in run["trajectory"])
            hands.append(dict(seed=deal[0], level=deal[1], starting_player=deal[2],
                              steps=run["steps"], samples=len(run["trajectory"]),
                              terminal_digest=digest, team_rewards=list(settlement.team_rewards),
                              replay_verified=True))
        self.rng.shuffle(samples)
        weighted_loss = 0.0
        auxiliary_candidates = 0
        auxiliary_weighted_loss = 0.0
        batch_records = []
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
            auxiliary = dict(loss=0., candidates=0, requests=0)
            if self.config['auxiliary_weight']:
                auxiliary = auxiliary_backward(self.model, [x[3] for x in batch],
                    self.config['auxiliary_weight'], self.config['chunk_size'])
            auxiliary_candidates += auxiliary['candidates']
            auxiliary_weighted_loss += auxiliary['loss'] * len(batch)
            batch_records.append(dict(samples=len(batch), dmc_loss=float(loss.item()),
                auxiliary_loss=auxiliary['loss'], auxiliary_candidates=auxiliary['candidates'],
                auxiliary_requests=auxiliary['requests']))
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
        self.episodes += len(deals)
        self.waves += 1
        self.used_deal_seeds.extend(deal[0] for deal in deals)
        self.ready_for_checkpoint = True
        return dict(wave=self.waves, behavior_version=behavior_version,
                    learning_version=self.model_version, episodes=self.episodes,
                    updates=self.updates, samples=len(samples), mean_loss=mean_loss,
                    hands=hands, scored_candidates=scored_candidates,
                    score_requests=score_requests, score_batches=score_batches,
                    auxiliary_candidates=auxiliary_candidates,
                    auxiliary_loss=auxiliary_weighted_loss / len(samples),
                    auxiliary_weight=self.config['auxiliary_weight'], batches=batch_records,
                    device_proof=dict(device=str(device), dtype=str(torch.float32),
                                      forward_checks=forward_count,
                                      gradient_checks=gradient_count, adam_tensor_checks=adam_count))
