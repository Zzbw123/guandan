"""Fixed paired teacher pretraining; no new validation or reserved test."""
from random import Random
from guandan.evaluation.contracts import Trial

VERSION = 'gd-p5f-ranking-pretraining-v1'
INITS = [314560, 314561, 314562]
ARMS = ['regression', 'ranking']
TRAIN = [(100000+i, 2+i%13, i%4) for i in range(1600)]
DEV = [(108100+i, 2+i%13, i%4) for i in range(26)]


def config(seed, objective):
    if type(seed) is not int or seed not in INITS or objective not in ARMS:
        raise ValueError('fixed initializer/objective required')
    return dict(seed=seed, objective=objective, lr=.001, adam_eps=1e-4, batch_size=64, chunk_size=1024)


def order(seed):
    rng = Random(seed)
    blocks = INITS.copy()
    rng.shuffle(blocks)
    jobs = []
    for initializer in blocks:
        arms = ARMS.copy()
        rng.shuffle(arms)
        jobs.extend(f'{a}-{initializer}' for a in arms)
    return jobs


def schedule():
    trials = []
    for opponent in ('greedy', 'random'):
        for i, (seed, level, _) in enumerate(DEV):
            for rotation in range(4):
                for swap in range(2):
                    focal = (rotation+swap)%4
                    policies = tuple('dmc' if s%2 == focal%2 else opponent for s in range(4))
                    trials.append(Trial(f'dmc|dmc|{opponent}', i, seed, level, rotation,
                                        swap, 0, focal, policies, 0))
    return trials


def specification():
    return dict(version=VERSION, initializers=INITS, arms=ARMS,
                configs={f'{a}-{s}': config(s,a) for s in INITS for a in ARMS},
                training_deals=[list(d) for d in TRAIN], development_deals=[list(d) for d in DEV],
                training_order=order(314563), development_order=order(314564),
                passes=1, batch_size=64, fit_probe_hands=13, fit_probe_per_hand=2,
                engineering_overfit_updates=512, engineering_overfit_seed=314560,
                teacher_ties='exact float64 equality; all physical candidates retained',
                inference='raw logits argmax; first maximum',
                fit_thresholds=dict(train=.99, development=.95),
                game_thresholds=dict(greedy=.45, random=.80),
                bootstrap_seed=314565, bootstrap_replicates=5000,
                development_games_per_job=416, total_development_games=2496,
                process_timeout_s=7200, validation_games=0, model_promoted=False,
                reserved_test_executed=False)
