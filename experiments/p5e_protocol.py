"""Frozen, equal-hand P5e paired objective intervention."""
from dataclasses import asdict
from random import Random
from guandan.evaluation.contracts import Trial

VERSION='gd-p5e-paired-label-balanced-v1'
INITS=[314510,314511,314512]
ARMS=['ordinary','label_balanced']
TRAIN_SEEDS=list(range(100000,101600))
DEV=list(range(108100,108126))
VALIDATION=list(range(208000,208065))
BOOTSTRAP_SEED=314553

def config(seed,arm='ordinary'):
    if type(seed) is not int or seed not in INITS or arm not in ARMS:
        raise ValueError('fixed initializer/arm required')
    return dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,
                num_envs=4,mode='mixed',objective=arm)

def schedule(validation=False):
    if type(validation) is not bool: raise ValueError('boolean validation selector required')
    result=[]
    for opponent in (('greedy','random','team') if validation else ('greedy',)):
        for i,seed in enumerate(VALIDATION if validation else DEV):
            for rotation in range(4):
                for swap in range(2):
                    focal=(rotation+swap)%4
                    policies=tuple('dmc' if seat%2==focal%2 else opponent for seat in range(4))
                    result.append(Trial(f'dmc|dmc|{opponent}',i,seed,2+i%13,rotation,swap,0,focal,policies,0))
    return result

def blocked_order(seed):
    rng=Random(seed);blocks=INITS.copy();rng.shuffle(blocks);jobs=[]
    for initializer in blocks:
        arms=ARMS.copy();rng.shuffle(arms);jobs.extend(f'{a}-{initializer}' for a in arms)
    return jobs

def specification():
    validation=['validation-'+a for a in ARMS];Random(314552).shuffle(validation)
    return dict(version=VERSION,arms=ARMS,initializers=INITS,
        configs={f'{a}-{s}':config(s,a) for s in INITS for a in ARMS},
        training_order=blocked_order(314550),development_order=blocked_order(314551),validation_order=validation,
        training_seeds=TRAIN_SEEDS,development_seeds=DEV,validation_seeds=VALIDATION,
        hands_per_job=1600,total_training_hands=9600,waves_per_job=400,validation_initializer=314510,
        primary_candidate='label_balanced-314510',paired_control='ordinary-314510',
        comparison='equal hands; total effect of training objective intervention; visits, samples, and updates may differ',
        initialization='from scratch; no inherited training; frozen is immutable initial weights',
        checkpoint_waves=[0,1,2,3,4,200,400],candidate_selection='final wave 400 only',
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=5000,policy_seed=271828,
        point_threshold=.55,ci_lower_threshold=.5,process_timeout_s=3600,
        startup_timeout_s=60.,decision_timeout_s=2.,model_promoted=False,reserved_test_executed=False,
        development_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule()],
        validation_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule(True)])
