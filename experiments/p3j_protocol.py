"""Fixed two-arm centering experiment; paired blocks and one validation candidate."""
from dataclasses import asdict
from random import Random
from guandan.evaluation.contracts import Trial

VERSION='gd-p3j-centered-v1'
INITS=[314380,314381,314382]
ARMS=['absolute','centered']
TRAIN_SEEDS=list(range(100200,100800))
DEV=list(range(108100,108126))
VALIDATION=list(range(205000,205065))
BOOTSTRAP_SEED=314415

def config(seed,arm='absolute'):
    if type(seed) is not int or seed not in INITS or arm not in ARMS:raise ValueError('fixed initializer/arm required')
    return dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,auxiliary_weight=.1,objective=arm)

def schedule(validation=False):
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
        arms=ARMS.copy();rng.shuffle(arms);jobs += [f'{arm}-{initializer}' for arm in arms]
    return jobs

def specification():
    validation=['validation-'+a for a in ARMS];Random(314416).shuffle(validation)
    return dict(version=VERSION,arms=ARMS,initializers=INITS,
        configs={f'{a}-{s}':config(s,a) for s in INITS for a in ARMS},
        training_order=blocked_order(314413),development_order=blocked_order(314414),validation_order=validation,
        training_seeds=TRAIN_SEEDS,development_seeds=DEV,validation_seeds=VALIDATION,
        inherited_teacher_hands=200,new_hands_per_job=600,validation_initializer=314380,
        primary_candidate='centered-314380',auxiliary_weight=.1,
        loss=dict(absolute='DMC MSE + .1 state-mean full-candidate teacher MSE',centered='DMC MSE + .1 state-mean variance(q - teacher_target)'),
        singletons='included in state denominator; zero centered term',candidate_policy='complete; no cap or subsampling',
        batch_policy='same shuffled DMC observations; one Adam step per batch',
        compute_matching='environment hands only; no equality of updates, labels, trajectories or time',
        common_reset='only P3f teacher-200 model; reset Adam and Python/CPU/CUDA RNG',
        fidelity_subset='unchanged first 128 multi-candidate teacher-314380 phase1 states',
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=5000,policy_seed=271828,
        point_threshold=.55,ci_lower_threshold=.5,process_timeout_s=3600,
        startup_timeout_s=60.,decision_timeout_s=2.,model_promoted=False,reserved_test_executed=False,
        development_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule()],
        validation_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule(True)])
