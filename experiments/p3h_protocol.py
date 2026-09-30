"""P3h fixed single-condition paired auxiliary-loss experiment."""
from dataclasses import asdict
from random import Random
from guandan.evaluation.contracts import Trial

VERSION='gd-p3h-aux-v1'
INITS=[314380,314381,314382]
ARMS=['control','aux']
TRAIN_SEEDS=list(range(100200,100800))
DEV=list(range(108100,108126))
VALIDATION=list(range(204000,204065))
BOOTSTRAP_SEED=314405

def config(seed, arm='control'):
    if seed not in INITS or arm not in ARMS: raise ValueError('unknown fixed initializer/arm')
    return dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,
                auxiliary_weight=.1 if arm=='aux' else 0.)

def schedule(validation=False):
    trials=[]
    for opponent in (('greedy','random','team') if validation else ('greedy',)):
        for i,seed in enumerate(VALIDATION if validation else DEV):
            for rotation in range(4):
                for swap in range(2):
                    focal=(rotation+swap)%4
                    policies=tuple('dmc' if s%2==focal%2 else opponent for s in range(4))
                    trials.append(Trial(f'dmc|dmc|{opponent}',i,seed,2+i%13,
                        rotation,swap,0,focal,policies,0))
    return trials

def specification():
    train=[f'{arm}-{seed}' for seed in INITS for arm in ARMS]
    dev=train.copy();Random(314403).shuffle(train);Random(314404).shuffle(dev)
    return dict(version=VERSION,arms=ARMS,initializers=INITS,
        configs={f'{arm}-{seed}':config(seed,arm) for seed in INITS for arm in ARMS},
        training_order=train,development_order=dev,validation_order=['validation-control','validation-aux'],
        training_seeds=TRAIN_SEEDS,development_seeds=DEV,validation_seeds=VALIDATION,
        inherited_teacher_hands=200,new_hands_per_job=600,validation_initializer=314380,
        loss='mean_selected_DMC_MSE + lambda * mean_observation(mean_full_candidate_teacher_MSE)',
        auxiliary_weight=.1,target_range=[-.8,.8],singletons='include_zero_targets',
        candidate_policy='complete; no clipping, cap or subsampling',
        batch_policy='same shuffled DMC observations; exactly one Adam step per batch',
        compute_matching='environment hands only; updates, trajectories, labels and time may differ',
        common_reset='load P3f teacher phase1 model only; reset Adam and all RNG to initializer',
        fidelity_subset='first 128 multi-candidate observations from fixed teacher-314380 phase1 replay; six final models',
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=5000,policy_seed=271828,
        point_threshold=.55,ci_lower_threshold=.5,process_timeout_s=3600,
        startup_timeout_s=60.,decision_timeout_s=2.,model_promoted=False,reserved_test_executed=False,
        development_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule()],
        validation_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule(True)])
