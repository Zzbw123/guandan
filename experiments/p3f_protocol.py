"""Controller fixed single phase-one intervention, paired seeds and holdout."""
from dataclasses import asdict
from random import Random
from guandan.evaluation.contracts import Trial

VERSION = 'gd-p3f-teacher-v1'
INITS = [314380,314381,314382]
ARMS = ['control','teacher']
TRAIN_SEEDS = list(range(100000,100800))
DEV = list(range(108100,108126))
VALIDATION = list(range(203000,203065))
BOOTSTRAP_SEED = 314385

def config(seed):
    return dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4)

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
    dev=train.copy();Random(314383).shuffle(train);Random(314384).shuffle(dev)
    return dict(version=VERSION,configs=[config(s) for s in INITS],arms=ARMS,
        training_seeds=TRAIN_SEEDS,development_seeds=DEV,validation_seeds=VALIDATION,
        training_order=train,development_order=dev,phase1_hands=200,phase2_hands=600,
        teacher_observation_batch=64,teacher_target_range=[-.8,.8],teacher_singletons='include_zero_targets',
        teacher_objective='mean_per_observation_full_candidate_MSE',teacher_epochs=1,
        common_boundary_reset='Adam and all RNG reset to initializer; carry only model tensors',
        validation_initializer=314380,validation_order=['validation-control','validation-teacher'],
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=5000,policy_seed=271828,
        point_threshold=.55,ci_lower_threshold=.5,process_timeout_s=2400,
        startup_timeout_s=60.,decision_timeout_s=2.,model_promoted=False,reserved_test_executed=False,
        development_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule()],
        validation_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule(True)])
