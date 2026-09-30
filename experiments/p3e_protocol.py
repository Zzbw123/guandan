"""Controller-owned fixed budgets, paired development and one validation candidate."""
from dataclasses import asdict
from random import Random
from guandan.evaluation.contracts import Trial

VERSION = 'gd-p3e-scale-v1'
INITS = [314370, 314371, 314372]
DEV = list(range(108000,108026))
VALIDATION = list(range(202000,202065))
TRAIN = list(range(105000,106600))
OLD = 'artifacts/evaluations/p3d-validation-v2/training/wave-100'
OLD_HASH = '32935318ccd3bb76a38f71ddc21796cd204c1bc938ed8b3e6b6b9d081564049d'

def config(seed):
    return dict(seed=seed,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4)

def schedule(validation=False,baseline=False):
    trials=[]
    focal_name='greedy' if baseline else 'dmc'
    for opponent in (('greedy','random','team') if validation else ('greedy',)):
        for i,seed in enumerate(VALIDATION if validation else DEV):
            for rotation in range(4):
                for swap in range(2):
                    focal=(rotation+swap)%4
                    policies=tuple(focal_name if s%2==focal%2 else opponent for s in range(4))
                    trials.append(Trial(f'{focal_name}|{focal_name}|{opponent}',i,seed,2+i%13,
                                        rotation,swap,0,focal,policies,0))
    return trials

def specification():
    training_order=INITS.copy(); Random(314373).shuffle(training_order)
    development_order=[f'{s}-{n}' for s in INITS for n in (400,1600)]
    Random(314374).shuffle(development_order)
    return dict(version=VERSION,configs=[config(s) for s in INITS],training_seeds=TRAIN,
        development_seeds=DEV,validation_seeds=VALIDATION,training_order=training_order,
        development_order=development_order,budgets=[400,1600],waves=400,
        candidate='314370-1600',old_checkpoint=OLD,old_sha256=OLD_HASH,
        bootstrap_seed=314375,bootstrap_replicates=5000,policy_seed=271828,
        point_threshold=.55,ci_lower_threshold=.5,process_timeout_s=1800,
        startup_timeout_s=60.,decision_timeout_s=2.,model_promoted=False,reserved_test_executed=False,
        validation_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule(True)],
        development_trials=[{**asdict(t),'policies':list(t.policies)} for t in schedule()])
