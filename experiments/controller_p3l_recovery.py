"""Persistent controller evidence for separate-process recovery and fail-closed IO."""
from pathlib import Path
import sys,json,subprocess,copy,io
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'experiments')]
import torch
from experiments.p3l_training import GPUTrainer
from experiments.p3l_checkpoint import capture,save_checkpoint,load_checkpoint,META
from experiments.p3l_protocol import config
from experiments.p3e_common import read,write,check,digest
from experiments.review_p3l import equal
from unittest.mock import patch
from hashlib import sha256

def same(a,b):
    fields=('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','waves','updates','used_deal_seeds')
    for k in fields:check(equal(a[k],b[k]),'bitwise restored '+k)
    return list(fields)

def main():
    folder=ROOT/'artifacts/evaluations'/ (sys.argv[1] if len(sys.argv)>1 else 'p3l-controller-recovery-v1')
    check(not folder.exists(),'fresh evidence directory');folder.mkdir()
    t=GPUTrainer(config(314381,'normcap'))
    first=t.train_wave([(109110+i,2+i,i) for i in range(4)])
    manifest=save_checkpoint(t,folder/'before')
    expected=t.train_wave([(109120+i,6+i,i) for i in range(4)])
    payload=capture(t);save_checkpoint(t,folder/'continuous')
    child=subprocess.run([sys.executable,str(ROOT/'scripts/p3l_resume_test.py'),str(folder/'before'),manifest['sha256'],str(folder/'restored')],capture_output=True,text=True,timeout=240)
    (folder/'child.log').write_text(child.stdout+child.stderr,encoding='utf-8')
    check(child.returncode==0,'child recovery exit')
    restored=torch.load(folder/'restored/checkpoint.pt',map_location='cpu',weights_only=True)
    fields=same(payload,restored);check(read(folder/'wave.json')==expected,'entire resumed wave/log')
    write(folder/'continuous-wave.json',expected);write(folder/'first-wave.json',first)
    negatives={}
    pristine=torch.load(folder/'before/checkpoint.pt',map_location='cpu',weights_only=True)
    faults=[('versions',{'checkpoint':'wrong'}),('sources',{}),('boundary','mid-wave'),('episodes',5),
            ('used_deal_seeds',[109110]*4),('config',{**pristine['config'],'objective':'unknown'}),
            ('optimizer',{'state':{},'param_groups':pristine['optimizer']['param_groups']}),
            ('torch_rng',torch.zeros(1,dtype=torch.uint8))]
    for key,value in faults:
        bad=copy.deepcopy(pristine);bad[key]=value
        p=folder/('tampered-'+key);p.mkdir()
        buffer=io.BytesIO();torch.save(bad,buffer);data=buffer.getvalue()
        (p/'checkpoint.pt').write_bytes(data)
        m={k:bad[k] for k in META};m.update(sha256=sha256(data).hexdigest(),bytes=len(data));write(p/'manifest.json',m)
        cpu,cuda=torch.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()]
        try:load_checkpoint(p)
        except (ValueError,RuntimeError,TypeError):negatives[key]='REJECTED'
        else:raise AssertionError('tamper accepted '+key)
        check(torch.equal(cpu,torch.get_rng_state()) and equal(cuda,torch.cuda.get_rng_state_all()),'failed load RNG rollback')
    t=load_checkpoint(folder/'before',manifest['sha256'])
    with patch('experiments.p3l_training.joint_backward',side_effect=ValueError('injected')):
        try:t.train_wave([(109120+i,6+i,i) for i in range(4)])
        except ValueError:pass
        else:raise AssertionError('fault not injected')
    check(not t.ready_for_checkpoint,'failed trainer locks')
    try:save_checkpoint(t,folder/'should-not-exist')
    except ValueError:negatives['failed_wave_save']='REJECTED'
    else:raise AssertionError('failed trainer committed')
    t=load_checkpoint(folder/'before',manifest['sha256'])
    check(t.train_wave([(109120+i,6+i,i) for i in range(4)])==expected,'failed wave replay')
    same(capture(t),payload)
    from experiments.p3l_guard import GPUInferenceGuard
    from benchmark_learning_runtime import high_branch_fixture
    with GPUInferenceGuard(folder/'before',manifest['sha256']) as guard:
        obs,legal=high_branch_fixture();action,info=guard.act(obs,legal)
        check(action in legal and info['scored_candidates']==8769,'real guard all candidates')
    check(guard.closed and not guard.is_alive(),'guard reaped')
    for mode in ('hang_start','hang_act','crash_act','bad_index'):
        try:
            with GPUInferenceGuard(Path('unused'),'0'*64,startup_timeout=.5 if mode=='hang_start' else 60,decision_timeout=.2,_test_mode=mode) as g:
                g.act(obs,legal)
        except (RuntimeError,TimeoutError):negatives[mode]='REJECTED'
        else:raise AssertionError('guard fault accepted')
    paths=[p for directory,pattern in [('experiments','p3l*.py'),('scripts','p3l*.py'),('tests','test_p3l*.py')] for p in (ROOT/directory).glob(pattern)]
    receipt=dict(status='PASS',bitwise_fields=fields,full_wave_log_equal=True,negatives=negatives,
        high_candidates=8769,script_sha256=digest(Path(__file__)),
        source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        artifacts={p.relative_to(ROOT).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()})
    write(ROOT/'artifacts/evaluations/p3l-resume-test.json',receipt)
    print(json.dumps(dict(status='PASS',bitwise_fields=fields,negative_cases=len(negatives))))
if __name__=='__main__':main()
