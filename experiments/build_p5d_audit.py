"""One-time construction of the isolated controller; does not run experiments."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=(root/'experiments/controller_p5a_audit.py').read_text('utf-8')
source=source[:source.index('\ndef main():')]
source=source.replace('Independent P5a','Independent P5d')
source=source.replace("cfg=before['config']; learner", "cfg=before['config']; check(row['objective']==cfg['objective'],'objective binding'); learner")
source=source.replace("numerical_result={}\n", "audit_batch_rows(samples,row,cfg)\n    numerical_result={}\n")
source=source.replace("loss=torch.sum(residual.square())/len(batch)", """positive=sum(x[2]==1 for x in batch); negative=len(batch)-positive
            # Independent per-sample weighted sum, not the production class means.
            wp=len(batch)/(2*positive) if cfg['objective']=='label_balanced' and positive and negative else 1.0
            wn=len(batch)/(2*negative) if cfg['objective']=='label_balanced' and positive and negative else 1.0
            weights=torch.tensor([wp if x[2]==1 else wn for x in batch],device='cuda')
            loss=torch.sum(weights*residual.square())/len(batch)
            check(abs(float(loss.detach())-row['batches'][start//cfg['batch_size']]['loss'])<=3e-6,'batch numerical loss')""")
extra='''

def audit_batch_rows(samples,row,cfg):
    expected_batches=math.ceil(len(samples)/cfg['batch_size'])
    check(len(row['batches'])==expected_batches,'batch count')
    weighted_sum=0.0
    for bi,start in enumerate(range(0,len(samples),cfg['batch_size'])):
        batch=samples[start:start+cfg['batch_size']]; n=len(batch)
        positive=sum(x[2]==1 for x in batch);negative=n-positive
        balanced=cfg['objective']=='label_balanced' and positive>0 and negative>0
        wp=n/(2*positive) if balanced else float(positive>0)
        wn=n/(2*negative) if balanced else float(negative>0)
        expected=dict(objective=cfg['objective'],start=start,n=n,positive=positive,negative=negative,
                      positive_weight=wp,negative_weight=wn,single_class=not(positive and negative))
        actual=row['batches'][bi]
        check(set(actual)==set(expected)|{'loss'},'batch schema')
        for key,value in expected.items():
            check(type(actual[key]) is type(value) and actual[key]==value,'batch '+key)
        check(type(actual['loss']) is float and math.isfinite(actual['loss']) and actual['loss']>=0,'finite batch loss')
        weighted_sum+=actual['loss']*n
    check(abs(weighted_sum/len(samples)-row['mean_loss'])<1e-12,'batch aggregate loss')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();out=args.run.resolve()
    output=args.output.resolve() if args.output else out/'controller-audit.json'
    check(not output.exists(),'fresh controller report')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from experiments.p5d_checkpoint import sources,load_checkpoint
    from experiments.p5d_package import protected
    pre=read(out/'preregistration.json');receipt=read(out/'run-receipt.json')
    check(receipt['status']=='PASS','run completion')
    for name,h in {**pre['sources'],**pre['inputs']}.items():check(digest(ROOT/name)==h,'frozen source/input '+name)
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'run binding '+name)
    check(sources()==pre['checkpoint_sources'],'registered checkpoint sources')
    with zipfile.ZipFile(out/'source-snapshot.zip') as archive:
        check(set(archive.namelist())==set(pre['sources']),'archive paths')
        for name,h in pre['sources'].items():check(sha256(archive.read(name)).hexdigest()==h,'archive source')
    audits={};all_payloads={};all_rows={}
    for objective in ('ordinary','label_balanced'):
        folder=out/objective
        cfg=dict(seed=314500,epsilon=.1,lr=.001,batch_size=256,chunk_size=1024,num_envs=4,mode='mixed',objective=objective)
        paths=[folder/'initial']+[folder/f'wave-{w}' for w in range(1,5)]
        payloads=[]
        for path in paths:
            load_checkpoint(path)
            value=load_payload(path)
            check(value['config']==cfg and value['sources']==sources(),'fixed config/source')
            payloads.append(value)
        torch.manual_seed(314500);expected=DMCNetwork()
        check(equal(expected.state_dict(),payloads[0]['model']) and equal(payloads[0]['model'],payloads[0]['frozen']),'initial weights')
        check(payloads[0]['episodes']==payloads[0]['updates']==payloads[0]['waves']==0 and payloads[0]['used_deal_seeds']==[],'fresh initial state')
        complete=read(folder/'complete.json')
        check(complete==dict(status='PASS',objective=objective,episodes=16,waves=4,updates=payloads[-1]['updates'],
            source_sha256=sources(),scope='ENGINEERING_ONLY',model_promoted=False,validation_games=0,reserved_test_executed=False),'completion receipt')
        rows=[read(folder/f'wave-{w}.json') for w in range(1,5)]
        checks=[]
        for wi,row in enumerate(rows):
            checks.append(audit_wave(row,payloads[wi],payloads[wi+1],wi))
            print(objective,'independent wave',wi+1,'PASS',flush=True)
        audits[objective]=checks;all_payloads[objective]=payloads;all_rows[objective]=rows
    core=('model','optimizer','policy_rng','torch_rng','cuda_rng','episodes','updates','waves','used_deal_seeds','frozen','pool_sha256')
    check(all(equal(all_payloads['ordinary'][0][k],all_payloads['label_balanced'][0][k]) for k in core),'paired initial state')
    historical=ROOT/'artifacts/evaluations/p5a-pool-v2'
    for wi,payload in enumerate(all_payloads['ordinary']):
        old=load_payload(historical/('initial' if wi==0 else f'wave-{wi}'))
        for key in core:check(equal(payload[key],old[key]),'ordinary P5a bitwise '+key)
        if wi:
            new={k:v for k,v in all_rows['ordinary'][wi-1].items() if k not in ('objective','batches')}
            check(new==read(historical/f'wave-{wi}.json'),'ordinary historical wave log')
    for wi in (3,4):
        load_checkpoint(out/'resume'/f'wave-{wi}')
        check(equal(load_payload(out/'resume'/f'wave-{wi}'),all_payloads['label_balanced'][wi]),'fresh-process exact checkpoint')
        check(read(out/'resume'/f'wave-{wi}.json')==all_rows['label_balanced'][wi-1],'fresh-process exact log')
    check(read(out/'resume/complete.json')==read(out/'label_balanced/complete.json'),'resume completion')
    negative=negatives(all_rows['label_balanced'][0],all_payloads['label_balanced'][0],all_payloads['label_balanced'][1])
    row=all_rows['label_balanced'][0];before=all_payloads['label_balanced'][0];after=all_payloads['label_balanced'][1]
    mutations={
        'objective':lambda r:r.update(objective='ordinary'),
        'batch_count':lambda r:r['batches'].pop(),
        'batch_positive':lambda r:r['batches'][0].update(positive=r['batches'][0]['positive']+1),
        'batch_denominator':lambda r:r['batches'][0].update(n=r['batches'][0]['n']+1),
        'batch_weight':lambda r:r['batches'][0].update(positive_weight=r['batches'][0]['positive_weight']+.25),
        'batch_single_class':lambda r:r['batches'][0].update(single_class=not r['batches'][0]['single_class']),
        'batch_start':lambda r:r['batches'][0].update(start=1),
        'batch_loss':lambda r:r['batches'][0].update(loss=r['batches'][0]['loss']+.1)}
    for name,mutate in mutations.items():
        bad=copy.deepcopy(row);mutate(bad)
        try:audit_wave(bad,before,after,0,numerical=False)
        except (ValueError,AssertionError) as exc:negative[name]=str(exc)
        else:raise ValueError('negative accepted '+name)
    # Coordinated edits to both loss aggregates must still fail numerical recomputation.
    bad=copy.deepcopy(row);bad['batches'][0]['loss']+=.125
    bad['mean_loss']+=.125*bad['batches'][0]['n']/bad['samples']
    try:audit_wave(bad,before,after,0,numerical=True)
    except (ValueError,AssertionError) as exc:negative['coordinated_loss']=str(exc)
    else:raise ValueError('coordinated loss mutation accepted')
    history=protected()
    for name,h in {**pre['sources'],**pre['inputs']}.items():check(digest(ROOT/name)==h,'post-audit binding '+name)
    for name,h in receipt['artifacts'].items():check(digest(out/name)==h,'post-audit artifact '+name)
    result=dict(status='PASS',scope='ACCEPTED_LABEL_BALANCED_ENGINEERING_V1',training_hands=32,
        unique_deals=16,resume_repeated_hands=8,validation_games=0,reserved_test_executed=False,
        model_promoted=False,strength='NOT_ESTABLISHED',fresh_process_resume='BITWISE_EQUAL',
        ordinary_p5a_compatibility='BITWISE_EQUAL',waves=audits,negative_rejections=negative,
        historical=history,source_sha256=pre['sources'],controller_sha256=digest(Path(__file__)),
        artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    write(output,result)
    print(json.dumps(dict(status=result['status'],negative_rejections=len(negative),
        samples={k:sum(x['samples'] for x in v) for k,v in audits.items()},fresh_process_resume=result['fresh_process_resume'])),flush=True)


if __name__=='__main__':main()
'''
target=root/'experiments/controller_p5d_audit.py'
with target.open('x',encoding='utf-8') as f:f.write(source+extra)
