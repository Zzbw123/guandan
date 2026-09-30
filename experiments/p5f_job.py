"""P5f fixed-budget FIFO training and public-state teacher-fit measurements."""
from collections import Counter,defaultdict
import gzip,json,math,time
from pathlib import Path
from experiments.p5f_run_common import check,digest,read,write,verify_frozen
from experiments.p5f_protocol import config
from experiments.p5f_data import iter_batches,iter_hands,verify_corpus
from experiments.p5f_training import TeacherTrainer
from experiments.p5f_checkpoint import save_checkpoint,load_model


def train(root,job):
    verify_frozen(root)
    objective,seed=job.split('-');cfg=config(int(seed),objective)
    corpus=root/'corpus/train';manifest=verify_corpus(corpus)
    out=root/'training'/job;out.mkdir(parents=True,exist_ok=False)
    trainer=TeacherTrainer(cfg,digest(corpus/'manifest.json'))
    initial=save_checkpoint(trainer,out/'initial')
    started=time.perf_counter()
    with (out/'batches.jsonl').open('x',encoding='utf-8') as log:
        for requests in iter_batches(corpus,64):
            row=trainer.update(requests)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if trainer.updates in (1,2):save_checkpoint(trainer,out/f'batch-{trainer.updates}')
            if trainer.updates%100==0:
                print(json.dumps(dict(job=job,updates=trainer.updates,samples=trainer.samples,
                    candidates=trainer.candidates,seconds=time.perf_counter()-started)),flush=True)
    check(trainer.samples==manifest['observations'] and trainer.candidates==manifest['candidates']
          and trainer.updates==math.ceil(trainer.samples/64),'complete one-pass training')
    final=save_checkpoint(trainer,out/'final')
    write(out/'report.json',dict(status='PASS',job=job,config=cfg,initial=initial,final=final,
        hands=manifest['hands'],samples=trainer.samples,candidates=trainer.candidates,
        updates=trainer.updates,seconds=time.perf_counter()-started,
        batches_sha256=digest(out/'batches.jsonl'),model_promoted=False,reserved_test_executed=False))
    verify_frozen(root)


def probe_requests(corpus):
    chosen=[]
    for meta,requests in iter_hands(corpus):
        if meta['index']>=13:break
        chosen.extend([(meta['index'],step,obs,legal) for step,(obs,legal) in
                       enumerate(requests) if len(legal)>1][:2])
    return chosen


def score_rows(model,selected):
    from experiments.p5f_objective import score_requests,teacher_labels
    result=[]
    for deal,step,obs,legal in selected:
        scores=score_requests(model,[(obs,legal)],1024)[0].tolist()
        labels=teacher_labels(obs,legal);teacher=labels['scores']
        index=max(range(len(scores)),key=scores.__getitem__)
        lo,hi=min(teacher),max(teacher)
        result.append(dict(deal_index=deal,step=step,candidates=len(legal),
            index=index,match=labels['optimal'][index],multi=len(legal)>1,
            long_candidates=len(legal)>1024,endgame=len(obs.hand)<=5,
            finish_opportunity=any(a.kind!='pass' and len(a.cards)==len(obs.hand) for a in legal),
            regret=0. if hi==lo else (hi-teacher[index])/(hi-lo),
            logits=scores,teacher_scores=teacher))
    return result


def fit_summary(rows):
    output={}
    for group in ('all','multi','long_candidates','endgame','finish_opportunity'):
        selected=rows if group=='all' else [r for r in rows if r[group]]
        deals=defaultdict(list)
        for r in selected:deals[r['deal_index']].append(int(r['match']))
        output[group]=dict(observations=len(selected),matches=sum(r['match'] for r in selected),
            accuracy=sum(r['match'] for r in selected)/len(selected) if selected else None,
            mean_regret=sum(r['regret'] for r in selected)/len(selected) if selected else None,
            represented_deals=len(deals),deal_equal_accuracy=
                sum(sum(v)/len(v) for v in deals.values())/len(deals) if deals else None)
    return output


def fit(root,job):
    verify_frozen(root)
    checkpoint=root/'training'/job/'final'
    candidates=read(root/'candidates.json')['candidates']
    model,_=load_model(checkpoint,candidates[job],digest(checkpoint/'manifest.json'))
    out=root/'fit'/job;out.mkdir(parents=True,exist_ok=False)
    results={}
    for source in ('train_probe','development'):
        if source=='train_probe':selected=probe_requests(root/'corpus/train')
        else:
            selected=[(meta['index'],step,obs,legal) for meta,requests in iter_hands(root/'corpus/development')
                      for step,(obs,legal) in enumerate(requests)]
        rows=score_rows(model,selected)
        with gzip.open(out/f'{source}.jsonl.gz','xt',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        results[source]=fit_summary(rows)
    write(out/'report.json',dict(status='PASS',job=job,candidate_sha256=candidates[job],
        summary=results,artifact_sha256={f'{s}.jsonl.gz':digest(out/f'{s}.jsonl.gz') for s in results},
        scope='development teacher agreement, not playing strength'))
    print(json.dumps(dict(stage='fit',job=job,summary=results)),flush=True)
    verify_frozen(root)
