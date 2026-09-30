"""Export audited P3l paired results and batch gradient diagnostics; no model selection."""
from pathlib import Path
import sys,json,csv,statistics
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import read,write,digest,check
from experiments.p3l_protocol import specification

def main():
    root=Path(sys.argv[1]).resolve();audit=read(root/'controller-audit.json')
    check(audit['status']=='PASS','independent review required')
    out=root/'descriptives';check(not out.exists(),'immutable summary output');out.mkdir()
    records=[];summaries={};groups=[]
    for job in specification()['training_order']:
        waves=[json.loads(s) for s in (root/'training'/job/'waves.jsonl').read_text('utf-8').splitlines()]
        selected=[]
        for w in waves:
            for i,b in enumerate(w['batches']):
                r=dict(job=job,wave=w['wave'],batch=i,observations=b['samples'],candidates=b['auxiliary_candidates'],
                    singleton_observations=b['single_candidate_requests'],dmc_loss=b['dmc_loss'],centered_loss=b['auxiliary_loss'],
                    **{k:b[k] for k in ('d','c','alpha','auxiliary_to_dmc_ratio','cosine','cap_triggered','joint_descent_factor','alpha_branch')})
                records.append(r);selected.append(r)
        summaries[job]=dict(batches=len(selected),observations=sum(x['observations'] for x in selected),
            candidates=sum(x['candidates'] for x in selected),singleton_observations=sum(x['singleton_observations'] for x in selected),
            alpha_min=min(x['alpha'] for x in selected),alpha_median=statistics.median(x['alpha'] for x in selected),
            alpha_max=max(x['alpha'] for x in selected),cap_batches=sum(x['cap_triggered'] for x in selected),
            negative_cosine_batches=sum(x['cosine'] is not None and x['cosine']<0 for x in selected),
            ratio_max=max((x['auxiliary_to_dmc_ratio'] for x in selected if x['auxiliary_to_dmc_ratio'] is not None),default=None),
            descent_factor_min=min((x['joint_descent_factor'] for x in selected if x['joint_descent_factor'] is not None),default=None))
    for comparison in audit['paired']:
        for seed,level,delta in zip(comparison['deal_seeds'],comparison['levels'],comparison['group_delta']):
            groups.append(dict(validation=comparison['validation'],initializer=comparison['seed'],matchup=comparison['matchup'],
                deal_seed=seed,level=level,normcap_minus_constant=delta))
    for name,rows in [('batch-gradients.csv',records),('paired-deal-groups.csv',groups)]:
        with (out/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'summary.json',dict(status='PASS',role='descriptive only; no new hypothesis or selection',training=summaries,
        paired=audit['paired'],validation_gate=audit['validation_gate']))
    write(out/'trace.json',dict(status='PASS',audit_sha256=digest(root/'controller-audit.json'),
        script_sha256=digest(Path(__file__)),files={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        batch_rows=len(records),paired_group_rows=len(groups)))
    print(json.dumps(dict(status='PASS',batch_rows=len(records),paired_group_rows=len(groups))))
if __name__=='__main__':main()
