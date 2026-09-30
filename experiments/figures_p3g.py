"""Plot only controller-audited P3g descriptive evidence; never select a model."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from research_figure_export import export_figure_bundle

def main():
    root=Path(sys.argv[1]).resolve();a=json.loads((root/'controller-audit.json').read_text('utf-8'));assert a['status']=='PASS'
    out=root/'descriptives';out.mkdir(exist_ok=True)
    paired=[];metrics=[]
    for seed,v in a['paired'].items():
        for j,(first,last) in enumerate(zip(v['phase1_values'],v['final_values'])):
            paired.append(dict(initializer=seed,deal_seed=108100+j,level=2+j%13,phase1=first,final=last,delta=last-first))
    for source,models in a['aggregates'].items():
        for model,groups in models.items():
            for group,values in groups.items():metrics.append(dict(source=source,model=model,subset=group,**values))
    for name,rows in [('paired-groups.csv',paired),('score-metrics.csv',metrics)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    colors=['#4477AA','#EE7733','#228833']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    contract=dict(claim='Separate teacher fit on identical states from development win-rate changes after DMC.',
        backend='python',archetype='paired quantitative panels and shared-scale annotated matrices',
        evidence='controller-audit; 26 original development deal clusters, eight variants each; no new validation',
        color_mapping=dict(zip(a['paired'],colors)),scales=dict(win_percent=[0,100],delta_pp=[-100,100],top1_percent=[0,100],regret=[0,1]),
        missing='not allowed in shown metrics',overflow='not clipped',exports='300 dpi PNG and editable SVG, CSV and hashes',
        review_risks=['Reused development data; intervals conditional on initializer','Purposive state sample is not a population estimate','Teacher agreement is not strength','Training source is a subset of demonstrations'])
    sources=[root/'controller-audit.json',out/'paired-groups.csv',out/'score-metrics.csv']
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.2),layout='constrained')
    for i,(seed,v) in enumerate(a['paired'].items()):
        col=colors[i];first=100*v['phase1'];last=100*v['final'];delta=100*v['delta'];lo,hi=[100*x for x in v['ci95']]
        ax[0].plot([0,1],[first,last],'-o',color=col,label=f'{seed}: {first:.1f}% → {last:.1f}%')
        vals=[100*(b-c) for b,c in zip(v['final_values'],v['phase1_values'])]
        ax[1].scatter(vals,[i+((j%7)-3)*.027 for j in range(26)],s=17,color=col,alpha=.45)
        ax[1].errorbar(delta,i+.2,xerr=[[delta-lo],[hi-delta]],fmt='D',color=col,capsize=4)
        ax[1].text(-97,i+.34,f'{delta:+.1f} pp [{lo:+.1f}, {hi:+.1f}]',fontsize=8,color=col)
    ax[0].set(xlim=(-.35,1.35),ylim=(0,100),xticks=[0,1],xticklabels=['Teacher phase (200 hands)','After DMC (+600 hands)'],ylabel='Win rate vs greedy (%)',title='A  Same 26 development deals')
    ax[0].axhline(50,color='#888888',ls='--',lw=1);ax[0].legend(title='Initialization',loc='upper left')
    ax[1].set(xlim=(-100,100),ylim=(-.35,2.7),yticks=[0,1,2],yticklabels=list(a['paired']),xlabel='Final minus phase1 (percentage points)',title='B  Paired group changes and 95% intervals')
    ax[1].axvline(0,color='#888888',ls='--',lw=1)
    ax[1].text(.02,.015,'Dots: deal groups; diamonds: mean and 95% interval',transform=ax[1].transAxes,fontsize=7.5)
    fig.suptitle('P3g development diagnostic — no training or new validation',fontsize=12,fontweight='bold')
    export_figure_bundle(fig,'development-change',out,contract,sources,Path(__file__),include_pdf=False,backend='python');plt.close(fig)
    order=['training']+[f'{phase}-{seed}' for phase in ('final','phase1') for seed in a['paired']]
    models=[f'{phase}-{seed}' for phase in ('phase1','final') for seed in a['paired']]
    labels=['Teacher train subset']+[f'{phase} trajectory {seed}' for phase in ('final','phase1') for seed in a['paired']]
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.3),layout='constrained')
    for panel,(key,scale,high,title) in enumerate([('mean_teacher_top1',100,100,'A  Teacher top-1 agreement (%)'),('mean_normalized_regret',1,1,'B  Normalized teacher regret')]):
        data=np.array([[a['aggregates'][s][m]['multi_choice'][key]*scale for m in models] for s in order]);assert np.isfinite(data).all() and data.min()>=0 and data.max()<=high
        im=axes[panel].imshow(data,vmin=0,vmax=high,cmap='viridis',aspect='auto')
        for i in range(len(order)):
            for j in range(len(models)):
                v=data[i,j];axes[panel].text(j,i,f'{v:.1f}' if scale==100 else f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if v/high<.5 else 'black')
        axes[panel].set(xticks=range(6),xticklabels=[m.replace('-','\n') for m in models],yticks=range(7),yticklabels=labels if panel==0 else [],title=title)
        axes[panel].axvline(2.5,color='white',lw=2);fig.colorbar(im,ax=axes[panel],shrink=.8)
    fig.suptitle('Same complete candidates scored by all six models — multi-choice states only\nAll finish opportunities + first four other choices/game; purposive diagnostic sample',fontsize=11)
    export_figure_bundle(fig,'teacher-fidelity',out,contract,sources,Path(__file__),include_pdf=False,backend='python');plt.close(fig)
    trace=dict(script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),audit_sha256=sha256((root/'controller-audit.json').read_bytes()).hexdigest(),paired_rows=len(paired),metric_rows=len(metrics),matplotlib=matplotlib.__version__,artifact_sha256={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name not in ('trace.json','visual-qa.json')})
    (out/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    print(json.dumps(dict(paired_rows=len(paired),metric_rows=len(metrics))))

if __name__=='__main__':main()
