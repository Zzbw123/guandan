"""Source-backed paired-condition figures; consumes controller audited results."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_figure_export import export_figure_bundle

def main():
    root=Path(sys.argv[1]).resolve();audit_path=root/'controller-audit.json'
    a=json.loads(audit_path.read_text(encoding='utf-8'));assert a['status']=='PASS'
    output=root/'descriptives';output.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    colors=['#4477AA','#EE7733','#228833'];rows=[]
    for job,e in a['evaluations'].items():
        for matchup,s in e['summary'].items():
            for g in s['clusters']:rows.append(dict(job=job,matchup=matchup,**g))
    csvpath=output/'groups.csv'
    with csvpath.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    contract=dict(claim='Compare one fixed auxiliary-loss condition with DMC from identical teacher weights; report validation separately.',
        archetype='paired point and interval panels',backend='python',unit='original deal mean across 8 variants',
        export='300 dpi PNG and editable SVG plus source CSV and hash trace',
        colors={**dict(zip(['initialization 314380','initialization 314381','initialization 314382'],colors)),
                'control':'#777777','aux':'#AA4499'},
        scales=dict(probability=[0,100],difference=[-100,100]),missing_values='none allowed',
        review_risks=['Intervals conditional on initializer','65 validation groups not 520 independent games',
                      'Auxiliary targets change subsequent on-policy data','Equal hands do not imply equal updates or compute',
                      'Secondary comparisons are pointwise descriptive'])
    dev=sorted([r for r in a['paired'] if not r['validation']],key=lambda r:r['seed'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for i,r in enumerate(dev):
        col=colors[i]
        axes[0].plot([0,1],[100*r['control'],100*r['aux']],'-o',color=col,label=str(r['seed']))
        for x,arm in enumerate(('control','aux')):
            metric=a['evaluations'][f"{arm}-{r['seed']}"]['summary']['dmc|dmc|greedy']['metrics']['win_rate']
            v=100*metric['estimate'];lo,hi=[100*t for t in metric['ci95']]
            axes[0].errorbar(x,v,yerr=[[v-lo],[hi-v]],color=col,capsize=3,fmt='none',alpha=.7)
        vals=[100*v for v in r['group_delta']]
        axes[1].scatter(vals,[i+((j%7)-3)*.035 for j in range(len(vals))],s=19,color=col,alpha=.55)
        lo,hi=r['ci95_delta_pp'];v=r['delta_pp']
        axes[1].errorbar(v,i+.23,xerr=[[v-lo],[hi-v]],fmt='D',color=col,capsize=4,ms=5)
        axes[1].text(-97,i+.31,f'{v:+.1f} pp [{lo:+.1f}, {hi:+.1f}]',color=col,fontsize=8)
    axes[0].axhline(50,color='#888888',ls='--',lw=1)
    axes[0].set(xlim=(-.25,1.25),ylim=(0,100),xticks=[0,1],xticklabels=['DMC control','DMC + auxiliary'],
        ylabel='Win rate vs greedy (%)',title='A  26 development deal groups')
    axes[0].legend(title='Paired initialization',loc='upper left')
    axes[1].axvline(0,color='#888888',ls='--',lw=1)
    axes[1].set(xlim=(-100,100),ylim=(-.5,2.7),yticks=range(3),yticklabels=[str(r['seed']) for r in dev],
        xlabel='Auxiliary minus control (percentage points)',title='B  Paired deal differences')
    axes[1].text(.02,.01,'Dots: original deals; diamonds: mean and 95% interval',transform=axes[1].transAxes,fontsize=7.5)
    fig.suptitle('600 new hands per arm; identical inherited teacher weights',fontsize=12,fontweight='bold')
    exports=export_figure_bundle(fig,'paired-development',output,contract,[csvpath,audit_path],Path(__file__),include_pdf=False,include_tiff=False,backend='python');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    val=[r for r in a['paired'] if r['validation']]
    for i,r in enumerate(val):
        for arm,dy,col,marker in [('control',-.12,'#777777','o'),('aux',.12,'#AA4499','s')]:
            metric=a['evaluations'][f'validation-{arm}']['summary'][r['matchup']]['metrics']['win_rate']
            lo,hi=[100*v for v in metric['ci95']];v=100*metric['estimate']
            axes[0].errorbar(v,i+dy,xerr=[[v-lo],[hi-v]],fmt=marker,color=col,capsize=4,label=arm if i==0 else None)
        vals=[100*v for v in r['group_delta']]
        axes[1].scatter(vals,[i+((j%9)-4)*.025 for j in range(len(vals))],s=16,color='#777777',alpha=.4)
        lo,hi=r['ci95_delta_pp'];v=r['delta_pp']
        axes[1].errorbar(v,i+.22,xerr=[[v-lo],[hi-v]],fmt='D',color='#AA4499',capsize=4)
        axes[1].text(-97,i+.31,f'{v:+.1f} pp [{lo:+.1f}, {hi:+.1f}]',fontsize=8)
    labels=[r['matchup'].split('|')[-1] for r in val]
    axes[0].axvline(50,color='#888888',ls='--',lw=1)
    axes[0].set(xlim=(0,100),ylim=(-.5,2.7),yticks=range(3),yticklabels=labels,xlabel='Win rate (%)',title='A  Fixed candidates, 65 new deal groups')
    axes[0].legend(loc='upper right')
    axes[1].axvline(0,color='#888888',ls='--',lw=1)
    axes[1].set(xlim=(-100,100),ylim=(-.5,2.7),yticks=range(3),yticklabels=labels,xlabel='Auxiliary minus control (percentage points)',title='B  Conditional paired 95% intervals')
    axes[1].text(.02,.01,'Dots: original deals; diamonds: mean and 95% interval',transform=axes[1].transAxes,fontsize=7.5)
    fig.suptitle('Independent validation: preregistered initialization 314380',fontsize=12,fontweight='bold')
    valexports=export_figure_bundle(fig,'paired-validation',output,contract,[csvpath,audit_path],Path(__file__),include_pdf=False,include_tiff=False,backend='python');plt.close(fig)
    raw=sorted(root.glob('evaluations/*/results.jsonl'))
    trace=dict(matplotlib=matplotlib.__version__,group_rows=len(rows),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),source_sha256={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in [csvpath]+raw},
        exports={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()},
        figures=dict(development=exports,validation=valexports))
    (output/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),group_rows=len(rows)),ensure_ascii=False))

if __name__=='__main__':main()
