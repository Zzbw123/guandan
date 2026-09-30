"""Source-backed P3e descriptive plots; no model selection or new tests."""
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
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status']=='PASS'
    output=root/'descriptives';output.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
    palette=['#4477AA','#EE7733','#228833']
    group_rows=[]
    for key,job in audit['evaluations'].items():
        for matchup,summary in job['summary'].items():
            for g in summary['clusters']:group_rows.append(dict(job=key,matchup=matchup,**g))
    csvpath=output/'groups.csv'
    with csvpath.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(group_rows[0]));w.writeheader();w.writerows(group_rows)
    contract=dict(claim='Training-size effects vary across initializations; independent validation is a separate question.',
        archetype='quantitative grid',backend='python',unit='original deal, 8 paired variants; 3 training initializers',
        export='300 dpi PNG and editable SVG; source CSV plus SHA256 trace',
        review_risks=['Do not treat variants or reused deals as independent','Intervals condition on a fixed initializer',
                      'No best-seed selection','Descriptive development comparisons do not prove general strength'],
        colors=dict(zip(['314370','314371','314372'],palette)),missing_values='none permitted',
        probability_scale=[0,100],difference_scale=[-100,100],overflow='none',normalization='linear, no scalar color mapping')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for i,r in enumerate(audit['paired']):
        color=palette[i]
        axes[0].plot([400,1600],[100*r['win_rate_400'],100*r['win_rate_1600']],'-o',color=color,label=str(r['seed']))
        for x,y in zip([400,1600],[100*r['win_rate_400'],100*r['win_rate_1600']]):
            axes[0].annotate(f'{y:.1f}%',(x,y),xytext=(7,(-10,0,9)[i]),textcoords='offset points',fontsize=8,color=color)
        vals=[100*x for x in r['group_delta']]
        axes[1].scatter(vals,[i+((j%7)-3)*.04 for j in range(26)],s=18,color=color,alpha=.55)
        low,high=r['ci95_delta_pp'];mean=r['delta_pp']
        axes[1].errorbar(mean,i+.24,xerr=[[mean-low],[high-mean]],fmt='D',color=color,capsize=4,ms=5)
    axes[0].axhline(50,color='#888888',ls='--',lw=1)
    axes[0].set(xlim=(200,1950),ylim=(0,100),xticks=[400,1600],xlabel='Cumulative training hands',ylabel='Win rate vs greedy (%)',title='A  Same 26 development deals')
    axes[0].legend(title='Initialization',loc='upper left')
    axes[1].axvline(0,color='#888888',ls='--',lw=1)
    axes[1].set(xlim=(-100,100),ylim=(-.7,2.6),yticks=range(3),yticklabels=[str(r['seed']) for r in audit['paired']],xlabel='1600 minus 400 hands (percentage points)',ylabel='Initialization',title='B  Paired original-deal differences')
    axes[1].text(.02,.02,'Dots: 26 deal pairs each\nDiamonds: mean and conditional 95% interval',transform=axes[1].transAxes,fontsize=8,va='bottom')
    fig.suptitle('Fixed-budget training-size research (development only)',fontweight='bold',fontsize=12)
    exports=export_figure_bundle(fig,'training-scale',output,contract,[csvpath,audit_path],Path(__file__),include_pdf=False,include_tiff=False,backend='python')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharex=True,sharey=True,layout='constrained')
    for ax,opponent in zip(axes,('greedy','random','team')):
        summary=audit['evaluations']['validation']['summary'][f'dmc|dmc|{opponent}']
        vals=[g['win_rate'] for g in summary['clusters']];metric=summary['metrics']['win_rate']
        ax.bar([100*i/8 for i in range(9)],[vals.count(i/8) for i in range(9)],width=10,color='#4477AA',edgecolor='white')
        lo,hi=metric['ci95'];ax.text(.98,.97,f"65 original deals\nMean {100*metric['estimate']:.1f}%\n95% CI [{100*lo:.1f}, {100*hi:.1f}]%",transform=ax.transAxes,ha='right',va='top',fontsize=8)
        ax.set(title=f'vs {opponent}',xlim=(-8,108),ylim=(0,65),xticks=[0,25,50,75,100],xlabel='Win fraction across 8 variants (%)')
    axes[0].set_ylabel('Number of original deals')
    fig.suptitle('Preregistered candidate: initialization 314370, 1600 training hands',fontweight='bold',fontsize=11)
    val_exports=export_figure_bundle(fig,'validation-groups',output,contract,[csvpath,audit_path],Path(__file__),include_pdf=False,include_tiff=False,backend='python')
    plt.close(fig)
    raw_paths=sorted(root.glob('evaluations/*/results.jsonl'))
    trace=dict(matplotlib=matplotlib.__version__,script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),
        source_sha256={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in [csvpath]+raw_paths},
        exports={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()},
        figures={'development':exports,'validation':val_exports})
    (output/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'group_rows':len(group_rows)},ensure_ascii=False))
if __name__=='__main__':main()
