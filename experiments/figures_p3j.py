"""Two Python figures from audited paired groups; no new statistical selection."""
import csv,json,sys,statistics
from pathlib import Path
from hashlib import sha256
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_figure_export import export_figure_bundle

def main():
    root=Path(sys.argv[1]).resolve();audit_path=root/'controller-audit.json'
    a=json.loads(audit_path.read_text('utf-8'));assert a['status']=='PASS'
    out=root/'descriptives';out.mkdir();tables=out/'outputs/tables';tables.mkdir(parents=True)
    rows=[];desc={}
    for job,e in sorted(a['evaluations'].items()):
        for matchup,s in e['summary'].items():
            values=[g['win_rate'] for g in s['clusters']]
            desc[job+'/'+matchup]=dict(n=len(values),mean=statistics.mean(values),sd=statistics.stdev(values),
                median=statistics.median(values),min=min(values),max=max(values),missing=0,
                zero_groups=values.count(0),one_groups=values.count(1))
            for g in s['clusters']:rows.append(dict(job=job,matchup=matchup,**g))
    csvpath=tables/'plot_data_groups.csv'
    with csvpath.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'descriptives.json').write_text(json.dumps(desc,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#4477AA','#EE7733','#228833'];armcolors={'absolute':'#AA4499','centered':'#228833'}
    contract=dict(claim='Compare fixed centered versus absolute auxiliary targets, preserving paired uncertainty and raw deal differences.',
        archetype='quantitative grid',backend='python',unit='original deal mean over 8 symmetry variants',
        export='11 by 4.5 inch PNG and editable SVG',
        colors={'initializers':dict(zip(['314380','314381','314382'],colors)),**armcolors},
        scales={'win_percent':[0,100],'paired_difference_pp':[-100,100]},
        missing='none permitted',overflow='none; full possible range',
        intervals='95% level-stratified deal-cluster bootstrap, 5000 draws; conditional on initializer',
        risks=['26 reused development deals','65 new validation deals per matchup, not 520 independent games',
               'secondary intervals are pointwise descriptions, not multiplicity-adjusted discoveries',
               'equal environment hands do not mean equal updates or compute'])
    exports={}
    for validation in (False,True):
        subset=sorted([r for r in a['paired'] if r['validation']==validation],key=lambda r:r['matchup'] if validation else r['seed'])
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        fig.subplots_adjust(left=.08,right=.98,bottom=.16,top=.82,wspace=.26)
        for i,r in enumerate(subset):
            if validation:
                for arm,dy in [('absolute',-.12),('centered',.12)]:
                    m=a['evaluations'][f'validation-{arm}']['summary'][r['matchup']]['metrics']['win_rate']
                    v=100*m['estimate'];lo,hi=[100*x for x in m['ci95']]
                    axes[0].errorbar(v,i+dy,xerr=[[v-lo],[hi-v]],fmt='o' if arm=='absolute' else 's',capsize=3,
                        color=armcolors[arm],label=arm if i==0 else None)
                color=armcolors['centered']
            else:
                color=colors[i];axes[0].plot([0,1],[100*r['absolute'],100*r['centered']],'-o',color=color,label=str(r['seed']))
                for x,arm in enumerate(('absolute','centered')):
                    m=a['evaluations'][f"{arm}-{r['seed']}"]['summary']['dmc|dmc|greedy']['metrics']['win_rate']
                    v=100*m['estimate'];lo,hi=[100*y for y in m['ci95']]
                    axes[0].errorbar(x,v,yerr=[[v-lo],[hi-v]],fmt='none',capsize=3,color=color)
            vals=[100*v for v in r['group_delta']]
            axes[1].scatter(vals,[i+((j%9)-4)*.025 for j in range(len(vals))],s=16,color=color,alpha=.5)
            lo,hi=r['ci95_delta_pp'];v=r['delta_pp']
            axes[1].errorbar(v,i+.22,xerr=[[v-lo],[hi-v]],fmt='D',capsize=4,color=color)
            axes[1].text(-97,i+.34,f'{v:+.2f} pp [{lo:+.2f}, {hi:+.2f}]',fontsize=8)
        labels=[r['matchup'].split('|')[-1] if validation else str(r['seed']) for r in subset]
        if validation:
            axes[0].axvline(50,color='#999999',ls='--',lw=1)
            axes[0].set(xlim=(0,100),ylim=(-.5,2.8),yticks=range(3),yticklabels=labels,
                        xlabel='Win rate (%)',title='A  Fixed initialization 314380: 65 new deal groups')
            axes[0].legend(loc='upper right')
        else:
            axes[0].axhline(50,color='#999999',ls='--',lw=1)
            axes[0].set(ylim=(0,100),xlim=(-.2,1.2),xticks=[0,1],xticklabels=['Absolute auxiliary','Centered auxiliary'],
                        ylabel='Win rate vs greedy (%)',title='A  26 reused development deal groups')
            axes[0].legend(title='Paired initialization',loc='upper left')
        axes[1].axvline(0,color='#999999',ls='--',lw=1)
        axes[1].set(xlim=(-100,100),ylim=(-.5,2.8),yticks=range(3),yticklabels=labels,
                    xlabel='Centered minus absolute (percentage points)',title='B  Paired differences and conditional 95% intervals')
        axes[1].text(.02,.01,'Dots: original deals; diamonds: mean and 95% interval',transform=axes[1].transAxes,fontsize=7.5)
        name='paired-validation' if validation else 'paired-development'
        fig.suptitle('P3j: fixed 0.1 auxiliary weight; 600 new hands per arm',fontsize=12,fontweight='bold')
        exports[name]=export_figure_bundle(fig,name,out,contract,[csvpath,audit_path],Path(__file__),
            include_pdf=False,include_tiff=False,backend='python');plt.close(fig)
    trace=dict(status='GENERATED_PENDING_VISUAL_QA',group_rows=len(rows),matplotlib=matplotlib.__version__,
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),
        files={p.relative_to(out).as_posix():sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
    (out/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8');print(json.dumps(dict(group_rows=len(rows))))
if __name__=='__main__':main()
