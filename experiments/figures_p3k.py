"""Two traceable quantitative panels; Python workflow already used by this project."""
from pathlib import Path
import sys,json,math,csv
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'experiments')]
from experiments.p3e_common import read,write,digest,check
from experiments.p3k_run import OUT,paths

def main():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    check(read(OUT/'controller-audit.json')['status']=='PASS','audited inputs')
    summary=read(OUT/'summary.json');names=list(paths()[0]);lookup={(r['source'],r['model'],r['subset'],r['metric']):r for r in summary}
    folder=OUT/'outputs/figures';folder.mkdir(exist_ok=True)
    contract=dict(claim='Centered auxiliary gradient direction varies, while median weighted norm is below DMC across the fixed multi-candidate subsets.',
        role='fixed-weight mechanism diagnostic; no causal or strength claim',archetype='quantitative grid',backend='Python matplotlib',
        panels=['median DMC/centered cosine','median weighted centered norm ratio'],subset='multi-candidate only; 8 to 18 observations/source',
        normalization='cosine [-1,1]; log2(median ratio), symmetric around ratio=1; no pseudocounts',
        missing='gray NA; zero ratios separately labeled 0; no clipping',
        export=dict(inches=[12.4,6.8],dpi=300,formats=['PNG','editable SVG']),
        review_risks=['teacher source rows duplicated','small selected samples','singleton exclusion explicitly labeled','gradient ratios are not update ratios'])
    (folder/'figure-contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    data=[];cos=np.zeros((9,9));ratio=np.zeros((9,9));counts=[]
    for i,source in enumerate(names):
        counts.append(lookup[source,names[0],'multi','centered_cosine']['states'])
        for j,model in enumerate(names):
            c=lookup[source,model,'multi','centered_cosine'];r=lookup[source,model,'multi','centered_weighted_norm_ratio']
            cos[i,j]=c['median'] if c['median'] is not None else np.nan
            ratio[i,j]=r['median'] if r['median'] is not None else np.nan
            data.append(dict(source=source,model=model,states=c['states'],cosine_valid=c['valid'],median_cosine=c['median'],median_weighted_ratio=r['median']))
    with (OUT/'outputs/tables/figure-data.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    positive=ratio[np.isfinite(ratio)&(ratio>0)];limit=max(1,math.ceil(float(np.abs(np.log2(positive)).max())))
    colorratio=np.full(ratio.shape,np.nan);valid=np.isfinite(ratio)&(ratio>0);colorratio[valid]=np.log2(ratio[valid])
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.4,6.8),gridspec_kw={'wspace':.43})
    fig.subplots_adjust(left=.12,right=.88,top=.83,bottom=.23)
    cmap=mpl.colormaps['RdBu_r'].copy();cmap.set_bad('#d9d9d9')
    labels=[x.replace('teacher-314','T').replace('absolute-314','A').replace('centered-314','C') for x in names]
    for ix,(ax,values,span,title) in enumerate(zip(axes,(cos,colorratio),(1,limit),('a  Median gradient cosine','b  Median weighted norm ratio'))):
        img=ax.imshow(values,cmap=cmap,vmin=-span,vmax=span,aspect='auto')
        ax.set_title(title,loc='left',fontweight='bold',pad=10)
        ax.set_xticks(range(9),labels,rotation=45,ha='right');ax.set_yticks(range(9),[f'{l}  (n={n})' for l,n in zip(labels,counts)])
        ax.set_xlabel('Fixed model weights');ax.set_ylabel('Trajectory source' if ix==0 else '')
        for i in range(9):
            for j in range(9):
                value=values[i,j];text='NA' if not np.isfinite(value) else (f'{cos[i,j]:+.2f}' if ix==0 else f'{100*ratio[i,j]:.1f}%')
                if ix==1 and ratio[i,j]==0:text='0'
                ax.text(j,i,text,ha='center',va='center',fontsize=7.2,color='white' if np.isfinite(value) and abs(value)>span*.55 else '#182026')
        cbar=fig.colorbar(img,ax=ax,fraction=.04,pad=.025)
        if ix==0:cbar.set_ticks([-1,-.5,0,.5,1]);cbar.set_label('Cosine (0 = orthogonal)')
        else:
            ticks=[-limit,-limit/2,0,limit/2,limit];cbar.set_ticks(ticks);cbar.set_ticklabels([f'{2**x:.3g}' for x in ticks]);cbar.set_label('Ratio to DMC norm; log2 color (1 = equal)',fontsize=8)
    fig.suptitle('P3k | Centered auxiliary gradient: direction and relative scale',x=.12,ha='left',y=.97,fontsize=15,fontweight='bold')
    fig.text(.12,.91,'Complete legal candidates; auxiliary weight = 0.1; multi-candidate states only',fontsize=10)
    fig.text(.12,.10,'T = teacher start; A = absolute final; C = centered final. n = selected observations per source.',fontsize=9)
    fig.text(.12,.061,'The three T source rows use identical selected trajectories. Fixed-weight diagnostics, not independent trials or Adam updates.',fontsize=8.5)
    for ext in ('png','svg'):fig.savefig(folder/f'gradient-map.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    trace=dict(status='RENDERED_PENDING_VISUAL_QA',matplotlib=mpl.__version__,numpy=np.__version__,
        summary_sha256=digest(OUT/'summary.json'),data_sha256=digest(OUT/'outputs/tables/figure-data.csv'),script_sha256=digest(Path(__file__)),
        cosine_limits=[-1,1],log2_ratio_limits=[-limit,limit],ratio_reference=1,missing=int(np.isnan(ratio).sum()),zero=int((ratio==0).sum()),
        clipped=0,files={p.name:digest(p) for p in folder.glob('gradient-map.*')})
    (folder/'trace.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='RENDERED_PENDING_VISUAL_QA',cells=len(data),ratio_range=[float(positive.min()),float(positive.max())])))

if __name__=='__main__':main()
