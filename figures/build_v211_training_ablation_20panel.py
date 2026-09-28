"""Nature-style 20-panel training and spatial-partition robustness atlas."""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_mm_gtgnnwr_r3_indexed_seed_filtered/folds'
OUT=ROOT/'figures/v211_training_ablation_20panel'; OUT.mkdir(exist_ok=True)
FONT=ROOT/'assets/fonts/nimbus-sans'
BLUE,RED,WHITE,INK,GREY,GRID='#3B4CC0','#B40426','#F7F7F7','#292D35','#7C8490','#E3E8EF'
SCHEME_ORDER=['axis_recursive','polar_north_clockwise_equal_count','rotated45_recursive','y_equal_count_stripes']
SCHEME_LABEL={'axis_recursive':'Axis-recursive','polar_north_clockwise_equal_count':'Polar sectors','rotated45_recursive':'Rotated 45°','y_equal_count_stripes':'Equal-count stripes'}
SCHEME_SHORT={'axis_recursive':'AR','polar_north_clockwise_equal_count':'PC','rotated45_recursive':'R45','y_equal_count_stripes':'EQ'}
METRICS=[('train','Train RMSE'),('validation','Validation RMSE'),('test','Outer-test RMSE'),('ridge','Ridge baseline RMSE')]

def panel(ax,letter,x=-.13,y=1.04):
    ax.text(x,y,letter,transform=ax.transAxes,fontweight='bold',fontsize=8,ha='right',va='bottom')

def density(ax,values,lo,hi,color):
    values=np.asarray(values,float); x=np.linspace(lo,hi,320)
    if np.unique(values).size>1 and np.std(values)>1e-10: y=gaussian_kde(values)(x)
    else:
        width=max((hi-lo)*.025,1e-6); y=np.exp(-.5*((x-values[0])/width)**2)
    ax.fill_between(x,0,y,color=color,alpha=.24,lw=0); ax.plot(x,y,color=color,lw=1.1)
    return y

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':5.8,'axes.titlesize':6.3,'axes.labelsize':5.9,'xtick.labelsize':4.8,'ytick.labelsize':4.8,'axes.linewidth':.52,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    rows=[]
    for path in sorted(RUN.glob('*/metrics.json')):
        z=json.loads(path.read_text(encoding='utf-8')); fold=z['fold_id']; m=re.search(r'__v(\d{4})__B(\d+)$',fold)
        if not m: raise RuntimeError(f'Unrecognized fold id: {fold}')
        rows.append({'fold':fold,'scheme':fold.split('__')[0],'vintage':m.group(1),'block':int(m.group(2)),'partition':f'{m.group(1)[-2:]}·B{m.group(2)}','train':z['model_train_rmse'],'validation':z['model_validation_rmse'],'test':z['model_outer_test_rmse'],'ridge':z['ridge_train_rmse'],'n_train':z['inner_training'],'n_validation':z['inner_validation'],'n_test':z['outer_test']})
    d=pd.DataFrame(rows)
    if len(d)!=32 or set(d.scheme)!=set(SCHEME_ORDER): raise RuntimeError(f'Expected 32 folds and four schemes, got {len(d)} / {sorted(d.scheme.unique())}')
    d['generalization_gap']=d.test-d.train; d['model_gain_over_ridge']=d.ridge-d.train
    cmap=mpl.colors.LinearSegmentedColormap.from_list('project_rb',[BLUE,'#8DB0D5',WHITE,'#E6A6A1',RED]); scheme_colors=[cmap(x) for x in (.02,.28,.72,.98)]
    fig=plt.figure(figsize=(183/25.4,215/25.4)); gs=fig.add_gridspec(5,4,height_ratios=[1.20,1,1,1,1],left=.115,right=.982,bottom=.055,top=.96,wspace=.34,hspace=.43)

    ax=fig.add_subplot(gs[0,0]); partitions=['10·B1','10·B2','10·B3','10·B4','20·B1','20·B2','20·B3','20·B4']; mat=d.pivot(index='scheme',columns='partition',values='test').reindex(index=SCHEME_ORDER,columns=partitions); vals=mat.to_numpy(); mid=float(np.nanmedian(vals)); norm=TwoSlopeNorm(vmin=float(np.nanmin(vals)),vcenter=mid,vmax=float(np.nanmax(vals))); im=ax.imshow(vals,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest'); ax.set_xticks(np.arange(8),partitions,rotation=45,ha='right',fontsize=3.7); ax.set_yticks(np.arange(4),[SCHEME_SHORT[x] for x in SCHEME_ORDER]); ax.tick_params(length=0,pad=1)
    for i in range(4):
        for j in range(8): ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',fontsize=3.15,color='white' if abs(norm(vals[i,j])-.5)>.31 else INK)
    ax.set_title('Outer-test RMSE by partition',loc='left',pad=3); panel(ax,'a',x=-.18); cax=ax.inset_axes([1.035,.08,.035,.84]); cb=fig.colorbar(im,cax=cax); cb.set_ticks([vals.min(),mid,vals.max()]); cb.ax.tick_params(labelsize=3.3,length=.7,pad=.3); cb.outline.set_linewidth(.22)

    ax=fig.add_subplot(gs[0,1]); stages=['train','validation','test']; xx=np.arange(3)
    for scheme,color in zip(SCHEME_ORDER,scheme_colors):
        sub=d[d.scheme.eq(scheme)]
        for r in sub.itertuples(): ax.plot(xx,[r.train,r.validation,r.test],color=color,alpha=.11,lw=.45)
        med=sub[stages].median().to_numpy(); ax.plot(xx,med,color=color,lw=1.35,marker='o',ms=2.9,label=SCHEME_SHORT[scheme])
    ax.set_xticks(xx,['Train','Validation','Outer test']); ax.grid(axis='y',color=GRID,lw=.35); ax.set_axisbelow(True); ax.set_title('Error propagation across stages',loc='left',pad=3); panel(ax,'b'); ax.legend(ncol=2,fontsize=3.6,handlelength=1.3,columnspacing=.7,loc='upper left')

    ax=fig.add_subplot(gs[0,2]); allv=np.r_[d.train,d.ridge]; lo,hi=allv.min()-.08,allv.max()+.08; ax.plot([lo,hi],[lo,hi],ls='--',color=GREY,lw=.6)
    for scheme,color in zip(SCHEME_ORDER,scheme_colors):
        sub=d[d.scheme.eq(scheme)]; ax.scatter(sub.ridge,sub.train,s=11,color=color,alpha=.70,edgecolor='white',lw=.25,label=SCHEME_SHORT[scheme])
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_xlabel('Ridge train RMSE'); ax.set_ylabel('Model train RMSE'); ax.grid(color=GRID,lw=.35); ax.set_axisbelow(True); ax.set_title('Model versus ridge baseline',loc='left',pad=3); panel(ax,'c'); ax.text(.97,.04,'Below parity = model gain',transform=ax.transAxes,ha='right',va='bottom',fontsize=3.8,color='#555D68')

    ax=fig.add_subplot(gs[0,3]); summary=[]
    for yi,(scheme,color) in enumerate(zip(SCHEME_ORDER,scheme_colors)):
        v=d.loc[d.scheme.eq(scheme),'generalization_gap'].to_numpy(); q05,med,q95=np.quantile(v,[.05,.5,.95]); ax.hlines(yi,q05,q95,color=color,lw=1.8,alpha=.72); ax.scatter(v,np.full(len(v),yi),s=6,color=color,alpha=.26,lw=0); ax.plot(med,yi,'o',ms=3.8,mfc=color,mec=INK,mew=.2); summary.append({'scheme':scheme,'generalization_gap_median':med,'q05':q05,'q95':q95})
    ax.axvline(0,color=GREY,lw=.65); ax.set_yticks(np.arange(4),[SCHEME_SHORT[x] for x in SCHEME_ORDER]); ax.invert_yaxis(); ax.grid(axis='x',color=GRID,lw=.35); ax.set_axisbelow(True); ax.set_xlabel('Outer test − train RMSE'); ax.set_title('Generalization gap',loc='left',pad=3); panel(ax,'d')

    xlims={}; metric_norm={}
    for col,_ in METRICS:
        v=d[col].to_numpy(); pad=.055*(v.max()-v.min()); xlims[col]=(v.min()-pad,v.max()+pad); meds=d.groupby('scheme')[col].median(); metric_norm[col]=TwoSlopeNorm(vmin=float(meds.min()),vcenter=float(meds.median()),vmax=float(meds.max()))
    source=[]; letter=4
    for i,scheme in enumerate(SCHEME_ORDER):
        sub=d[d.scheme.eq(scheme)]
        for j,(col,title) in enumerate(METRICS):
            ax=fig.add_subplot(gs[i+1,j]); v=np.sort(sub[col].to_numpy(float)); lo,hi=xlims[col]; med=float(np.median(v)); q05,q25,q75,q95=np.quantile(v,[.05,.25,.75,.95]); color=cmap(metric_norm[col](med)); y=density(ax,v,lo,hi,color); ax.axvspan(q05,q95,ymin=.015,ymax=.060,color=color,alpha=.30,lw=0); ax.axvspan(q25,q75,ymin=.015,ymax=.085,color=color,alpha=.78,lw=0); rug=-.060*y.max(); ax.vlines(v,rug,0,color=GREY,lw=.40,alpha=.62); ax.axvline(med,color=INK,lw=.65,ymax=.80); ax.plot(med,0,'o',ms=3,mfc=INK,mec='white',mew=.3,zorder=5); ax.set_xlim(lo,hi); ax.set_ylim(rug*1.35,y.max()*1.13); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.text(.98,.87,f'{med:.2f}\n[{q05:.2f}, {q95:.2f}]',transform=ax.transAxes,ha='right',va='top',fontsize=4.2)
            if i==0: ax.set_title(title,loc='left',pad=2)
            if j==0: ax.text(-.12,.50,SCHEME_LABEL[scheme],transform=ax.transAxes,rotation=90,ha='right',va='center',fontsize=5.4)
            if i==3: ax.set_xlabel('RMSE')
            panel(ax,chr(97+letter),x=-.10); letter+=1; source.append({'scheme':scheme,'metric':col,'n_folds':len(v),'median':med,'q05':q05,'q25':q25,'q75':q75,'q95':q95})

    stem=OUT/'Fig_v211_training_ablation_20panel'; fig.savefig(stem.with_suffix('.pdf'),facecolor='white'); fig.savefig(stem.with_suffix('.svg'),facecolor='white'); fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    d.to_csv(OUT/'source_fold_metrics.csv',index=False); pd.DataFrame(source).to_csv(OUT/'source_panel_statistics.csv',index=False); pd.DataFrame(summary).to_csv(OUT/'source_generalization_gap_summary.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_training_ablation_20panel','panels':20,'outer_folds':len(d),'partition_schemes':[SCHEME_LABEL[x] for x in SCHEME_ORDER],'palette':[BLUE,'#8DB0D5',WHITE,'#E6A6A1',RED],'source_data_complete':True,'proxy_analysis':True,'formal':False,'backend':'python'},indent=2),encoding='utf-8')

if __name__=='__main__': main()
