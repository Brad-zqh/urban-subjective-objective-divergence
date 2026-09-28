# Academic Figure Skill Asset Confirmation (verified against existing project figures)
# (a-t) fold distributions -> legacy Nature density/interval grammar -> param inherit
"""Twenty-panel modality-ablation atlas using all 32 declared folds per condition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'workstreams/RQ2_model/runs'
OUT = ROOT / 'figures/v211_modality_ablation_20panel'
OUT.mkdir(exist_ok=True)
FONT = ROOT / 'assets/fonts/nimbus-sans'
BLUE, RED, WHITE, INK, GREY = '#3B4CC0', '#B40426', '#F7F7F7', '#292D35', '#8A929E'
CONDITIONS = ['gsv_only', 'google_only', 'flickr_only', 'image_only', 'text_only']
LABELS = {'gsv_only':'GSV', 'google_only':'Google', 'flickr_only':'Flickr',
          'image_only':'Image', 'text_only':'Text'}
METRICS = [('model_train_rmse','Train RMSE'), ('model_validation_rmse','Validation RMSE'),
           ('model_outer_test_rmse','Outer-test RMSE'), ('ridge_train_rmse','Ridge baseline RMSE')]
METRIC_COLORS = ['#B40426', '#7096BE', '#3B4CC0', '#737B86']

def load_data():
    rows=[]
    for condition in CONDITIONS:
        fold_dir=RUNS/f'v211_modality_ablation_{condition}_r2_indexed_seed_filtered'/'folds'
        files=sorted(fold_dir.glob('*/metrics.json'))
        if len(files)!=32: raise RuntimeError(f'{condition}: expected 32 folds, found {len(files)}')
        for p in files:
            z=json.loads(p.read_text(encoding='utf-8'))
            row={'condition':condition,'fold_id':z['fold_id']}
            row.update({m:z[m] for m,_ in METRICS}); rows.append(row)
    d=pd.DataFrame(rows)
    if d[list(x[0] for x in METRICS)].isna().any().any(): raise RuntimeError('Missing metric values')
    return d

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],
        'font.size':6.2,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':5.5,'ytick.labelsize':5.5,
        'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
        'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    d=load_data()
    xlims={}
    for col,_ in METRICS:
        vals=d[col].to_numpy(float); pad=.06*(vals.max()-vals.min()); xlims[col]=(vals.min()-pad,vals.max()+pad)
    # Interval-dot strips are more legible than filled KDEs at journal width:
    # every fold remains visible, while the interval hierarchy carries the
    # distribution summary without a large opaque area.
    fig,axes=plt.subplots(5,4,figsize=(183/25.4,178/25.4),sharex='col')
    fig.subplots_adjust(left=.125,right=.985,bottom=.095,top=.935,wspace=.18,hspace=.42)
    source=[]
    for i,condition in enumerate(CONDITIONS):
        sub=d[d.condition.eq(condition)]
        for j,(col,title) in enumerate(METRICS):
            ax=axes[i,j]; v=np.sort(sub[col].to_numpy(float)); lo,hi=xlims[col]
            med=float(np.median(v)); q05,q25,q75,q95=np.quantile(v,[.05,.25,.75,.95]); color=METRIC_COLORS[j]
            jitter=np.linspace(-.055,.055,len(v))
            ax.scatter(v, jitter, s=5.2, color=color, alpha=.34, linewidth=0, zorder=2)
            ax.plot([q05,q95],[0,0], color=color, alpha=.42, lw=1.15, solid_capstyle='round', zorder=3)
            ax.plot([q25,q75],[0,0], color=color, lw=3.2, solid_capstyle='round', zorder=4)
            ax.scatter([med],[0], s=27, color=color, edgecolor='white', linewidth=.5, zorder=5)
            ax.axhline(0, color='#BAC1CB', lw=.45, zorder=1)
            ax.set_xlim(lo,hi); ax.set_ylim(-.16,.16); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.grid(False)
            ax.text(.98,.88,f'{med:.2f}\n[{q05:.2f}, {q95:.2f}]',transform=ax.transAxes,ha='right',va='top',fontsize=4.25,color=INK,linespacing=.88)
            ax.text(-.10,1.03,chr(97+i*4+j),transform=ax.transAxes,fontweight='bold',fontsize=8)
            if i==0: ax.set_title(title.replace(' RMSE',''),pad=3.5,fontsize=7)
            if j==0: ax.text(-.115,.50,LABELS[condition],transform=ax.transAxes,ha='right',va='center',fontsize=5.9)
            source.append({'condition':condition,'metric':col,'n_folds':len(v),'median':med,'q05':q05,'q25':q25,'q75':q75,'q95':q95})
    for ax in axes[-1,:]: ax.set_xlabel('RMSE')
    fig.text(.125,.965,'Point = fold  ·  thick line = IQR  ·  thin line = 90% interval',
             ha='left',va='top',fontsize=4.7,color=INK)
    stem=OUT/'Fig_v211_modality_ablation_20panel'
    fig.savefig(stem.with_suffix('.svg'),facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'),facecolor='white')
    fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white')
    Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw')
    fig.savefig(stem.with_suffix('.jpg'),dpi=600,facecolor='white',pil_kwargs={'quality':97,'subsampling':0})
    plt.close(fig)
    d.to_csv(OUT/'source_all_fold_metrics.csv',index=False); pd.DataFrame(source).to_csv(OUT/'source_panel_statistics.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_modality_ablation_20panel','panels':20,'n_folds_per_condition':32,'center':'median','intervals':'empirical 50% and 90%','display':'interval-dot strips; all fold values retained as points','delivery_formats':['jpg','pdf','png','svg','tiff'],'proxy_analysis':True,'formal':False,'backend':'python'},indent=2),encoding='utf-8')
if __name__=='__main__': main()
