# Academic Figure Skill Asset Confirmation (verified against existing project figures)
# (a-p) perturbation distributions -> density/interval production grammar -> param inherit
"""Nature-style 16-panel graph-perturbation atlas using all outer-test rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import gaussian_kde
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_graph_perturbation_r3_indexed_seed_filtered/outer_test_graph_perturbations.parquet'
OUT=ROOT/'figures/v211_graph_perturbation_nature_16panel'; OUT.mkdir(exist_ok=True)
FONT=ROOT/'assets/fonts/nimbus-sans'
BLUE, RED, WHITE, INK, GREY='#3B4CC0','#B40426','#F7F7F7','#292D35','#8A929E'
REL=[('spatial_queen','Spatial contiguity'),('road_connectivity','Road connectivity'),('mobility_flow','Mobility flow'),('temporal_forward','Temporal links')]
COLS=[('relation_drop','prediction_difference','Drop: signed Δ'),('edge_weight_half','prediction_difference','Half weight: signed Δ'),('relation_drop','absolute_difference','Drop: |Δ|'),('edge_weight_half','absolute_difference','Half weight: |Δ|')]

def safe_density(values, x):
    """Return a display-only curve without perturbing constant observations."""
    values=np.asarray(values,float)
    if np.unique(values).size < 2 or np.std(values) <= 1e-12:
        width=max((x[-1]-x[0])*.012,1e-8)
        return np.exp(-.5*((x-values[0])/width)**2), 'point_mass'
    return gaussian_kde(values)(x), 'kde'

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':6.2,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':5.4,'ytick.labelsize':5.4,'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    d=pd.read_parquet(DATA); d=d[d.relation.ne('all_active') & d.perturbation.ne('baseline')].copy()
    expected={(r,p) for r,_ in REL for p in ['relation_drop','edge_weight_half']}; actual=set(zip(d.relation,d.perturbation)); missing=expected-actual
    if missing: raise RuntimeError(f'Missing relation/perturbation cells: {missing}')
    fig,axes=plt.subplots(4,4,figsize=(183/25.4,220/25.4),sharex='col'); fig.subplots_adjust(left=.13,right=.985,bottom=.08,top=.95,wspace=.15,hspace=.34)
    # Fixed limits within signed and absolute column families preserve comparability.
    lim_signed=float(np.nanquantile(np.abs(d.prediction_difference),.995)); lim_abs=float(np.nanquantile(d.absolute_difference,.995)); xlims=[(-lim_signed,lim_signed),(-lim_signed,lim_signed),(0,lim_abs),(0,lim_abs)]
    stats=[]; cmap=mpl.colors.LinearSegmentedColormap.from_list('nature_rb',[BLUE,'#8DB0D5',WHITE,'#E6A6A1',RED])
    signed_medians=[]
    for r,_ in REL:
        for p in ['relation_drop','edge_weight_half']:
            signed_medians.append(float(d[(d.relation==r)&(d.perturbation==p)].prediction_difference.median()))
    signed_lim=max(abs(min(signed_medians)),abs(max(signed_medians)),1e-12)
    snorm=mpl.colors.TwoSlopeNorm(vmin=-signed_lim,vcenter=0,vmax=signed_lim)
    for i,(relation,rlabel) in enumerate(REL):
        for j,(perturb,value,title) in enumerate(COLS):
            ax=axes[i,j]; sub=d[(d.relation==relation)&(d.perturbation==perturb)]; v=sub[value].to_numpy(float); lo,hi=xlims[j]
            # Density is evaluated on all data; extreme observations are retained in source/statistics.
            x=np.linspace(lo,hi,360); y,density_type=safe_density(v,x); med=float(np.median(v)); q05,q25,q75,q95=np.quantile(v,[.05,.25,.75,.95]); signed=float(sub.prediction_difference.median()); nz_rate=float(np.mean(np.abs(v)>1e-12)); color=cmap(snorm(signed))
            if nz_rate < .01: color=GREY
            if nz_rate < .01:
                ax.set_facecolor('#F1F4F8')
                ax.axvspan(lo,hi,color='#E8EDF3',alpha=.55,zorder=-2)
            ax.fill_between(x,0,y,color=color,alpha=.30 if nz_rate < .01 else .24,lw=0); ax.plot(x,y,color=color,lw=1.35 if nz_rate < .01 else 1.15)
            ax.axvspan(max(q05,lo),min(q95,hi),ymin=.01,ymax=.055,color=color,alpha=.30,lw=0); ax.axvspan(max(q25,lo),min(q75,hi),ymin=.01,ymax=.075,color=color,alpha=.75,lw=0)
            fold=sub.groupby('fold_id')[value].mean().sort_values().to_numpy(float); rug_y=-.075*y.max(); ax.vlines(fold,rug_y,0,color=INK,lw=.55,alpha=.70); ax.plot(med,0,'o',ms=3.4,mfc=INK,mec='white',mew=.35,zorder=5)
            if value=='prediction_difference': ax.axvline(0,color=GREY,lw=.6,ls='--',zorder=0)
            ax.set_xlim(lo,hi); ax.set_ylim(rug_y*1.3,y.max()*1.14); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.grid(False)
            note=f'median {med:+.3f}\n90% [{q05:+.3f}, {q95:+.3f}]\nn={len(v):,}; non-zero {100*nz_rate:.2f}%'
            if nz_rate < .01: note += '\npoint mass at 0'
            ax.text(.98,.88,note,transform=ax.transAxes,ha='right',va='top',fontsize=4.7)
            ax.text(-.10,1.03,chr(97+i*4+j),transform=ax.transAxes,fontweight='bold',fontsize=8)
            if i==0: ax.set_title(title,pad=4)
            if j==0: ax.text(-.11,.50,rlabel,transform=ax.transAxes,rotation=90,ha='right',va='center',fontsize=6.5)
            stats.append({'relation':relation,'perturbation':perturb,'quantity':value,'n_rows':len(v),'n_folds':sub.fold_id.nunique(),'median':med,'q05':q05,'q25':q25,'q75':q75,'q95':q95,'display_density':density_type})
    for j,ax in enumerate(axes[-1]): ax.set_xlabel('Prediction change (pp)' if j<2 else 'Absolute change (pp)')
    stem=OUT/'Fig_v211_graph_perturbation_nature_16panel'; fig.savefig(stem.with_suffix('.pdf')); fig.savefig(stem.with_suffix('.svg')); fig.savefig(stem.with_suffix('.png'),dpi=600); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    pd.DataFrame(stats).to_csv(OUT/'source_panel_statistics.csv',index=False); d.to_parquet(OUT/'source_all_outer_test_rows.parquet',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_graph_perturbation_nature_16panel','panels':16,'rows_used':len(d),'center':'median','intervals':'empirical 50% and 90%','fold_marks':'mean within each of 32 outer folds','display_limits':'common 99.5th percentile within signed/absolute families; all values retained in source data','proxy_analysis':True,'formal':False},indent=2),encoding='utf-8')
if __name__=='__main__': main()
