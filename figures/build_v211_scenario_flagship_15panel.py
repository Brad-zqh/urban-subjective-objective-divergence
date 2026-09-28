# Academic Figure Skill Asset Confirmation (verified against the legacy scenario figure)
# (a) scenario summary -> forest/lollipop grammar -> visual adapt
# (b-h) response distributions -> interval-density grammar -> visual adapt
# (i-o) spatial effects -> choropleth small multiples -> visual adapt
"""Fifteen-panel V211 scenario sensitivity flagship figure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'workstreams/RQ2_model/runs'; OUT=ROOT/'figures/v211_scenario_flagship_15panel'; OUT.mkdir(exist_ok=True); FONT=ROOT/'assets/fonts/nimbus-sans'
SCEN=[('D1_green_water_plus_1sd','Green/blue space +1 SD'),('D5_clean_maintenance_minus_1sd','Cleanliness/upkeep −1 SD'),('D7_heat_mitigation_minus_1sd','Heat exposure −1 SD'),('D8_air_pollution_minus_1sd','Air pollution −1 SD'),('D9_noise_mitigation_minus_1sd','Noise exposure −1 SD'),('D10_traffic_pressure_minus_1sd','Traffic pressure −1 SD'),('joint_six_dimensions','Joint six-action package')]
SHORT=['Green/blue space','Cleanliness/upkeep','Heat exposure','Air pollution','Noise exposure','Traffic pressure','Joint package']
BLUE,RED,WHITE,INK,GREY='#3B4CC0','#B40426','#F7F7F7','#292D35','#7C8490'

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':6.0,'axes.titlesize':6.4,'axes.labelsize':6.4,'xtick.labelsize':5.2,'ytick.labelsize':5.2,'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    s=pd.read_parquet(RUN/'v211_compact_proxy_scenarios_r3_indexed_seed_filtered/outer_test_scenario_predictions.parquet'); p=pd.read_parquet(RUN/'v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered/outer_test_predictions.parquet',columns=['fold_id','numeric_row','tract_geoid_native','health_reference_year']); p.tract_geoid_native=p.tract_geoid_native.astype(str).str.zfill(11)
    if len(s)!=236320 or s.fold_id.nunique()!=32: raise RuntimeError(f'Unexpected scenario structure: {s.shape}, folds={s.fold_id.nunique()}')
    joined=s.merge(p,on=['fold_id','numeric_row'],how='left',validate='many_to_one'); missing=int(joined.tract_geoid_native.isna().sum());
    if missing: raise RuntimeError(f'{missing} scenario rows lack tract-year mapping')
    gpath=ROOT/'Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg/2022/chicago_tiger_2022.gpkg'; g=gpd.read_file(gpath,layer='tracts_intersect_chicago',columns=['GEOID','ALAND','AWATER','geometry']).to_crs(26916); g.GEOID=g.GEOID.astype(str).str.zfill(11); g['water_share']=g.AWATER/(g.ALAND+g.AWATER); excluded=int((g.water_share.gt(.5)|g.ALAND.le(0)).sum()); g=g[g.water_share.le(.5)&g.ALAND.gt(0)].copy()
    q=joined[joined.health_reference_year.eq(2023)].groupby(['scenario','tract_geoid_native'],as_index=False).prediction_difference.mean(); frames={sc:g.merge(q[q.scenario.eq(sc)],left_on='GEOID',right_on='tract_geoid_native',how='inner',validate='one_to_one') for sc,_ in SCEN}; bb=np.array([x.total_bounds for x in frames.values()]); xmin,ymin,xmax,ymax=bb[:,0].min(),bb[:,1].min(),bb[:,2].max(),bb[:,3].max(); dx,dy=xmax-xmin,ymax-ymin; xmin-=.015*dx; xmax+=.015*dx; ymin-=.015*dy; ymax+=.015*dy
    cmap=mpl.colors.LinearSegmentedColormap.from_list('nature_rb',[BLUE,'#8DB0D5',WHITE,'#E6A6A1',RED]); allmap=np.concatenate([x.prediction_difference.to_numpy() for x in frames.values()]); maplim=max(float(np.quantile(np.abs(allmap),.98)),1e-9); mapnorm=TwoSlopeNorm(vmin=-maplim,vcenter=0,vmax=maplim)
    fig=plt.figure(figsize=(183/25.4,145/25.4)); gs=fig.add_gridspec(3,7,height_ratios=[1.12,.82,.78],left=.135,right=.955,bottom=.045,top=.955,wspace=.31,hspace=.18)
    # a: fold-level summary, so the independent unit shown is the declared outer fold.
    ax=fig.add_subplot(gs[0,:]); fold=s.groupby(['scenario','fold_id'],as_index=False).prediction_difference.mean(); rows=[]
    for yi,(sc,label) in enumerate(SCEN):
        v=fold[fold.scenario.eq(sc)].prediction_difference.to_numpy(); q025,mean,q975=np.quantile(v,[.025,.5,.975]); color=RED if mean>=0 else BLUE; ax.hlines(yi,q025,q975,color=color,lw=2.4,alpha=.72); ax.scatter(v,np.full(len(v),yi),s=9,color=color,alpha=.32,lw=0); ax.plot(mean,yi,'o',ms=4.8,mfc=color,mec='white',mew=.45); ax.text(q975,yi,f'  {mean:+.3f}',ha='left',va='center',fontsize=5.3,color=color); rows.append({'scenario':sc,'n_folds':len(v),'median_fold_mean':mean,'q025_fold_mean':q025,'q975_fold_mean':q975})
    ax.axvline(0,color=GREY,lw=.7); ax.set_yticks(np.arange(7),SHORT); ax.invert_yaxis(); ax.grid(axis='x',color='#E5E9EF',lw=.45); ax.set_axisbelow(True); ax.text(-.06,1.04,'a',transform=ax.transAxes,fontweight='bold',fontsize=8)
    # b-h: complete outer-test distributions with 32 fold means shown at the baseline.
    dlim=max(float(np.quantile(np.abs(s.prediction_difference),.995)),1e-9)
    for j,(sc,label) in enumerate(SCEN):
        ax=fig.add_subplot(gs[1,j]); sub=s[s.scenario.eq(sc)]; v=sub.prediction_difference.to_numpy(); x=np.linspace(-dlim,dlim,280); y=gaussian_kde(v)(x); med=np.median(v); q05,q25,q75,q95=np.quantile(v,[.05,.25,.75,.95]); color=RED if med>=0 else BLUE; ax.fill_between(x,0,y,color=color,alpha=.22,lw=0); ax.plot(x,y,color=color,lw=1.25); ax.axvline(0,color=GREY,lw=.55,ls='--'); ax.axvspan(q05,q95,ymin=.01,ymax=.05,color=color,alpha=.35,lw=0); ax.axvspan(q25,q75,ymin=.01,ymax=.075,color=color,alpha=.80,lw=0); fv=sub.groupby('fold_id').prediction_difference.mean().to_numpy(); ax.vlines(fv,-.06*y.max(),0,color=INK,lw=.45,alpha=.6); ax.plot(med,0,'o',ms=2.8,mfc=INK,mec='white',mew=.3); ax.set_xlim(-dlim,dlim); ax.set_ylim(-.08*y.max(),1.14*y.max()); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_title(SHORT[j].replace('/','/\n') if j in (0,1) else SHORT[j],pad=1,fontsize=5.3,linespacing=.88); ax.text(.98,.86,f'{med:+.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=4.6,color=color); ax.text(-.12,1.02,chr(98+j),transform=ax.transAxes,fontweight='bold',fontsize=7.3); ax.grid(False); ax.set_xlabel('Prediction change (pp)' if j==3 else '')
    # i-o: spatial distribution of the same seven strategies.
    map_audit=[]
    for j,(sc,label) in enumerate(SCEN):
        # Bind the colorbar to the final map Axes, not to the unshrunk GridSpec
        # cell.  This keeps colorbar and map heights identical after the equal-
        # aspect constraint has reduced the map's physical height.
        ax=fig.add_subplot(gs[2,j]); fm=frames[sc]
        fm.plot(column='prediction_difference',ax=ax,cmap=cmap,norm=mapnorm,edgecolor='white',linewidth=.035,missing_kwds={'color':'#D9DDE3'})
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal',adjustable='box'); ax.set_axis_off()
        ax.set_title(SHORT[j].replace('/','/\n') if j in (0,1) else SHORT[j],pad=1,fontsize=5.2,linespacing=.88)
        ax.text(.01,1.005,chr(105+j),transform=ax.transAxes,fontweight='bold',fontsize=7.3)
        cax=ax.inset_axes([1.025,0,.035,1],transform=ax.transAxes)
        cb=fig.colorbar(mpl.cm.ScalarMappable(norm=mapnorm,cmap=cmap),cax=cax)
        cb.ax.tick_params(labelsize=2.8,length=.8,pad=.2); cb.outline.set_linewidth(.22)
        map_audit.append({'scenario':sc,'n_tracts':len(fm),'display_limit':maplim,'colorbar_height_fraction_of_map_axes':1.0})
    stem=OUT/'Fig_v211_scenario_flagship_15panel'; fig.savefig(stem.with_suffix('.pdf')); fig.savefig(stem.with_suffix('.svg')); fig.savefig(stem.with_suffix('.png'),dpi=600); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'source_fold_summary.csv',index=False); pd.DataFrame(map_audit).to_csv(OUT/'source_map_audit.csv',index=False); q.to_csv(OUT/'source_2023_spatial_effects.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_scenario_flagship_15panel','panels':15,'scenario_rows':len(s),'n_outer_folds':32,'summary_interval':'empirical 2.5th–97.5th percentile of fold means','maps':'2023 scheme-average by tract','map_extent':'identical','individual_colorbars':True,'water_dominant_excluded':excluded,'interpretation':'model-projected sensitivity; not causal policy effect','proxy_analysis':True,'formal':False},indent=2),encoding='utf-8')
if __name__=='__main__': main()
