# Academic Figure Skill Asset Confirmation (verified against the legacy publication figure)
# (a) ranked importance -> grouped horizontal bar grammar -> visual adapt
# (b) local coefficient distribution -> beeswarm grammar -> visual adapt
# (c-t) coefficient maps -> choropleth small-multiple grammar -> visual adapt
"""Asymmetric 20-panel V211 explainability composite from current proxy outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered'
GEO=ROOT/'Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg/2022/chicago_tiger_2022.gpkg'
OUT=ROOT/'figures/v211_explainability_composite_20panel'; OUT.mkdir(exist_ok=True)
FONT=ROOT/'assets/fonts/nimbus-sans'
BLUE, RED, WHITE, INK, GRID='#3B4CC0','#B40426','#F7F7F7','#292D35','#E5E9EF'
DIMENSION_NAMES={
    1:'Green',
    2:'Density',
    3:'Access',
    4:'Safety',
    5:'Cleanliness',
    6:'Social',
    7:'Heat',
    8:'Air',
    9:'Noise',
    10:'Traffic',
}

def feature_pairs(columns):
    pairs=[]
    for i in range(1,11):
        for prefix,channel in [('o','O'),('s','S'),('delta','S−O')]:
            c=next((x for x in columns if x.startswith(f'{prefix}_d{i}_')),None)
            if c: pairs.append((f'{DIMENSION_NAMES[i]} · {channel}',c,c))
    return pairs

def short_label(label):
    return label

def map_label(label):
    # A deliberate line break preserves semantic labels without reducing them to D-codes.
    return label.replace(' · ', '\n')

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':6.0,'axes.titlesize':7,'axes.labelsize':6.4,'xtick.labelsize':5.2,'ytick.labelsize':5.2,'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    d=pd.read_parquet(RUN/'outer_test_local_coefficients_wide.parquet'); d.tract_geoid_native=d.tract_geoid_native.astype(str).str.zfill(11)
    pairs=feature_pairs(d.columns); ranking=[]
    for label,c,_ in pairs:
        v=d[c].to_numpy(float); ranking.append({'label':label,'column':c,'median_abs_coefficient':float(np.median(np.abs(v))),'median_coefficient':float(np.median(v)),'q05':float(np.quantile(v,.05)),'q95':float(np.quantile(v,.95))})
    rank=pd.DataFrame(ranking).sort_values('median_abs_coefficient',ascending=False).reset_index(drop=True); top18=rank.head(18)
    g=gpd.read_file(GEO,layer='tracts_intersect_chicago',columns=['GEOID','ALAND','AWATER','geometry']).to_crs(26916); g.GEOID=g.GEOID.astype(str).str.zfill(11); g['water_share']=g.AWATER/(g.ALAND+g.AWATER); water_excluded=int((g.water_share.gt(.5)|g.ALAND.le(0)).sum()); g=g[g.water_share.le(.5)&g.ALAND.gt(0)].copy()
    d23=d[d.health_reference_year.eq(2023)]; q=d23.groupby('tract_geoid_native',as_index=False)[top18.column.tolist()].mean(); m=g.merge(q,left_on='GEOID',right_on='tract_geoid_native',how='inner',validate='one_to_one'); xmin,ymin,xmax,ymax=m.total_bounds; dx,dy=xmax-xmin,ymax-ymin; xmin-=.015*dx; xmax+=.015*dx; ymin-=.015*dy; ymax+=.015*dy
    cmap=mpl.colors.LinearSegmentedColormap.from_list('nature_redblue',[(0.00,BLUE),(0.28,'#699BCB'),(0.47,'#D5E3EF'),(0.50,WHITE),(0.53,'#F1D6D1'),(0.72,'#D96C63'),(1.00,RED)])
    fig=plt.figure(figsize=(183/25.4,176/25.4)); gs=fig.add_gridspec(4,6,height_ratios=[1.30,1,1,1],left=.18,right=.955,bottom=.035,top=.955,wspace=.30,hspace=.12)
    axa=fig.add_subplot(gs[0,:3]); axb=fig.add_subplot(gs[0,3:])
    # a: all 30 coefficient fields are used in the ranking; top 15 are legible at final size.
    shown=rank.head(15).iloc[::-1]; vals=shown.median_abs_coefficient.to_numpy(); bnorm=Normalize(vals.min(),vals.max()); colors=[cmap(bnorm(x)) for x in vals]; axa.barh(np.arange(len(shown)),vals,color=colors,height=.72,edgecolor='none'); axa.set_yticks(np.arange(len(shown)),[short_label(x) for x in shown.label]); axa.tick_params(axis='y',labelsize=4.25,pad=1.2); axa.grid(axis='x',color=GRID,lw=.45); axa.set_axisbelow(True); axa.text(-.08,1.04,'a',transform=axa.transAxes,fontweight='bold',fontsize=8); axa.set_title('Global importance · median absolute local coefficient',loc='left',pad=4)
    # b: top 10 distributions; every one of the 33,760 coefficients is retained.
    bee=rank.head(10); rng=np.random.default_rng(211)
    for yi,row in enumerate(bee.itertuples()):
        v=d[row.column].to_numpy(float); q02,q98=np.quantile(v,[.02,.98]); local_lim=max(abs(q02),abs(q98),1e-12); z=np.clip((v+local_lim)/(2*local_lim),0,1); jitter=rng.uniform(-.24,.24,len(v)); axb.scatter(v,yi+jitter,c=z,cmap=cmap,s=2.2,alpha=.24,lw=0,rasterized=True); axb.plot(np.median(v),yi,'o',ms=3.1,mfc=INK,mec='white',mew=.3,zorder=4)
    axb.axvline(0,color='#77808C',lw=.65); axb.set_yticks(np.arange(len(bee)),[short_label(x) for x in bee.label]); axb.tick_params(axis='y',labelsize=4.25,pad=1.2); axb.invert_yaxis(); axb.grid(axis='x',color=GRID,lw=.45); axb.set_axisbelow(True); axb.text(-.08,1.04,'b',transform=axb.transAxes,fontweight='bold',fontsize=8); axb.set_title('Tract-year local coefficient distributions',loc='left',pad=4)
    cax=inset_axes(axb,width='2.7%',height='62%',loc='center right',bbox_to_anchor=(.075,0,1,1),bbox_transform=axb.transAxes,borderpad=0); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax); cb.set_ticks([0,1]); cb.set_ticklabels(['Low','High']); cb.ax.tick_params(labelsize=4.2,length=1,pad=1); cb.set_label('Within-feature rank',fontsize=4.7,labelpad=1.5)
    map_stats=[]
    for k,row in enumerate(top18.itertuples()):
        i,j=divmod(k,6); slot=gs[i+1,j].subgridspec(1,2,width_ratios=[1,.042],wspace=.035); ax=fig.add_subplot(slot[0,0]); cbar_ax=fig.add_subplot(slot[0,1]); v=m[row.column].to_numpy(float); lim=max(float(np.quantile(np.abs(v),.98)),1e-10); norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); m.plot(column=row.column,ax=ax,cmap=cmap,norm=norm,edgecolor='white',linewidth=.035,missing_kwds={'color':'#D9DDE3'}); ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal',adjustable='box'); ax.set_axis_off(); ax.set_title(map_label(row.label),fontsize=4.8,pad=.5,linespacing=.88); ax.text(.01,1.005,chr(99+k),transform=ax.transAxes,fontweight='bold',fontsize=7.1)
        cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cbar_ax); cbar.ax.tick_params(labelsize=2.75,length=.8,pad=.25); cbar.outline.set_linewidth(.22)
        map_stats.append({'panel':chr(99+k),'label':row.label,'column':row.column,'display_limit_abs_q98':lim,'n_tracts':len(m)})
    stem=OUT/'Fig_v211_explainability_composite_20panel'; fig.savefig(stem.with_suffix('.pdf'),facecolor='white'); fig.savefig(stem.with_suffix('.svg'),facecolor='white'); fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    rank.to_csv(OUT/'source_all_30_coefficient_rankings.csv',index=False); pd.DataFrame(map_stats).to_csv(OUT/'source_map_panel_audit.csv',index=False); q.to_csv(OUT/'source_2023_scheme_averaged_top18.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_explainability_composite_20panel','panels':20,'top_bar_fields':15,'beeswarm_fields':10,'map_fields':18,'selection':'ranked by median absolute coefficient over all 33,760 outer-test rows','map_year':2023,'map_crs':'EPSG:26916','display_extent':'identical for all 18 maps','individual_colorbars':True,'water_dominant_excluded':water_excluded,'proxy_analysis':True,'formal':False},indent=2),encoding='utf-8')
if __name__=='__main__': main()

