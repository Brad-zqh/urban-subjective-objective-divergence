"""Nature-style 20-panel explainability flagship from V211 outer-test outputs."""
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
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered'
GEO=ROOT/'Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg/2022/chicago_tiger_2022.gpkg'
OUT=ROOT/'figures/v211_explainability_flagship_20panel_v2'; OUT.mkdir(exist_ok=True)
FONT=ROOT/'assets/fonts/nimbus-sans'
# Exact V5/V6 project palette: cobalt blue -> warm off-white -> carmine red.
BLUE,RED,WHITE,INK,GRID='#3B4CC0','#B40426','#F7F7F7','#292D35','#E3E8EF'
NAMES={1:'Green',2:'Density',3:'Access',4:'Safety',5:'Cleanliness',6:'Social',7:'Heat',8:'Air',9:'Noise',10:'Traffic'}
SHORT=NAMES.copy()
CHANNELS=[('o','objective'),('s','social'),('delta','social − objective')]

def pairs(columns):
    out=[]
    for i in range(1,11):
        for prefix,channel in CHANNELS:
            col=next((x for x in columns if x.startswith(f'{prefix}_d{i}_')),None)
            if col:
                short_channel={'objective':'O','social':'S','social − objective':'S−O'}[channel]
                out.append({'dimension':i,'channel':channel,'label':f'{NAMES[i]} · {short_channel}','column':col})
    if len(out)!=30: raise RuntimeError(f'Expected 30 coefficient fields, found {len(out)}')
    return pd.DataFrame(out)

def add_panel(ax, letter, x=-.10, y=1.04):
    ax.text(x,y,letter,transform=ax.transAxes,fontweight='bold',fontsize=8,ha='right',va='bottom')

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':5.8,'axes.titlesize':6.5,'axes.labelsize':5.9,'xtick.labelsize':4.8,'ytick.labelsize':4.8,'axes.linewidth':.52,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    cmap=mpl.colors.LinearSegmentedColormap.from_list(
        'nature_rb',[BLUE,'#8DB0D5',WHITE,'#E6A6A1',RED]
    )
    d=pd.read_parquet(RUN/'outer_test_local_coefficients_wide.parquet'); d.tract_geoid_native=d.tract_geoid_native.astype(str).str.zfill(11)
    if len(d)!=33760: raise RuntimeError(f'Unexpected coefficient row count: {len(d)}')
    meta=pairs(d.columns); stats=[]
    for r in meta.itertuples():
        v=d[r.column].to_numpy(float); stats.append({'dimension':r.dimension,'channel':r.channel,'label':r.label,'column':r.column,'median_abs':np.median(np.abs(v)),'median':np.median(v),'q05':np.quantile(v,.05),'q95':np.quantile(v,.95)})
    rank=pd.DataFrame(stats).sort_values('median_abs',ascending=False).reset_index(drop=True); top15=rank.head(15)
    global_lim=max(float(np.quantile(np.abs(rank['median']),.98)),1e-12); stat_norm=TwoSlopeNorm(vmin=-global_lim,vcenter=0,vmax=global_lim)

    g=gpd.read_file(GEO,layer='tracts_intersect_chicago',columns=['GEOID','ALAND','AWATER','geometry']).to_crs(26916); g.GEOID=g.GEOID.astype(str).str.zfill(11); g['water_share']=g.AWATER/(g.ALAND+g.AWATER); water_excluded=int((g.water_share.gt(.5)|g.ALAND.le(0)).sum()); g=g[g.water_share.le(.5)&g.ALAND.gt(0)].copy()
    d23=d[d.health_reference_year.eq(2023)]; q=d23.groupby('tract_geoid_native',as_index=False)[top15.column.tolist()].mean(); maps=g.merge(q,left_on='GEOID',right_on='tract_geoid_native',how='inner',validate='one_to_one'); xmin,ymin,xmax,ymax=maps.total_bounds; dx,dy=xmax-xmin,ymax-ymin; xmin-=.012*dx; xmax+=.012*dx; ymin-=.012*dy; ymax+=.012*dy

    fig=plt.figure(figsize=(183/25.4,213/25.4)); outer=fig.add_gridspec(5,10,height_ratios=[1.08,.95,1,1,1],left=.16,right=.975,bottom=.035,top=.965,wspace=.72,hspace=.31)

    # a: magnitude ranking, with sign encoded by the diverging fill.
    ax=fig.add_subplot(outer[0,:5]); shown=rank.head(12).iloc[::-1]; y=np.arange(len(shown)); ax.barh(y,shown.median_abs,color=[cmap(stat_norm(x)) for x in shown['median']],height=.68,edgecolor='none'); ax.set_yticks(y,shown.label); ax.tick_params(axis='y',labelsize=4.0,pad=1); ax.grid(axis='x',color=GRID,lw=.4); ax.set_axisbelow(True); ax.set_title('Coefficient magnitude · median |local coefficient|',loc='left',pad=3); add_panel(ax,'a',x=-.06)

    # b: complete-distribution intervals for the ten leading fields.
    ax=fig.add_subplot(outer[0,5:]); shown=rank.head(10).iloc[::-1]; y=np.arange(len(shown)); ax.axvspan(-1,0,color=BLUE,alpha=.035,zorder=-5); ax.axvspan(0,1,color=RED,alpha=.035,zorder=-5)
    for yi,r in enumerate(shown.itertuples()):
        color=cmap(stat_norm(r.median)); ax.hlines(yi,r.q05,r.q95,color=color,lw=1.5,alpha=.65); ax.plot(r.median,yi,'o',ms=3.4,mfc=color,mec=INK,mew=.2)
    lim=max(abs(shown.q05.min()),abs(shown.q95.max()))*1.05; ax.set_xlim(-lim,lim); ax.set_yticks(y,shown.label); ax.tick_params(axis='y',labelsize=4.0,pad=1); ax.axvline(0,color='#727B87',lw=.65); ax.grid(axis='x',color=GRID,lw=.4); ax.set_axisbelow(True); ax.set_title('Coefficient stability · tract–year 5th–95th percentile',loc='left',pad=3); add_panel(ax,'b',x=-.06)

    # c: objective versus social coefficient magnitude by semantic dimension.
    ax=fig.add_subplot(outer[1,:3]); wide=rank.pivot(index='dimension',columns='channel',values=['median_abs','median']); x=wide['median_abs']['objective']; yv=wide['median_abs']['social']; contrast=wide['median']['social − objective']; lim=max(x.max(),yv.max())*1.12; ax.plot([0,lim],[0,lim],ls='--',lw=.6,color='#8A929D')
    offsets={1:(-29,-11),2:(4,2),3:(4,12),4:(-34,4),5:(4,-2),6:(-34,-10),7:(4,2),8:(4,-5),9:(-24,10),10:(4,3)}
    for i in range(1,11):
        ax.scatter(x[i],yv[i],s=18,c=[cmap(stat_norm(contrast[i]))],edgecolor=INK,lw=.25,zorder=3)
        ax.annotate(SHORT[i],(x[i],yv[i]),xytext=offsets[i],textcoords='offset points',fontsize=3.55,va='center',ha='left',arrowprops={'arrowstyle':'-','lw':.25,'color':'#7C8490'} if abs(offsets[i][0])>20 else None)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.grid(color=GRID,lw=.35); ax.set_axisbelow(True); ax.text(.98,.035,'Objective median |coefficient| →',transform=ax.transAxes,ha='right',va='bottom',fontsize=4.0,color='#525A65'); ax.text(.035,.98,'Social median |coefficient| →',transform=ax.transAxes,ha='left',va='top',rotation=90,fontsize=4.0,color='#525A65'); ax.set_title('Objective–social coefficient magnitude',loc='left',pad=3); add_panel(ax,'c',x=-.10)

    # d: five years × three channels, retaining all outer-test rows in each median.
    ax=fig.add_subplot(outer[1,3:7]); years=[int(x) for x in sorted(d.health_reference_year.dropna().astype(int).unique())]; cols=[]; arr=[]
    for i in range(1,11):
        row=[]
        for channel in ['objective','social','social − objective']:
            col=meta[(meta.dimension.eq(i))&(meta.channel.eq(channel))].column.iloc[0]
            for yr in years: row.append(float(d.loc[d.health_reference_year.eq(yr),col].median()))
        arr.append(row)
    for channel in ['Obj.','Social','Contrast']:
        for yr in years: cols.append(f'{channel} {str(yr)[-2:]}')
    arr=np.asarray(arr); lim=max(float(np.quantile(np.abs(arr),.98)),1e-12); norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); im=ax.imshow(arr,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest'); ax.set_yticks(np.arange(10),[SHORT[i] for i in range(1,11)]); ax.set_xticks(np.arange(len(cols)),[str(yr)[-2:] for _ in range(3) for yr in years],fontsize=3.7); ax.tick_params(length=0,pad=1); ax.axvline(len(years)-.5,color='white',lw=1.4); ax.axvline(2*len(years)-.5,color='white',lw=1.4); ax.set_title('Temporal signatures by information channel',loc='left',pad=11); ax.text(.165,1.01,'Objective',transform=ax.transAxes,ha='center',fontsize=4.1); ax.text(.500,1.01,'Social',transform=ax.transAxes,ha='center',fontsize=4.1); ax.text(.835,1.01,'Social − objective',transform=ax.transAxes,ha='center',fontsize=4.1); add_panel(ax,'d',x=-.07,y=1.09); cax=ax.inset_axes([1.015,.08,.025,.84]); cb=fig.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=3.6,length=.8,pad=.3); cb.outline.set_linewidth(.25)

    # e: cross-dimensional structure in social–objective contrast coefficients.
    ax=fig.add_subplot(outer[1,7:]); delta_cols=[meta[(meta.dimension.eq(i))&(meta.channel.eq('social − objective'))].column.iloc[0] for i in range(1,11)]; corr=d[delta_cols].corr().to_numpy(); im=ax.imshow(corr,aspect='equal',cmap=cmap,norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),interpolation='nearest'); ax.set_xticks([]); ax.set_yticks(np.arange(10),[SHORT[i] for i in range(1,11)],fontsize=3.4); ax.tick_params(length=0); ax.set_title('Contrast-coefficient correlation',loc='left',pad=3); add_panel(ax,'e',x=-.08); cax=ax.inset_axes([1.02,.08,.035,.84]); cb=fig.colorbar(im,cax=cax); cb.set_ticks([-1,0,1]); cb.ax.tick_params(labelsize=3.6,length=.8,pad=.3); cb.outline.set_linewidth(.25)

    # f–t: fifteen leading 2023 spatial fields, with independent aligned colorbars.
    map_audit=[]
    for k,r in enumerate(top15.itertuples()):
        rr,cc=divmod(k,5); slot=outer[rr+2,cc*2:(cc+1)*2].subgridspec(1,2,width_ratios=[1,.04],wspace=.035); ax=fig.add_subplot(slot[0,0]); cax=fig.add_subplot(slot[0,1]); v=maps[r.column].to_numpy(float); lim=max(float(np.quantile(np.abs(v),.98)),1e-12); norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); maps.plot(column=r.column,ax=ax,cmap=cmap,norm=norm,edgecolor='white',linewidth=.035,missing_kwds={'color':'#D8DDE4'}); ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal',adjustable='box'); ax.set_axis_off(); ax.set_title(r.label.replace(' · ','\n'),fontsize=4.65,pad=.4,linespacing=.86); ax.text(.01,1.005,chr(102+k),transform=ax.transAxes,fontweight='bold',fontsize=7.2); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax); cb.ax.tick_params(labelsize=2.7,length=.7,pad=.2); cb.outline.set_linewidth(.22); map_audit.append({'panel':chr(102+k),'label':r.label,'column':r.column,'n_tracts':len(maps),'display_limit_abs_q98':lim})

    stem=OUT/'Fig_v211_explainability_flagship_20panel_v2'; fig.savefig(stem.with_suffix('.pdf'),facecolor='white'); fig.savefig(stem.with_suffix('.svg'),facecolor='white'); fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    rank.to_csv(OUT/'source_all_30_coefficient_summary.csv',index=False); pd.DataFrame(map_audit).to_csv(OUT/'source_map_panel_audit.csv',index=False); pd.DataFrame(arr,index=[NAMES[i] for i in range(1,11)],columns=cols).to_csv(OUT/'source_temporal_channel_matrix.csv'); pd.DataFrame(corr,index=[NAMES[i] for i in range(1,11)],columns=[NAMES[i] for i in range(1,11)]).to_csv(OUT/'source_contrast_correlation.csv')
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_explainability_flagship_20panel_v2','panels':20,'outer_test_rows':len(d),'coefficient_fields':30,'years':years,'map_panels':15,'map_year':2023,'map_extent':'identical','individual_colorbars':True,'water_dominant_excluded':water_excluded,'semantic_labels_no_dimension_codes':True,'proxy_analysis':True,'formal':False},indent=2),encoding='utf-8')

if __name__=='__main__': main()
