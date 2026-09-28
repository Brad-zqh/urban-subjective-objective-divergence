"""20-panel V211 spatiotemporal atlas, ported from the legacy map grammar.

Rows are observed, predicted, residual and absolute residual; columns are five
health-reference years. Predictions are averaged over the four declared
partition schemes for each tract-year. Water-dominant polygons are excluded
from the thematic layer and documented in the audit CSV; no city background is
drawn.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import font_manager
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered'; GEO=ROOT/'Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg'; OUT=ROOT/'figures/v211_spatiotemporal_20panel'; OUT.mkdir(exist_ok=True)
BLUE='#3B4CC0'; RED='#B40426'; PALE='#F7F7F7'; INK='#30343B'; GRID='#E7EAF0'; FONT=ROOT/'assets/fonts/nimbus-sans'
YEARS=[2014,2017,2019,2021,2023]; GEOYEAR={2014:2014,2017:2017,2019:2019,2021:2019,2023:2022}

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':6.2,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'axes.facecolor':'white','figure.facecolor':'white','text.color':INK})
    d=pd.read_parquet(R/'outer_test_predictions.parquet'); d.tract_geoid_native=d.tract_geoid_native.astype(str).str.zfill(11)
    # One displayed value per tract-year: declared scheme-average, not a sampled fold.
    q=d.groupby(['tract_geoid_native','health_reference_year'],as_index=False).agg(observed=('observed_outcome','mean'),prediction=('prediction','mean'),residual=('residual','mean'),scheme_count=('partition_scheme','nunique'))
    assert q.scheme_count.eq(4).all(), q.scheme_count.value_counts().to_dict(); q['abs_residual']=q.residual.abs()
    frames=[]; audits=[]
    for year in YEARS:
        gyear=GEOYEAR[year]; gp=GEO/str(gyear)/f'chicago_tiger_{gyear}.gpkg'; g=gpd.read_file(gp,layer='tracts_intersect_chicago',columns=['GEOID','ALAND','AWATER','geometry']).to_crs(26916); g.GEOID=g.GEOID.astype(str).str.zfill(11)
        sub=q[q.health_reference_year.eq(year)].copy(); joined=g.merge(sub,left_on='GEOID',right_on='tract_geoid_native',how='inner',validate='one_to_one'); joined['water_share']=joined.AWATER/(joined.ALAND+joined.AWATER); joined['water_dominant']=joined.water_share.gt(.5); m=joined.loc[~joined.water_dominant & joined.ALAND.gt(0)].copy(); frames.append(m)
        audits.append({'health_reference_year':year,'geometry_year':gyear,'input_tract_year_rows':int(len(sub)),'joined_rows':int(len(joined)),'displayed_land_rows':int(len(m)),'excluded_water_dominant_rows':int(joined.water_dominant.sum()),'excluded_zero_land_rows':int(joined.ALAND.le(0).sum()),'city_background_drawn':False,'crs':'EPSG:26916'})
    allvals=np.concatenate([f[['observed','prediction']].to_numpy().ravel() for f in frames]); omin,omax=np.quantile(allvals,[.01,.99]); rlim=float(np.quantile(np.concatenate([f.residual.abs().to_numpy() for f in frames]),.99)); alimit=rlim
    bb=np.array([f.total_bounds for f in frames]); xmin,ymin,xmax,ymax=bb[:,0].min(),bb[:,1].min(),bb[:,2].max(),bb[:,3].max(); dx,dy=xmax-xmin,ymax-ymin; xmin-=.015*dx; xmax+=.015*dx; ymin-=.015*dy; ymax+=.015*dy
    fig,ax=plt.subplots(4,5,figsize=(183/25.4,247/25.4)); fig.subplots_adjust(left=.095,right=.975,bottom=.045,top=.945,wspace=.10,hspace=.13)
    specs=[('observed','Observed MHLTH (%)',Normalize(vmin=omin,vmax=omax),mpl.colors.LinearSegmentedColormap.from_list('obs',[BLUE,'#8DB0D5',PALE,'#E6A6A1',RED])),('prediction','V211 prediction (%)',Normalize(vmin=omin,vmax=omax),mpl.colors.LinearSegmentedColormap.from_list('pred',[BLUE,'#8DB0D5',PALE,'#E6A6A1',RED])),('residual','Observed − predicted (pp)',TwoSlopeNorm(vmin=-rlim,vcenter=0,vmax=rlim),mpl.colors.LinearSegmentedColormap.from_list('res',[BLUE,'#8DB0D5',PALE,'#E6A6A1',RED])),('abs_residual','|residual| (pp)',Normalize(vmin=0,vmax=alimit),mpl.colors.LinearSegmentedColormap.from_list('abs',[PALE,'#E6A6A1',RED]))]
    for i,(col,title,norm,cmap) in enumerate(specs):
        for j,(year,m) in enumerate(zip(YEARS,frames)):
            a=ax[i,j]; m.plot(column=col,ax=a,cmap=cmap,norm=norm,edgecolor='white',linewidth=.06,missing_kwds={'color':'#E9EDF2'}); a.set_xlim(xmin,xmax); a.set_ylim(ymin,ymax); a.set_axis_off(); a.set_aspect('equal',adjustable='box')
            if i==0:a.set_title(str(year),fontsize=7,pad=3)
            if j==0:a.text(-.08,.5,title,transform=a.transAxes,rotation=90,ha='right',va='center',fontsize=6.6); a.text(-.02,1.05,chr(97+i),transform=a.transAxes,fontweight='bold',fontsize=8)
            a.text(.96,.03,f'n={len(m):,}',transform=a.transAxes,ha='right',va='bottom',fontsize=5.2,bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':.7})
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax=inset_axes(a,width='3.2%',height='100%',loc='lower left',bbox_to_anchor=(1.025,0,1,1),bbox_transform=a.transAxes,borderpad=0); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax); cb.ax.tick_params(labelsize=3.6,length=1.2,pad=.4); cb.outline.set_linewidth(.3)
    stem=OUT/'Fig_v211_spatiotemporal_20panel'; fig.savefig(stem.with_suffix('.pdf'),facecolor='white'); fig.savefig(stem.with_suffix('.svg'),facecolor='white'); fig.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    q.to_csv(OUT/'source_scheme_averaged_predictions.csv',index=False); pd.DataFrame(audits).to_csv(OUT/'map_geometry_audit.csv',index=False); (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_spatiotemporal_20panel','panels':20,'years':YEARS,'scheme_average':'four partition schemes per tract-year','water_exclusion':'water_share > 0.5 or ALAND <= 0','city_background_drawn':False,'proxy_analysis':True,'formal':False,'backend':'python'},indent=2),encoding='utf-8')
    print(pd.DataFrame(audits).to_string(index=False))
if __name__=='__main__':main()
