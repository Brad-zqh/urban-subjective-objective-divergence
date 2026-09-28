"""Full ten-dimension coefficient atlas (20 panels) for the current V211 proxy run."""
from pathlib import Path
import json, numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered'; GEO=ROOT/'Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg/2022/chicago_tiger_2022.gpkg'; OUT=ROOT/'figures/v211_full_coefficient_atlas_20panel'; OUT.mkdir(exist_ok=True); FONT=ROOT/'assets/fonts/nimbus-sans'
BLUE='#3B4CC0'; RED='#B40426'; PALE='#F7F7F7'; INK='#30343B'; DIMS=[f'D{i}' for i in range(1,11)]
def main():
 for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
 mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial'],'font.size':5.8,'pdf.fonttype':42,'svg.fonttype':'none','axes.facecolor':'white','figure.facecolor':'white','text.color':INK})
 d=pd.read_parquet(RUN/'outer_test_local_coefficients_wide.parquet'); d.tract_geoid_native=d.tract_geoid_native.astype(str).str.zfill(11); d=d[d.health_reference_year.eq(2023)]
 cols=[]
 for i in range(1,11):
  oc=next(c for c in d.columns if c.startswith(f'o_d{i}_')); dc=next(c for c in d.columns if c.startswith(f'delta_d{i}_')); cols += [(f'D{i} objective',oc),(f'D{i} social − objective',dc)]
 q=d.groupby('tract_geoid_native',as_index=False)[[c for _,c in cols]].mean()
 g=gpd.read_file(GEO,layer='tracts_intersect_chicago',columns=['GEOID','ALAND','AWATER','geometry']).to_crs(26916); g.GEOID=g.GEOID.astype(str).str.zfill(11); g['water_share']=g.AWATER/(g.ALAND+g.AWATER); g=g[g.water_share.le(.5)&g.ALAND.gt(0)].copy(); m=g.merge(q,left_on='GEOID',right_on='tract_geoid_native',how='inner'); xmin,ymin,xmax,ymax=m.total_bounds; dx,dy=xmax-xmin,ymax-ymin; xmin-=.015*dx; xmax+=.015*dx; ymin-=.015*dy; ymax+=.015*dy
 cmap=mpl.colors.LinearSegmentedColormap.from_list('nature_redblue',[BLUE,'#8DB0D5',PALE,'#E6A6A1',RED]); fig,ax=plt.subplots(5,4,figsize=(183/25.4,247/25.4)); fig.subplots_adjust(left=.055,right=.985,bottom=.035,top=.955,wspace=.22,hspace=.24)
 for k,(label,col) in enumerate(cols):
  i,j=divmod(k,4); a=ax[i,j]; v=m[col].to_numpy(float); lim=max(float(np.nanquantile(np.abs(v),.98)),1e-8); norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); m.plot(column=col,ax=a,cmap=cmap,norm=norm,edgecolor='white',linewidth=.04,missing_kwds={'color':'#E9EDF2'}); a.set_xlim(xmin,xmax); a.set_ylim(ymin,ymax); a.set_aspect('auto'); a.set_axis_off(); a.set_title(label,fontsize=5.8,pad=1.5); a.text(.02,1.02,chr(97+k),transform=a.transAxes,fontweight='bold',fontsize=7.5); a.text(.93,.03,f'n={len(m):,}',transform=a.transAxes,ha='right',va='bottom',fontsize=4.0,bbox={'facecolor':'white','alpha':.7,'edgecolor':'none','pad':.4}); cax=inset_axes(a,width='3.2%',height='100%',loc='lower left',bbox_to_anchor=(1.025,0,1,1),bbox_transform=a.transAxes,borderpad=0); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax); cb.ax.tick_params(labelsize=3.0,length=.9,pad=.25); cb.outline.set_linewidth(.22)
 stem=OUT/'Fig_v211_full_coefficient_atlas_20panel'; fig.savefig(stem.with_suffix('.pdf')); fig.savefig(stem.with_suffix('.svg')); fig.savefig(stem.with_suffix('.png'),dpi=600); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig); q.to_csv(OUT/'source_data_2023_scheme_average.csv',index=False); (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_full_coefficient_atlas_20panel','panels':20,'year':2023,'dimensions':10,'water_exclusion':'water_share > 0.5 or ALAND <= 0','delivery_formats':['jpg','pdf','png','svg','tiff'],'proxy_analysis':True,'formal':False},indent=2),encoding='utf-8')
if __name__=='__main__': main()
