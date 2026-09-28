"""V211 20-panel local-coefficient map matrix with water-dominant exclusion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_local_coefficients_r3_indexed_seed_filtered'; GEO=ROOT/'Data/05_Administrative_Boundaries/Chicago_Annual_TIGER_2014_2025/processed_chicago_gpkg'; OUT=ROOT/'figures/v211_local_coefficient_20panel'; OUT.mkdir(exist_ok=True); FONT=ROOT/'assets/fonts/nimbus-sans'; RED='#B40426'; BLUE='#3B4CC0'; PALE='#F7F7F7'; INK='#30343B'
YEARS=[2014,2017,2019,2021,2023]; GEOYEAR={2014:2014,2017:2017,2019:2019,2021:2019,2023:2022}; DIMS=[('D1','Green/water'),('D3','Walkability'),('D5','Cleanliness'),('D8','Air pollution')]
def main():
 for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
 mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':6.0,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55,'figure.facecolor':'white','axes.facecolor':'white','text.color':INK})
 d=pd.read_parquet(R/'outer_test_local_coefficients_wide.parquet'); d.tract_geoid_native=d.tract_geoid_native.astype(str).str.zfill(11)
 cols=[]
 for dim,_ in DIMS:
  for p in ['o','s','delta']:
   c=next((c for c in d.columns if c.startswith(f'{p}_d{int(dim[1:])}_')),None); assert c; cols.append((dim,p,c))
 # Map deviation slopes: preserve all four partition schemes by declared mean per tract-year.
 delta_cols={dim:c for dim,p,c in cols if p=='delta'}; q=d.groupby(['tract_geoid_native','health_reference_year'],as_index=False)[list(delta_cols.values())].mean()
 frames={}; audits=[]
 for year in YEARS:
  gy=GEOYEAR[year]; gp=GEO/str(gy)/f'chicago_tiger_{gy}.gpkg'; g=gpd.read_file(gp,layer='tracts_intersect_chicago',columns=['GEOID','ALAND','AWATER','geometry']).to_crs(26916); g.GEOID=g.GEOID.astype(str).str.zfill(11); sub=q[q.health_reference_year.eq(year)]; m=g.merge(sub,left_on='GEOID',right_on='tract_geoid_native',how='inner',validate='one_to_one'); m['water_share']=m.AWATER/(m.ALAND+m.AWATER); before=len(m); m=m.loc[m.water_share.le(.5)&m.ALAND.gt(0)].copy(); frames[year]=m; audits.append({'health_reference_year':year,'geometry_year':gy,'joined_rows':before,'displayed_land_rows':len(m),'excluded_water_dominant_rows':before-len(m),'excluded_zero_land_rows':0,'city_background_drawn':False,'crs':'EPSG:26916'})
 limits={dim:max(float(np.nanquantile(np.abs(pd.concat([frames[y][c] for y in YEARS]).to_numpy()),.98)),1e-8) for dim,c in delta_cols.items()}
 cmap=mpl.colors.LinearSegmentedColormap.from_list('nature_redblue',[BLUE,'#8DB0D5',PALE,'#E6A6A1',RED]); bb=np.array([m.total_bounds for m in frames.values()]); xmin,ymin,xmax,ymax=bb[:,0].min(),bb[:,1].min(),bb[:,2].max(),bb[:,3].max(); dx,dy=xmax-xmin,ymax-ymin; xmin-=.015*dx; xmax+=.015*dx; ymin-=.015*dy; ymax+=.015*dy; fig,ax=plt.subplots(4,5,figsize=(183/25.4,215/25.4)); fig.subplots_adjust(left=.105,right=.975,bottom=.045,top=.945,wspace=.10,hspace=.13)
 for i,(dim,label) in enumerate(DIMS):
  for j,y in enumerate(YEARS):
   a=ax[i,j]; m=frames[y]; norm=TwoSlopeNorm(vmin=-limits[dim],vcenter=0,vmax=limits[dim]); m.plot(column=delta_cols[dim],ax=a,cmap=cmap,norm=norm,edgecolor='white',linewidth=.055,missing_kwds={'color':'#E9EDF2'}); a.set_xlim(xmin,xmax); a.set_ylim(ymin,ymax); a.set_axis_off(); a.set_aspect('equal',adjustable='box');
   if i==0:a.set_title(str(y),fontsize=7,pad=3)
   if j==0:a.text(-.08,.5,label,transform=a.transAxes,rotation=90,ha='right',va='center',fontsize=6.5); a.text(-.02,1.05,chr(97+i),transform=a.transAxes,fontweight='bold',fontsize=8)
   a.text(.96,.03,f'n={len(m):,}',transform=a.transAxes,ha='right',va='bottom',fontsize=5,bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':.6})
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   cax=inset_axes(a,width='3.2%',height='100%',loc='lower left',bbox_to_anchor=(1.025,0,1,1),bbox_transform=a.transAxes,borderpad=0); cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax); cb.ax.tick_params(labelsize=3.6,length=1.2,pad=.4); cb.outline.set_linewidth(.3)
 stem=OUT/'Fig_v211_local_coefficient_20panel'; fig.savefig(stem.with_suffix('.pdf')); fig.savefig(stem.with_suffix('.svg')); fig.savefig(stem.with_suffix('.png'),dpi=600); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
 pd.DataFrame(audits).to_csv(OUT/'map_geometry_audit.csv',index=False); q.to_csv(OUT/'source_scheme_averaged_delta_coefficients.csv',index=False); (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_local_coefficient_20panel','panels':20,'dimensions':[x[0] for x in DIMS],'years':YEARS,'scheme_average':'four partition schemes per tract-year','water_exclusion':'water_share > 0.5 or ALAND <= 0','city_background_drawn':False,'proxy_analysis':True,'formal':False,'backend':'python'},indent=2),encoding='utf-8')
 print(pd.DataFrame(audits).to_string(index=False))
if __name__=='__main__': main()
