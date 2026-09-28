# Academic Figure Skill Asset Confirmation (verified against the legacy moderation figure)
# (a) scheme-by-SES matrix -> diverging heatmap grammar -> visual adapt
# (b) cross-scheme summary -> forest grammar -> visual adapt
# (c-o) scheme-specific gaps -> compact lollipop grammar -> visual adapt
"""Fifteen-panel V211 multidimensional inequality flagship figure."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'workstreams/RQ2_model/runs/v211_compact_proxy_multidimensional_inequality_r3_indexed_seed_filtered/ses_prediction_disparity_by_scheme.csv'; OUT=ROOT/'figures/v211_inequality_flagship_15panel'; OUT.mkdir(exist_ok=True); FONT=ROOT/'assets/fonts/nimbus-sans'
BLUE,RED,WHITE,INK,GREY='#3B4CC0','#B40426','#F7F7F7','#292D35','#7C8490'
SES=['ses__age_65plus_pct','ses__age_under18_pct','ses__bachelors_or_higher_pct_25plus','ses__black_alone_pct','ses__commute_transit_pct','ses__commute_walk_pct','ses__gross_rent_30plus_pct','ses__hispanic_latino_pct','ses__log_income_nominal','ses__male_pct','ses__population_log1p','ses__poverty_pct','ses__unemployment_pct']
LAB=['Age ≥65','Age <18','Higher education','Black population','Transit commute','Walk commute','Rent burden','Hispanic/Latino','Log income','Male population','Population size','Poverty','Unemployment']
def scheme_code(x):
    mp={'axis_recursive':'AR','polar_north_clockwise_equal_count':'PC','rotated45_recursive':'R45','y_equal_count_stripes':'EQ'}; a,v=x.rsplit('__v',1); return f'{mp[a]}·{v[-2:]}'

def main():
    for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
    mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':5.9,'axes.titlesize':6.3,'axes.labelsize':6.2,'xtick.labelsize':5.0,'ytick.labelsize':5.0,'axes.linewidth':.55,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white','text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    d=pd.read_csv(DATA); schemes=list(d.scheme.drop_duplicates());
    if len(d)!=104 or len(schemes)!=8 or set(d.ses_variable)!=set(SES): raise RuntimeError(f'Unexpected inequality structure: rows={len(d)}, schemes={len(schemes)}, SES={d.ses_variable.nunique()}')
    mat=d.pivot(index='ses_variable',columns='scheme',values='standardized_prediction_gap').reindex(index=SES,columns=schemes); lim=max(float(np.nanquantile(np.abs(mat.to_numpy()),.98)),1e-9); norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim); cmap=mpl.colors.LinearSegmentedColormap.from_list('nature_rb',[BLUE,'#8DB0D5',WHITE,'#E6A6A1',RED]); codes=[scheme_code(x) for x in schemes]
    fig=plt.figure(figsize=(183/25.4,198/25.4)); outer=fig.add_gridspec(4,1,height_ratios=[1.38,1,1,1],left=.09,right=.985,bottom=.055,top=.95,hspace=.29); top=outer[0].subgridspec(1,5,wspace=.55); axa=fig.add_subplot(top[0,:3]); axb=fig.add_subplot(top[0,3:])
    im=axa.imshow(mat.to_numpy(),aspect='auto',cmap=cmap,norm=norm,interpolation='nearest'); axa.set_xlim(-.5,8.55); axa.set_xticks(np.arange(8),codes,rotation=45,ha='right'); axa.set_yticks(np.arange(13),LAB); axa.tick_params(length=0); axa.text(-.13,1.03,'a',transform=axa.transAxes,fontweight='bold',fontsize=8); axa.set_title('Standardized Q3 − Q1 prediction gaps',loc='left',pad=4); cax=axa.inset_axes([.94,.08,.025,.82]); cb=fig.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=4.2,length=1,pad=.5); cb.set_ticks([-lim,0,lim]); cb.set_ticklabels([f'−{lim:.1f}','0',f'{lim:.1f}'])
    summary=[]
    for yi,(var,label) in enumerate(zip(SES,LAB)):
        v=mat.loc[var].to_numpy(float); lo,med,hi=np.min(v),np.median(v),np.max(v); color=cmap(norm(med)); axb.hlines(yi,lo,hi,color=color,lw=1.5,alpha=.55); axb.scatter(v,np.full(8,yi),s=6,color=color,alpha=.28,lw=0); axb.plot(med,yi,'o',ms=3.8,mfc=color,mec='white',mew=.35); summary.append({'ses_variable':var,'median_across_schemes':med,'minimum':lo,'maximum':hi})
    axb.axvspan(-lim,0,color=BLUE,alpha=.035,zorder=-5); axb.axvspan(0,lim,color=RED,alpha=.035,zorder=-5); axb.axvline(0,color=GREY,lw=.65); axb.set_yticks(np.arange(13),LAB); axb.invert_yaxis(); axb.grid(axis='x',color='#E5E9EF',lw=.4); axb.set_axisbelow(True); axb.set_xlabel('Standardized Q3 − Q1 gap'); axb.text(-.17,1.03,'b',transform=axb.transAxes,fontweight='bold',fontsize=8); axb.set_title('Cross-scheme robustness',loc='left',pad=4)
    rows=[SES[:4],SES[4:9],SES[9:]]; letter=2
    for ri,vars_ in enumerate(rows,1):
        subgs=outer[ri].subgridspec(1,len(vars_),wspace=.32)
        for j,var in enumerate(vars_):
            ax=fig.add_subplot(subgs[0,j]); sub=d[d.ses_variable.eq(var)].set_index('scheme').reindex(schemes); vals=sub.standardized_prediction_gap.to_numpy(float); y=np.arange(8); colors=[cmap(norm(x)) for x in vals]; ax.axvspan(-lim*1.08,0,color=BLUE,alpha=.028,zorder=-5); ax.axvspan(0,lim*1.08,color=RED,alpha=.028,zorder=-5); ax.axvline(0,color=GREY,lw=.55); ax.hlines(y,0,vals,color='#C9D1DC',lw=.8); ax.scatter(vals,y,s=16,c=colors,edgecolor=INK,linewidth=.18,zorder=3); ax.set_xlim(-lim*1.08,lim*1.08); ax.set_ylim(7.45,-.45); ax.set_yticks(y,codes if j==0 else ['']*8); ax.tick_params(axis='y',length=0); ax.grid(axis='x',color='#E5E9EF',lw=.35); ax.set_axisbelow(True); idx=SES.index(var); ax.set_title(LAB[idx],loc='left',pad=2); ax.text(-.13,1.03,chr(97+letter),transform=ax.transAxes,fontweight='bold',fontsize=7.5); letter+=1
            if ri==3: ax.set_xlabel('Standardized gap')
    stem=OUT/'Fig_v211_inequality_flagship_15panel'; fig.savefig(stem.with_suffix('.pdf')); fig.savefig(stem.with_suffix('.svg')); fig.savefig(stem.with_suffix('.png'),dpi=600); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
    d.to_csv(OUT/'source_all_scheme_ses_gaps.csv',index=False); pd.DataFrame(summary).to_csv(OUT/'source_cross_scheme_summary.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_inequality_flagship_15panel','panels':15,'rows':len(d),'ses_dimensions':13,'schemes':8,'effect':'standardized prediction difference, high SES quartile minus low SES quartile','interpretation':'descriptive model-projected disparity; not causal','proxy_analysis':True,'formal':False},indent=2),encoding='utf-8')
if __name__=='__main__': main()
