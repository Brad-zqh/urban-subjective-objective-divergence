"""20-panel V211 projected-sensitivity and inequality atlas.

Seven scenario panels show complete outer-test prediction-difference ECDFs;
thirteen panels show every SES gradient for every frozen partition scheme.
These are model-projected proxy sensitivities, not causal policy estimates.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'workstreams/RQ2_model/runs'; OUT=ROOT/'figures/v211_scenario_inequality_20panel'; OUT.mkdir(exist_ok=True); FONT=ROOT/'assets/fonts/nimbus-sans'; RED='#B40426'; BLUE='#3B4CC0'; PURPLE='#9B6BAA'; PALE='#C9D6EC'; INK='#30343B'; MID='#8D96A3'
SCENARIOS=['D1_green_water_plus_1sd','D5_clean_maintenance_minus_1sd','D7_heat_mitigation_minus_1sd','D8_air_pollution_minus_1sd','D9_noise_mitigation_minus_1sd','D10_traffic_pressure_minus_1sd','joint_six_dimensions']; SHORT={'D1_green_water_plus_1sd':'Green/water +1 SD','D5_clean_maintenance_minus_1sd':'Maintenance −1 SD','D7_heat_mitigation_minus_1sd':'Heat mitigation −1 SD','D8_air_pollution_minus_1sd':'Air pollution −1 SD','D9_noise_mitigation_minus_1sd':'Noise mitigation −1 SD','D10_traffic_pressure_minus_1sd':'Traffic pressure −1 SD','joint_six_dimensions':'Joint six dimensions'}
SES=['ses__age_65plus_pct','ses__age_under18_pct','ses__bachelors_or_higher_pct_25plus','ses__black_alone_pct','ses__commute_transit_pct','ses__commute_walk_pct','ses__gross_rent_30plus_pct','ses__hispanic_latino_pct','ses__log_income_nominal','ses__male_pct','ses__population_log1p','ses__poverty_pct','ses__unemployment_pct']; SES_SHORT={'ses__age_65plus_pct':'Age ≥65 (%)','ses__age_under18_pct':'Age <18 (%)','ses__bachelors_or_higher_pct_25plus':'Higher education (%)','ses__black_alone_pct':'Black alone (%)','ses__commute_transit_pct':'Transit commute (%)','ses__commute_walk_pct':'Walk commute (%)','ses__gross_rent_30plus_pct':'Rent burden (%)','ses__hispanic_latino_pct':'Hispanic/Latino (%)','ses__log_income_nominal':'Log income','ses__male_pct':'Male (%)','ses__population_log1p':'Population log1p','ses__poverty_pct':'Poverty (%)','ses__unemployment_pct':'Unemployment (%)'}
def main():
 for p in FONT.glob('NimbusSans-*.otf'): font_manager.fontManager.addfont(str(p))
 mpl.rcParams.update({'font.family':'Nimbus Sans','font.sans-serif':['Nimbus Sans','Helvetica','Arial','DejaVu Sans'],'font.size':6.0,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'figure.facecolor':'white'})
 s=pd.read_parquet(R/'v211_compact_proxy_scenarios_r3_indexed_seed_filtered/outer_test_scenario_predictions.parquet'); i=pd.read_csv(R/'v211_compact_proxy_multidimensional_inequality_r3_indexed_seed_filtered/ses_prediction_disparity_by_scheme.csv'); schemes=list(i.scheme.drop_duplicates());
 fig,ax=plt.subplots(4,5,figsize=(183/25.4,247/25.4)); fig.subplots_adjust(left=.10,right=.985,bottom=.07,top=.93,wspace=.27,hspace=.55)
 # Scenario row: each panel keeps all rows, with an ECDF and a compact fold summary inset.
 allv=s.prediction_difference.to_numpy(float); lo,hi=np.quantile(allv,[.005,.995])
 for k,sc in enumerate(SCENARIOS):
  a=ax.flat[k]; v=np.sort(s.loc[s.scenario.eq(sc),'prediction_difference'].to_numpy(float)); a.plot(v,np.arange(1,len(v)+1)/len(v),color=RED if k<6 else PURPLE,lw=.9); med=np.median(v); a.axvline(med,color=RED,lw=.7); a.set_xlim(lo,hi); a.set_ylim(0,1); a.set_title(SHORT[sc],fontsize=6.4,loc='left',pad=3); a.text(.96,.10,f'n={len(v):,}\nmedian={med:.3g} pp',transform=a.transAxes,ha='right',va='bottom',fontsize=5); a.text(-.16,1.05,chr(97+k),transform=a.transAxes,fontweight='bold',fontsize=7.5); a.set_ylabel('ECDF' if k%5==0 else ''); a.set_xlabel('Prediction difference (pp)' if k//5==1 else ''); a.grid(False)
 # Inequality row: one panel per SES variable, all eight schemes shown with redundant marker/colour cues.
 colors=[BLUE,BLUE,BLUE,BLUE,RED,RED,RED,RED]; y=np.arange(len(schemes));
 short_scheme={x:('AR·10' if x=='axis_recursive__v2010' else 'AR·20' if x=='axis_recursive__v2020' else 'PC·10' if x=='polar_north_clockwise_equal_count__v2010' else 'PC·20' if x=='polar_north_clockwise_equal_count__v2020' else 'R45·10' if x=='rotated45_recursive__v2010' else 'R45·20' if x=='rotated45_recursive__v2020' else 'EQ·10' if x=='y_equal_count_stripes__v2010' else 'EQ·20') for x in schemes}
 for j,var in enumerate(SES):
  k=7+j; a=ax.flat[k]; sub=i[i.ses_variable.eq(var)].set_index('scheme').reindex(schemes); vals=sub.standardized_prediction_gap.to_numpy(float); a.axvline(0,color=MID,lw=.55); a.hlines(y,0,vals,color='#DCE1E8',lw=.65,zorder=1); a.scatter(vals,y,s=14,c=colors,edgecolor='white',lw=.3,zorder=3); a.set_yticks(y, [short_scheme[x] if j==0 else '' for x in schemes]); a.invert_yaxis(); a.set_title(SES_SHORT[var],fontsize=6.2,loc='left',pad=3); a.set_xlabel('Q3 − Q1 (standardized)' if j//5==2 else ''); a.text(-.16,1.05,chr(97+k),transform=a.transAxes,fontweight='bold',fontsize=7.5); a.grid(axis='x',color='#E7EAF0',lw=.35); a.text(.97,.08,f'n={int(sub.n_low_q1.iloc[0]):,} + {int(sub.n_high_q3.iloc[0]):,}',transform=a.transAxes,ha='right',va='bottom',fontsize=4.7,color='#596273')
 stem=OUT/'Fig_v211_scenario_inequality_20panel'; fig.savefig(stem.with_suffix('.pdf')); fig.savefig(stem.with_suffix('.svg')); fig.savefig(stem.with_suffix('.png'),dpi=600); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.tiff'),dpi=(600,600),compression='tiff_lzw'); Image.open(stem.with_suffix('.png')).convert('RGB').save(stem.with_suffix('.jpg'),quality=96,subsampling=0,dpi=(600,600),optimize=True); plt.close(fig)
 s.to_csv(OUT/'source_scenario_predictions.csv',index=False); i.to_csv(OUT/'source_ses_prediction_disparity.csv',index=False); (OUT/'manifest.json').write_text(json.dumps({'figure':'Fig_v211_scenario_inequality_20panel','panels':20,'scenario_rows':len(s),'inequality_rows':len(i),'schemes':schemes,'proxy_analysis':True,'formal':False,'backend':'python'},indent=2),encoding='utf-8')
if __name__=='__main__': main()
