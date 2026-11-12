import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='UrbanHeat AI | Bengaluru', page_icon='🌆', layout='wide')
ROOT = Path(__file__).parent
FEATURES = ['NDVI','NDBI','EVI','MNDWI','NO2_Emissions','Pop_Density','Relative_Humidity','SAVI','Wind_Speed','Latitude','Longitude','Building_Density_Ratio','Dist_Water','Dist_Park','Dist_Highway','Street_Density','Height_Variability','SVF_Proxy','Albedo','Soil_Moisture','Surface_Pressure_hPa','Relative_Wealth_Index','Dist_to_RWI_Node']
DEFAULTS = {'NDVI':.25,'NDBI':.15,'EVI':.20,'MNDWI':.05,'NO2_Emissions':20.,'Pop_Density':5000.,'Relative_Humidity':55.,'SAVI':.20,'Wind_Speed':2.,'Latitude':12.95,'Longitude':77.60,'Building_Density_Ratio':.45,'Dist_Water':2.,'Dist_Park':1.,'Dist_Highway':1.5,'Street_Density':5.,'Height_Variability':8.,'SVF_Proxy':.5,'Albedo':.2,'Soil_Moisture':.2,'Surface_Pressure_hPa':950.,'Relative_Wealth_Index':0.,'Dist_to_RWI_Node':1.}

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 12% 0%,#173451 0%,#0a1423 43%,#07101c 100%);color:#eaf2ff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#f2f7ff!important}
p,label,li{color:#d2deed}.block-container{padding-top:1.6rem;max-width:1500px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2c,#091321);border-right:1px solid #223852}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(24,48,73,.94),rgba(13,29,47,.94));border:1px solid #2a4562;padding:16px;border-radius:16px}
[data-testid="stMetricLabel"]{color:#a9bfd6!important}[data-testid="stMetricValue"]{color:#f5fbff!important;font-family:'Space Grotesk',sans-serif}
div.stButton>button,div.stDownloadButton>button,button[kind="primary"]{border-radius:11px;border:1px solid #3c7e88;background:linear-gradient(100deg,#147e83,#246b9d);color:white;font-weight:700}
.hero{padding:27px 30px;border-radius:22px;background:linear-gradient(110deg,rgba(20,68,91,.94),rgba(23,42,70,.9) 55%,rgba(49,45,82,.86));border:1px solid rgba(100,207,215,.32);margin-bottom:22px}
.kicker{color:#6ee7d0;text-transform:uppercase;font-size:.75rem;letter-spacing:.19em;font-weight:700}.hero h1{font-size:2.35rem;margin:.4rem 0}.hero p{color:#c2d6e8;font-size:1rem;margin:0}
.panel{background:rgba(15,31,50,.82);border:1px solid #263f5b;border-radius:17px;padding:18px}hr{border-color:#263d59!important}
</style>''', unsafe_allow_html=True)

def hero(kicker,title,subtitle):
    st.markdown(f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

# Accept both the exact current GitHub filename and the clean standard filename.
candidates=[ROOT/'UrbanHeat_AI_Final_Model (15).pkl',ROOT/'UrbanHeat_AI_Final_Model.pkl']+sorted(ROOT.glob('UrbanHeat_AI_Final_Model*.pkl'),key=lambda p:p.stat().st_mtime,reverse=True)
model_path=next((p for p in candidates if p.exists()),None)
if model_path is None:
    st.error('Trained model not found. Upload the original .pkl file beside app.py. Accepted names include UrbanHeat_AI_Final_Model (15).pkl and UrbanHeat_AI_Final_Model.pkl.')
    st.stop()

@st.cache_resource
def load_model(path,mtime):
    saved=joblib.load(path)
    if isinstance(saved,dict):
        model=saved.get('model',saved.get('geo_model',saved.get('estimator')))
        features=saved.get('features',FEATURES)
    else: model,features=saved,FEATURES
    if model is None: raise ValueError('Pickle does not contain a model object.')
    return model,list(features)
@st.cache_data
def read_csv(path,mtime): return pd.read_csv(path)
try: model,model_features=load_model(str(model_path),model_path.stat().st_mtime)
except Exception as e:
    st.error(f'Model loading failed: {e}')
    st.info('Use the original joblib .pkl exported from Colab, not a ZIP/archive.')
    st.stop()

def optional_csv(name):
    p=ROOT/name
    if not p.exists(): return pd.DataFrame()
    try: return read_csv(str(p),p.stat().st_mtime)
    except Exception: return pd.DataFrame()
map_data=optional_csv('UrbanHeat_Map_Data.csv')
if not {'Latitude','Longitude','LST'}.issubset(map_data.columns): map_data=pd.DataFrame()
importance=optional_csv('UrbanHeat_Feature_Importance.csv')
if not {'Feature','Importance'}.issubset(importance.columns): importance=pd.DataFrame()

def predict(vals):
    row={f:float(vals.get(f,DEFAULTS.get(f,0))) for f in model_features}
    frame=pd.DataFrame([row],columns=model_features)
    return float(model.predict(frame)[0]),frame

st.sidebar.markdown('## 🌆 UrbanHeat AI')
st.sidebar.caption('BENGALURU • URBAN CLIMATE INTELLIGENCE')
page=st.sidebar.radio('WORKSPACE',['Command Centre','Heat Explorer','AI Prediction','Cooling Lab','Model Insights','Project Brief'])
st.sidebar.markdown('---')
st.sidebar.caption('Supervised learning • Random Forest regression')
st.sidebar.caption(f'Loaded model: {model_path.name}')

if page=='Command Centre':
    hero('BENGALURU URBAN CLIMATE INTELLIGENCE','See the heat. Understand the city.','An AI workspace to explore sampled land-surface-temperature patterns and estimate LST from environmental and built-up features.')
    a,b,c,d=st.columns(4);a.metric('Geographic validation R²','0.517');b.metric('Mean absolute error','1.068 °C');c.metric('RMSE','1.465 °C');d.metric('Algorithm','Random Forest','Regression')
    left,right=st.columns([1.45,1])
    with left:
        st.subheader('🌡️ Bengaluru heat snapshot')
        if not map_data.empty:
            x,y,z=st.columns(3);x.metric('Sample points',f'{len(map_data):,}');y.metric('Mean observed LST',f'{map_data.LST.mean():.2f} °C');z.metric('Maximum observed LST',f'{map_data.LST.max():.2f} °C')
            fig=px.histogram(map_data,x='LST',nbins=24,title='Observed LST distribution',color_discrete_sequence=['#4fd7c5']);fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(12,27,44,.4)',height=320);st.plotly_chart(fig,use_container_width=True)
        else: st.info('Add UrbanHeat_Map_Data.csv with Latitude, Longitude and LST columns to unlock map analytics. Prediction works without it.')
    with right:
        st.subheader('🧭 Intelligence modules')
        st.markdown('<div class="panel"><b>01 · Heat Explorer</b><br>Interactive spatial heat view and sample filters.<hr><b>02 · AI Prediction</b><br>Estimate LST from model input features.<hr><b>03 · Cooling Lab</b><br>Compare hypothetical input scenarios.<hr><b>04 · Model Insights</b><br>Validation metrics and feature importance.</div>',unsafe_allow_html=True)
    st.subheader('AI pipeline');st.markdown('**Environmental + built-up features** → **Random Forest Regression** → **Estimated LST** → **Spatial exploration & planning insights**')
    st.caption('Metrics are from the project geographic-validation experiment. This prototype is not connected to live satellite feeds or live weather.')

elif page=='Heat Explorer':
    hero('SPATIAL ANALYTICS','Bengaluru heat explorer','Explore sampled coordinates included in your project dataset.')
    if map_data.empty: st.warning('Upload UrbanHeat_Map_Data.csv with Latitude, Longitude and LST columns to enable the map.')
    else:
        lo,hi=float(map_data.LST.min()),float(map_data.LST.max());rng=st.slider('Observed LST range (°C)',float(np.floor(lo)),float(np.ceil(hi)),(float(np.floor(lo)),float(np.ceil(hi))))
        df=map_data[map_data.LST.between(*rng)].copy();a,b,c=st.columns(3);a.metric('Visible samples',f'{len(df):,}');b.metric('Average LST',f'{df.LST.mean():.2f} °C' if len(df) else '—');c.metric('Highest LST',f'{df.LST.max():.2f} °C' if len(df) else '—')
        if len(df):
            df['color']=df.LST.apply(lambda t:[55,145,255,190] if t<map_data.LST.quantile(.33) else ([255,190,55,210] if t<map_data.LST.quantile(.66) else [245,65,75,225]))
            layers=[pdk.Layer('HeatmapLayer',data=df,get_position='[Longitude, Latitude]',get_weight='LST',radiusPixels=55,intensity=1.2),pdk.Layer('ScatterplotLayer',data=df,get_position='[Longitude, Latitude]',get_fill_color='color',get_radius=90,pickable=True)]
            view=pdk.ViewState(latitude=float(df.Latitude.mean()),longitude=float(df.Longitude.mean()),zoom=10.5,pitch=35)
            st.pydeck_chart(pdk.Deck(layers=layers,initial_view_state=view,tooltip={'html':'<b>LST:</b> {LST} °C<br/><b>Lat:</b> {Latitude}<br/><b>Lon:</b> {Longitude}'}),use_container_width=True)
            fig=px.scatter(df,x='Longitude',y='Latitude',color='LST',color_continuous_scale=['#398bff','#ffc04d','#f34b55'],hover_data=['LST'],title='Sampled coordinates by observed LST');fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(12,27,44,.4)',height=330);st.plotly_chart(fig,use_container_width=True)
            st.dataframe(df.drop(columns=['color']),use_container_width=True,hide_index=True);st.download_button('Download filtered samples',df.drop(columns=['color']).to_csv(index=False).encode(),'UrbanHeat_filtered_samples.csv','text/csv')
        else: st.info('No sampled points fall in this temperature range.')

elif page=='AI Prediction':
    hero('MODEL INFERENCE','Predict land-surface temperature','Enter feature values in the same units and scales used during training.')
    st.caption('Preset values are illustrative starting points, not measured values for a real address.')
    with st.form('prediction_form'):
        vals={};cols=st.columns(3)
        for i,f in enumerate(model_features):
            with cols[i%3]: vals[f]=st.number_input(f,value=float(DEFAULTS.get(f,0)),format='%.5f')
        go=st.form_submit_button('⚡ Run AI prediction',use_container_width=True)
    if go:
        try:
            pred,frame=predict(vals);st.metric('Estimated land-surface temperature',f'{pred:.2f} °C');st.caption('Model estimate, not a live measurement, official heat alert or future forecast.')
            report=frame.copy();report['Estimated_LST_C']=pred;st.download_button('Download prediction report',report.to_csv(index=False).encode(),'UrbanHeat_AI_prediction.csv','text/csv')
        except Exception as e: st.error(f'Prediction failed: {e}')

elif page=='Cooling Lab':
    hero('WHAT-IF SCENARIO STUDIO','Explore urban cooling scenarios','Adjust selected model inputs and compare estimated LST for two hypothetical feature combinations.')
    st.info('This is a model scenario comparison, not a causal simulation. Other inputs stay at illustrative preset values; actual outcomes depend on local conditions and training data.')
    x,y=st.columns(2)
    with x:
        st.markdown('#### 🌳 Greener neighbourhood');ndvi=st.slider('Vegetation index (NDVI)',-.2,.9,.45,.01);evi=st.slider('Enhanced vegetation (EVI)',-.2,.9,.35,.01);savi=st.slider('Soil-adjusted vegetation (SAVI)',-.2,.9,.35,.01);park=st.slider('Distance to park feature',0.,10.,.5,.1)
    with y:
        st.markdown('#### 🏙️ Built environment');ndbi=st.slider('Built-up index (NDBI)',-.5,.9,.10,.01);building=st.slider('Building density ratio',0.,1.,.30,.01);humidity=st.slider('Relative humidity (%)',10.,100.,55.,1.);wind=st.slider('Wind speed feature',0.,15.,2.5,.1)
    base={f:float(DEFAULTS.get(f,0)) for f in model_features};scenario=dict(base)
    for k,v in {'NDVI':ndvi,'EVI':evi,'SAVI':savi,'Dist_Park':park,'NDBI':ndbi,'Building_Density_Ratio':building,'Relative_Humidity':humidity,'Wind_Speed':wind}.items():
        if k in scenario: scenario[k]=v
    try:
        p0,_=predict(base);p1,_=predict(scenario);delta=p1-p0;a,b,c=st.columns(3);a.metric('Baseline estimate',f'{p0:.2f} °C');b.metric('Scenario estimate',f'{p1:.2f} °C',f'{delta:+.2f} °C vs baseline',delta_color='inverse');c.metric('Absolute difference',f'{abs(delta):.2f} °C')
        comp=pd.DataFrame({'Scenario':['Illustrative baseline','Adjusted scenario'],'Estimated LST (°C)':[p0,p1]});fig=px.bar(comp,x='Scenario',y='Estimated LST (°C)',color='Scenario',color_discrete_sequence=['#438ee8','#4fd7c5'],text_auto='.2f');fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(12,27,44,.4)',showlegend=False,height=350);st.plotly_chart(fig,use_container_width=True);st.download_button('Download scenario comparison',comp.to_csv(index=False).encode(),'UrbanHeat_scenario_comparison.csv','text/csv')
    except Exception as e: st.error(f'Scenario prediction failed: {e}')

elif page=='Model Insights':
    hero('MODEL TRANSPARENCY','Understand the model','Review geographic validation metrics and inspect features used by the trained Random Forest.')
    a,b,c=st.columns(3);a.metric('R² score','0.517');b.metric('MAE','1.068 °C');c.metric('RMSE','1.465 °C')
    st.markdown('**R²** measures explained target variation. **MAE** is average absolute error. **RMSE** penalizes larger errors more strongly.')
    st.warning('The validation split groups nearby coordinates rounded to two decimal places. It is a spatial-group holdout, not ward-based validation.')
    if not importance.empty:
        n=st.slider('Features to display',5,min(23,len(importance)),min(12,len(importance)));top=importance.sort_values('Importance',ascending=False).head(n).sort_values('Importance');fig=px.bar(top,x='Importance',y='Feature',orientation='h',color='Importance',color_continuous_scale=['#246b9d','#4fd7c5']);fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(12,27,44,.4)',height=460,coloraxis_showscale=False);st.plotly_chart(fig,use_container_width=True);st.dataframe(importance.sort_values('Importance',ascending=False),use_container_width=True,hide_index=True);st.download_button('Download feature importance',importance.to_csv(index=False).encode(),'UrbanHeat_feature_importance.csv','text/csv');st.caption('Feature importance indicates model reliance, not proof of causation.')
    else: st.info('Upload UrbanHeat_Feature_Importance.csv with Feature and Importance columns to show this chart.')

else:
    hero('PROJECT OVERVIEW','UrbanHeat AI','AI-powered urban heat exploration and land-surface-temperature estimation for sampled locations in Bengaluru, Karnataka.')
    st.subheader('Objective');st.write('Use supervised machine learning to estimate land-surface temperature from environmental, built-up and geographic features, and explore spatial patterns in available samples.')
    a,b,c=st.columns(3);a.markdown('<div class="panel"><b>Machine learning</b><br>Python · Scikit-learn · Random Forest</div>',unsafe_allow_html=True);b.markdown('<div class="panel"><b>Data analysis</b><br>Pandas · NumPy · Feature importance</div>',unsafe_allow_html=True);c.markdown('<div class="panel"><b>Application</b><br>Streamlit · PyDeck · Plotly</div>',unsafe_allow_html=True)
    st.subheader('Workflow');st.markdown('1. Prepare environmental, built-up and location features.\n2. Train a supervised Random Forest regression model.\n3. Evaluate with a geographic holdout split.\n4. Explore sampled locations and test model scenarios.')
    st.subheader('Important limitations');st.markdown('- Dataset covers sampled Bengaluru coordinates, not every location in the city.\n- No live satellite or weather connection is included.\n- Predictions are estimates, not official alerts or future forecasts.\n- Cooling Lab is a what-if model comparison, not proof of real-world causal impact.\n- Results depend on training-data quality and coverage.')
    st.caption('UrbanHeat AI • Academic supervised-learning prototype')
