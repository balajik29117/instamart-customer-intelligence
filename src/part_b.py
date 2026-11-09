"""Part B: RFM segmentation and leakage-controlled lapse prediction."""
from pathlib import Path
import json, copy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
SNAP=pd.Timestamp('2025-10-01'); CUTOFF=pd.Timestamp('2025-09-01')

def main():
    o=pd.read_csv(DATA/'clean/orders.csv',parse_dates=['order_ts']); c=pd.read_csv(DATA/'clean/customers.csv',parse_dates=['signup_date']); p=pd.read_csv(DATA/'clean/products.csv')
    done=o[o.status.eq('completed')].copy()
    rfm=done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(SNAP-rfm.last_order).dt.days; rfm=rfm.drop(columns='last_order')
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
    ks=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
        ks.append({'k':k,'inertia':round(float(km.inertia_),1),'silhouette':round(float(silhouette_score(X,km.labels_)),3)})
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); rfm['segment_id']=km.labels_
    profiles=rfm.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).round(2)
    premium_id=int(profiles.monetary.idxmax())
    # Assign business names based on profile, keeping cluster id as an arbitrary label.
    names={}
    for sid,row in profiles.iterrows():
        if sid==premium_id: names[sid]='Premium'
        elif row.frequency>=profiles.frequency.median(): names[sid]='Frequent Loyalists'
        elif row.recency>60: names[sid]='At-Risk Customers'
        else: names[sid]='Occasional Shoppers'
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
    rfm.to_csv(OUT/'customer_segments.csv',index=False)
    pca=PCA(n_components=2).fit(X)
    # Cohort and label: only pre-cutoff information defines features.
    pre=o[(o.status.isin(['completed','cancelled']))&(o.order_ts<CUTOFF)].copy()
    pre_done=pre[pre.status.eq('completed')]
    last=pre_done.groupby('customer_id').order_ts.max()
    cohort=last.index
    future=done[(done.order_ts>=CUTOFF)&(done.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').size()
    y=pd.Series((~pd.Index(cohort).isin(future.index)).astype(int),index=cohort,name='lapsed')
    feat=pre_done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'),fav_category=('category',lambda s:s.mode().iloc[0] if not s.mode().empty else 'Unknown')).reset_index() if 'category' in pre_done.columns else None
    # Add product category before aggregation.
    pre_done=pre_done.merge(p[['product_id','category']],on='product_id',how='left')
    feat=pre_done.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'),fav_category=('category',lambda s:s.mode().iloc[0] if not s.mode().empty else 'Unknown')).reset_index()
    feat['recency']=(CUTOFF-feat.last_order).dt.days; feat=feat.drop(columns='last_order')
    cancel=pre.groupby('customer_id').status.apply(lambda s:(s=='cancelled').sum()/len(s)).rename('cancellation_rate')
    feat=feat.merge(cancel,on='customer_id',how='left').merge(c[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left')
    feat['tenure_days']=(CUTOFF-feat.signup_date).dt.days; feat=feat.drop(columns='signup_date')
    feat=feat.set_index('customer_id').reindex(cohort)
    feat['avg_delivery']=feat.avg_delivery.fillna(feat.avg_delivery.median())
    feat['fav_category']=feat.fav_category.fillna('Unknown')
    feat['cancellation_rate']=feat.cancellation_rate.fillna(0)
    feat['city']=feat.city.fillna('Unknown'); feat['gender']=feat.gender.fillna('unknown')
    assert not feat.isna().any().any(), 'Feature matrix contains missing values'
    Xdf=feat.copy(); y=y.loc[Xdf.index]
    Xtr,Xte,ytr,yte=train_test_split(Xdf,y,test_size=.2,stratify=y,random_state=42)
    num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']; cat=['city','gender','fav_category']
    prep=ColumnTransformer([('num',Pipeline([('scale',StandardScaler())]),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
    lr=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,random_state=42))])
    gs_lr=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    rf=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    gs_rf=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    candidates={'logistic_regression':gs_lr.best_estimator_,'random_forest':gs_rf.best_estimator_}
    metrics={}
    for name,model in candidates.items():
        pred=model.predict(Xte); metrics[name]={'best_params':(gs_lr.best_params_ if name.startswith('logistic') else gs_rf.best_params_),'cv_f1':round(float(gs_lr.best_score_ if name.startswith('logistic') else gs_rf.best_score_),4),'confusion_matrix':confusion_matrix(yte,pred).tolist(),'precision':round(float(precision_score(yte,pred,zero_division=0)),4),'recall':round(float(recall_score(yte,pred,zero_division=0)),4),'f1':round(float(f1_score(yte,pred,zero_division=0)),4)}
    chosen_name=max(metrics,key=lambda n:metrics[n]['f1']); chosen=candidates[chosen_name]
    # class-balance experiment: oversample positive training rows only, preserve all originals.
    lr_best=gs_lr.best_estimator_; Xtr_bal=Xtr.copy(); ytr_bal=ytr.copy(); pos=Xtr[ytr==1]
    if len(pos)<int((ytr==0).sum()):
        extra=pos.sample(n=int((ytr==0).sum())-len(pos),replace=True,random_state=42)
        Xtr_bal=pd.concat([Xtr_bal,extra]); ytr_bal=pd.concat([ytr_bal,pd.Series(1,index=extra.index)])
    balanced=copy.deepcopy(lr_best)
    balanced.fit(Xtr_bal,ytr_bal); bp=balanced.predict(Xte)
    metrics['balanced_logistic']={'precision':round(float(precision_score(yte,bp,zero_division=0)),4),'recall':round(float(recall_score(yte,bp,zero_division=0)),4),'f1':round(float(f1_score(yte,bp,zero_division=0)),4)}
    # model persistence and exact reload check
    joblib.dump(chosen,OUT/'lapse_model.joblib'); loaded=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(chosen.predict(Xte),loaded.predict(Xte))
    # Premium list is a post-hoc whole-window label, used only for illustrative outreach ranking.
    seg=rfm[['customer_id','is_premium']]
    test_ids=set(Xte.index); premium_ids=set(seg.loc[seg.is_premium,'customer_id']) & test_ids
    probs=chosen.predict_proba(Xte)[:,list(chosen.classes_).index(1)]
    pred_df=pd.DataFrame({'customer_id':Xte.index,'lapse_probability':probs,'actually_lapsed':yte.values})
    top=pred_df[pred_df.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10)
    top.to_csv(OUT/'retention_list.csv',index=False)
    # feature importance + logistic coefficients, best RF and LR
    rf_model=gs_rf.best_estimator_; feat_names=rf_model.named_steps['prep'].get_feature_names_out(); imp=rf_model.named_steps['model'].feature_importances_
    top_imp=sorted(zip(feat_names,imp),key=lambda z:z[1],reverse=True)[:5]
    lr_model=gs_lr.best_estimator_; lr_names=lr_model.named_steps['prep'].get_feature_names_out(); coef=lr_model.named_steps['model'].coef_[0]
    coef3=sorted(zip(lr_names,coef),key=lambda z:abs(z[1]),reverse=True)[:3]
    full_mean=float(Xdf.frequency.mean()); train_freq_mean=float(gs_lr.best_estimator_.named_steps['prep'].named_transformers_['num'].named_steps['scale'].mean_[1])
    report={'rfm':{'customers':len(rfm),'recency_median':float(rfm.recency.median()),'recency_min':int(rfm.recency.min()),'recency_max':int(rfm.recency.max()),'frequency_mean':float(rfm.frequency.mean()),'frequency_max':int(rfm.frequency.max()),'monetary_mean':float(rfm.monetary.mean()),'monetary_max':float(rfm.monetary.max())},'kmeans':ks,'chosen_k':4,'profiles':profiles.reset_index().to_dict('records'),'premium_segment_id':premium_id,'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'cohort_size':len(y),'lapsed_count':int(y.sum()),'lapsed_pct':round(float(y.mean()*100),2),'train_size':len(Xtr),'test_size':len(Xte),'baseline':{'accuracy':round(float(accuracy_score(yte,np.zeros(len(yte)))),4),'lapsed_f1':0},'models':metrics,'selected_model':chosen_name,'reload_predictions_identical':True,'top_rf_features':[(str(a),round(float(b),5)) for a,b in top_imp],'top_lr_coefficients':[(str(a),round(float(b),5)) for a,b in coef3],'scaler_train_frequency_mean':train_freq_mean,'full_cohort_frequency_mean':full_mean,'retention_top10_true_lapsed':int(top.actually_lapsed.sum()),'retention_rows':len(top)}
    (OUT/'part_b_metrics.json').write_text(json.dumps(report,indent=2,default=str))
    print('Part B complete:',chosen_name,'test F1',metrics[chosen_name]['f1'])
if __name__=='__main__': main()
