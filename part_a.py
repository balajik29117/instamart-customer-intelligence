"""Part A: audit, clean, validate, persist and query the synthetic exports."""
from pathlib import Path
import json, sqlite3
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'; CLEAN=ROOT/'data'/'clean'; OUT=ROOT/'outputs'; DB=ROOT/'data'/'instamart.db'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)

def main():
    customers=pd.read_csv(RAW/'customers.csv',keep_default_na=False)
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    audit={}
    for name,df in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
        audit[name]={'shape':list(df.shape),'dtypes':{c:str(t) for c,t in df.dtypes.items()},'missing':{c:{'count':int((df[c].isna() | df[c].astype(str).eq('')).sum()),'pct':round(float((df[c].isna() | df[c].astype(str).eq('')).mean()*100),2)} for c in df.columns}}
    raw_age=pd.to_numeric(customers.age.replace('',np.nan),errors='coerce')
    raw_amount=pd.to_numeric(orders.order_amount,errors='coerce')
    raw_delivery=pd.to_numeric(orders.delivery_minutes.replace('',np.nan),errors='coerce')
    audit['reference_counts']={
      'raw_orders':len(orders),'exact_duplicate_rows':int(orders.duplicated().sum()),'pending_rows':int((orders.status=='pending').sum()),
      'distinct_order_id':int(orders.order_id.nunique()),'non_numeric_amounts':int(raw_amount.isna().sum()),
      'negative_delivery_minutes':int((raw_delivery<0).sum()),'blank_delivery_minutes':int(orders.delivery_minutes.eq('').sum()),
      'blank_age':int(customers.age.eq('').sum()),'ages_outside_10_100':int(((raw_age<10)|(raw_age>100)).sum()),
      'raw_city_spellings':int(customers.city.nunique()),'raw_gender_spellings_nonblank':int(customers.loc[customers.gender.ne(''),'gender'].nunique()),'blank_gender':int(customers.gender.eq('').sum())}
    customers['city']=customers.city.astype(str).str.strip().str.title()
    customers['gender']=customers.gender.astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
    customers['age']=pd.to_numeric(customers.age.replace('',np.nan),errors='coerce')
    customers.loc[~customers.age.between(10,100),'age']=np.nan
    customers['age']=customers.age.fillna(customers.groupby('city').age.transform('median')).round().astype(int)
    customers['signup_date']=pd.to_datetime(customers.signup_date)
    orders['order_ts']=pd.to_datetime(orders.order_ts)
    orders=orders.drop_duplicates().copy()
    orders=orders.sort_values(['order_ts']).drop_duplicates('order_id',keep='last').copy()
    orders['quantity']=pd.to_numeric(orders.quantity)
    orders['discount_pct']=pd.to_numeric(orders.discount_pct)
    orders['order_amount']=pd.to_numeric(orders.order_amount,errors='coerce')
    orders['delivery_minutes']=pd.to_numeric(orders.delivery_minutes.replace('',np.nan),errors='coerce')
    orders.loc[orders.delivery_minutes<0,'delivery_minutes']=np.nan
    orders=orders.merge(products[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    calc=(orders.price_inr*orders.quantity*(1-orders.discount_pct/100)).round(2)
    orders['order_amount']=orders.order_amount.fillna(calc).round(2)
    completed=orders.status.eq('completed')
    med=orders.loc[completed].groupby('category').delivery_minutes.transform('median')
    # category medians aligned to completed rows
    orders.loc[completed,'delivery_minutes']=orders.loc[completed].groupby('category').delivery_minutes.transform(lambda s:s.fillna(s.median()))
    orders=orders.drop(columns=['price_inr','category'])
    products['price_inr']=pd.to_numeric(products.price_inr)
    # Audit and clean artifacts
    for name,df in [('customers',customers),('products',products),('orders',orders)]:
        df.to_csv(CLEAN/f'{name}.csv',index=False)
    with sqlite3.connect(DB) as con:
        for name,df in [('customers',customers),('products',products),('orders',orders)]:
            d=df.copy()
            if name=='customers': d['signup_date']=d.signup_date.dt.strftime('%Y-%m-%d')
            if name=='orders': d['order_ts']=d.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
            d.to_sql(name,con,if_exists='replace',index=False)
        con.execute('CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)')
    # SQL analyses and pandas cross-check
    with sqlite3.connect(DB) as con:
        q={f'Q{i}':pd.read_sql_query(sql,con) for i,sql in enumerate(SQL_QUERIES,1)}
    comp=orders[orders.status.eq('completed')]
    pivot=pd.pivot_table(comp.merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id'),index='city',columns='category',values='order_amount',aggfunc='sum',fill_value=0,margins=True)
    x=comp.order_amount; q1,q3=x.quantile([.25,.75]); iqr=q3-q1
    correlations=comp[['order_amount','discount_pct','quantity','delivery_minutes']].corr()
    audit['clean_counts']={'customers':len(customers),'cities':customers.city.value_counts().to_dict(),'genders':customers.gender.value_counts().to_dict(),'missing_ages':int(customers.age.isna().sum()),'orders':len(orders),'statuses':orders.status.value_counts().to_dict(),'completed_missing_delivery':int(orders.loc[completed,'delivery_minutes'].isna().sum()),'cancelled_missing_delivery':int(orders.loc[~completed,'delivery_minutes'].isna().sum()),'completed_revenue_inr':round(float(comp.order_amount.sum()),2)}
    audit['sql_counts']={name:len(v) for name,v in q.items()}
    audit['pandas']={'pivot_grand_total':round(float(pivot.loc['All','All']),2),'by_city':comp.merge(customers[['customer_id','city']],on='customer_id').groupby('city').order_amount.agg(['sum','mean','count']).round(2).to_dict('index'),'amount_stats':{'mean':round(float(x.mean()),2),'median':round(float(x.median()),2),'skew':round(float(x.skew()),2),'q1':round(float(q1),2),'q3':round(float(q3),2),'lower_fence':round(float(q1-1.5*iqr),2),'upper_fence':round(float(q3+1.5*iqr),2),'below':int((x<q1-1.5*iqr).sum()),'above':int((x>q3+1.5*iqr).sum())},'correlations':correlations.round(3).to_dict()}
    (OUT/'part_a_audit.json').write_text(json.dumps(audit,indent=2,default=str))
    # eight-plus deterministic checkpoints
    assert len(customers)==1200 and customers.city.nunique()==6 and customers.age.isna().sum()==0
    assert len(orders)==12309 and orders.order_id.nunique()==12309 and not (orders.status=='pending').any()
    assert int((orders.status=='completed').sum())==11110 and int((orders.status=='cancelled').sum())==1199
    assert round(float(comp.order_amount.sum()),2)==2998668.80
    assert orders.loc[orders.status=='completed','delivery_minutes'].isna().sum()==0
    assert orders.loc[orders.status=='cancelled','delivery_minutes'].isna().sum()==1199
    assert len(q['Q2'])==30 and len(q['Q3'])==11 and len(q['Q4'])==25 and len(q['Q5'])==550 and len(q['Q7'])==183
    print('ALL CHECKPOINTS PASSED')

SQL_QUERIES=[
"SELECT customer_id, SUM(order_amount) AS spend FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;",
"SELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;",
"SELECT * FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);",
"SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;",
"WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT * FROM spend WHERE total_spend>(SELECT avg_total FROM avg_spend);",
"WITH spend AS (SELECT c.city,c.customer_id,SUM(o.order_amount) total_spend FROM customers c JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.city,c.customer_id) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) row_num FROM spend;",
"WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily ORDER BY day;"
]
if __name__=='__main__': main()
