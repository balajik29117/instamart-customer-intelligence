"""Independent Part A checkpoint validation for cleaned exports and SQLite."""
from pathlib import Path
import sqlite3
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    customers=pd.read_csv(ROOT/'data/clean/customers.csv')
    orders=pd.read_csv(ROOT/'data/clean/orders.csv')
    with sqlite3.connect(ROOT/'data/instamart.db') as con:
        tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'customers','products','orders'} <= tables
        db_orders=pd.read_sql_query("SELECT * FROM orders",con)
    assert len(customers)==1200
    assert customers.city.nunique()==6
    assert customers.age.notna().all()
    assert len(orders)==12309 and orders.order_id.nunique()==12309
    assert not orders.status.eq('pending').any()
    assert int(orders.status.eq('completed').sum())==11110
    assert int(orders.status.eq('cancelled').sum())==1199
    assert round(orders.loc[orders.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
    assert orders.loc[orders.status.eq('completed'),'delivery_minutes'].notna().all()
    assert orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().all()
    assert len(db_orders)==len(orders)
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': main()
