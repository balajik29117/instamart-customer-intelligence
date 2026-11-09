"""Part C: privacy-safe, guarded, retrying support ticket triage."""
from pathlib import Path
import json,re,random
import pandas as pd
import mock_llm
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'raw'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
MAX_INPUT_TOKENS=400; BASE_DELAY=.05
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
ZERO="""You are a support-ticket classifier. CLASSIFY each user message. Allowed categories: billing, delivery, technical, fraud. Return JSON only in exactly this shape: {\"category\": \"...\", \"reason\": \"...\"}. Do not include markdown."""
FEW="""You are a support-ticket classifier. CLASSIFY each user message. Allowed categories: billing, delivery, technical, fraud. Return JSON only in exactly this shape: {\"category\": \"...\", \"reason\": \"...\"}. Do not include markdown.
EXAMPLES:
Ticket: I was charged twice and need a refund. -> {\"category\":\"billing\",\"reason\":\"duplicate charge\"}
Ticket: Someone made an unauthorized transaction; this is fraud. -> {\"category\":\"fraud\",\"reason\":\"unauthorized transaction\"}"""
ALLOWED={'billing','delivery','technical','fraud'}
def redact(text):
    counts={}
    text,n=EMAIL.subn('[EMAIL]',text); counts['emails']=n
    text,n=PHONE.subn('[PHONE]',text); counts['phones']=n
    text,n=ORDER.subn('[ORDER_ID]',text); counts['order_ids']=n
    assert not EMAIL.search(text) and not PHONE.search(text) and not ORDER.search(text)
    return text,counts

def call_with_retry(messages,ticket_id,log):
    for retry in range(4):
        try: return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry>=3: raise
            wait=BASE_DELAY*(2**retry)*random.Random(42+retry).uniform(.9,1.15)
            print(f"RETRY ticket_id={ticket_id} retry_number={retry} wait_seconds={wait:.5f}")
            log.append({'ticket_id':ticket_id,'retry_number':retry,'wait_seconds':round(wait,5)})

def classify_run(tickets,system,budget=None):
    mock_llm.reset(); total=0; rows=[]; retries=[]; redaction={'emails':0,'phones':0,'order_ids':0}
    for r in tickets.to_dict('records'):
        text=str(r['text']); tid=r['ticket_id']
        if not text.strip(): rows.append({**r,'status':'skipped_empty','category':'','tokens_used':0}); continue
        if mock_llm.est_tokens(text)>MAX_INPUT_TOKENS: rows.append({**r,'status':'flagged_long','category':'','tokens_used':0}); continue
        safe,c=redact(text)
        for k,v in c.items(): redaction[k]+=v
        if budget is not None and total>=budget:
            rows.append({**r,'status':'budget_exhausted','category':'','tokens_used':0}); continue
        # The user message is exactly the redacted text, without metadata.
        messages=[{'role':'system','content':system},{'role':'user','content':safe}]
        response=call_with_retry(messages,tid,retries)
        used=response.usage.total_tokens; total+=used
        raw=response.choices[0].message.content or ''
        status='flagged_invalid_output'; category=''
        try:
            obj=json.loads(raw); candidate=obj.get('category')
            if candidate in ALLOWED: category=candidate; status='classified_json'
        except (json.JSONDecodeError,AttributeError,TypeError):
            words=re.findall(r'[a-z]+',raw.lower()); hits=set(words)&ALLOWED
            if len(hits)==1: category=next(iter(hits)); status='classified_keyword'
        rows.append({**r,'status':status,'category':category,'tokens_used':used})
    return pd.DataFrame(rows),{'tokens':total,'retries':retries,'redaction':redaction,'status_counts':pd.DataFrame(rows).status.value_counts().to_dict()}

def main():
    t=pd.read_csv(DATA/'tickets.csv',keep_default_na=False)
    zero,zm=classify_run(t,ZERO)
    few,fm=classify_run(t,FEW)
    for df in (zero,few):
        valid=df[df.status.isin(['classified_json','classified_keyword'])]
        df.attrs['accuracy']=float((valid.category==valid.true_category).mean()) if len(valid) else 0
    # Budget demo: deliberately finite budget so the stop condition is observable.
    demo,dm=classify_run(t,ZERO,budget=5000); demo.to_csv(OUT/'triage_budget_demo.csv',index=False)
    # Final routing uses few-shot run with generous token budget.
    final,finalm=classify_run(t,FEW,budget=100000)
    seg=pd.read_csv(OUT/'customer_segments.csv')[['customer_id','segment_name','is_premium']]
    final=final.merge(seg,on='customer_id',how='left'); final['is_premium']=final.is_premium.fillna(False)
    routes=[]; priorities=[]
    for r in final.to_dict('records'):
        if r['status'] in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: route,priority='human_review','NORMAL'
        elif r['category']=='fraud': route,priority='security_team','HIGH'
        elif bool(r['is_premium']): route,priority='premium_desk','HIGH'
        else: route,priority='support_queue','NORMAL'
        routes.append(route); priorities.append(priority)
    final['route']=routes; final['priority']=priorities
    final[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].fillna('').to_csv(OUT/'triage_results.csv',index=False)
    def summary(df,meta):
        v=df[df.status.isin(['classified_json','classified_keyword'])]
        return {'classified':len(v),'accuracy':round(float((v.category==v.true_category).mean()*100),2) if len(v) else 0,'tokens':meta['tokens'],'tokens_per_classified':round(meta['tokens']/len(v),2) if len(v) else 0,'status_counts':meta['status_counts'],'retry_count':len(meta['retries']),'redaction':meta['redaction']}
    # Confusion matrix over classified tickets, zero-shot.
    from sklearn.metrics import confusion_matrix
    zv=zero[zero.status.isin(['classified_json','classified_keyword'])]
    cm=confusion_matrix(zv.true_category,zv.category,labels=['billing','delivery','technical','fraud']).tolist()
    report={'zero_shot':summary(zero,zm),'few_shot':summary(few,fm),'zero_shot_confusion_labels':['billing','delivery','technical','fraud'],'zero_shot_confusion_matrix':cm,'budget_demo':{'budget':5000,'status_counts':dm['status_counts'],'budget_exhausted':int((demo.status=='budget_exhausted').sum()),'tokens':dm['tokens']},'final_few_shot':summary(final,finalm),'final_rows':len(final),'retry_log_zero_shot':zm['retries'],'retry_log_few_shot':fm['retries'],'remaining_pii_matches_after_redaction':0}
    (OUT/'part_c_metrics.json').write_text(json.dumps(report,indent=2))
    print('Part C complete: final rows',len(final),'few-shot accuracy',report['few_shot']['accuracy'])
if __name__=='__main__': main()
