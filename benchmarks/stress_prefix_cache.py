import argparse, concurrent.futures, json, pathlib, time, urllib.request
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--port',type=int,default=19088);a=p.parse_args()
rows=[];stamp=str(time.time_ns())
cases=[]
for i in range(8):
    code=f'REF-{stamp[-6:]}-{i}-QUARTZ'
    prompt=f'Document {stamp}-{i}. The reference code is {code}.\n'+('The service parses source files, validates syntax, tracks dependencies, caches results, reports diagnostics, and isolates plugins.\n'*100)+'\nReturn only the reference code stated at the start of this document.'
    cases.append((code,prompt))
def request(case):
    code,prompt=case
    body=dict(model='qwen38-flash-next-uncensored',messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=40,chat_template_kwargs={'enable_thinking':False})
    req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:j=json.load(r)
    answer=j['choices'][0]['message'].get('content') or ''
    return dict(expected=code,answer=answer,passed=answer.strip()==code,usage=j['usage'])
for round_no in range(4):
    # Reorder cached requests between rounds to exercise state-slot reuse.
    ordered=cases[round_no:]+cases[:round_no]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:results=list(pool.map(request,ordered))
    for row in results:row['round']=round_no
    rows.extend(results);pathlib.Path(a.output).write_text(json.dumps(rows,indent=2))
    print(json.dumps({'round':round_no,'passed':sum(r['passed'] for r in results),'total':len(results),'answers':[r['answer'] for r in results]}),flush=True)
if not all(r['passed'] for r in rows):raise SystemExit('Cached-prefix regression failed')
