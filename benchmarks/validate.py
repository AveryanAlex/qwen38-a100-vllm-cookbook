import argparse, concurrent.futures, json, pathlib, time, urllib.request
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19088);p.add_argument('--output',required=True);a=p.parse_args()
base=f'http://127.0.0.1:{a.port}';model='qwen38-flash-next-uncensored';results=[]
def chat(messages,thinking=False,**kwargs):
    body=dict(model=model,messages=messages,temperature=0,max_tokens=256,chat_template_kwargs={'enable_thinking':thinking});body.update(kwargs)
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r:return json.load(r)
def check(name,j,expected):
    text=j['choices'][0]['message'].get('content') or ''
    row=dict(name=name,passed=expected in text,answer=j['choices'][0]['message'],usage=j['usage'])
    results.append(row);pathlib.Path(a.output).write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)

j=chat([{'role':'user','content':'Compute 17 times 19. Show your reasoning and finish with the integer answer.'}],thinking=True)
check('thinking',j,'323')
notes='\n'.join(f'Entry {i}: source analysis must preserve token positions, reject malformed input, report diagnostics, and cache results.' for i in range(450))
for marker in ('COPPER-7319','VIOLET-4261'):
    prompt='Read these notes.\n'+notes+'\nThe deployment marker is '+marker+'.\n'+notes[:1000]+'\nReturn only the deployment marker.'
    j=chat([{'role':'user','content':prompt}],max_tokens=32);check('long-prefix-'+marker,j,marker)
tools=[{'type':'function','function':{'name':'get_weather','description':'Get current weather for a city','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
messages=[{'role':'user','content':'Use get_weather for Paris, then tell me the temperature in Celsius.'}]
j=chat(messages,tools=tools,tool_choice='auto');message=j['choices'][0]['message'];calls=message.get('tool_calls') or []
if calls:
    messages.append(message)
    messages.append({'role':'tool','tool_call_id':calls[0]['id'],'content':'{"city":"Paris","temperature_celsius":15}'})
    j=chat(messages,tools=tools);check('tool-roundtrip',j,'15')
else:check('tool-roundtrip',j,'__MISSING_TOOL_CALL__')

# Close an in-flight stream, then verify a subsequent request remains healthy.
body=dict(model=model,messages=[{'role':'user','content':'Write a long essay about software testing.'}],temperature=0,max_tokens=512,stream=True,chat_template_kwargs={'enable_thinking':False})
req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=600) as r:
    for line in r:
        if b'content' in line:break
time.sleep(1)
j=chat([{'role':'user','content':'Return only the integer: 31 plus 12.'}],max_tokens=32);check('after-cancellation',j,'43')
cases=[(17,19),(23,7),(31,29),(13,11),(41,3),(19,23),(37,5),(29,17)]
def isolated_case(case):
    x,y=case
    return chat([{'role':'user','content':f'Return only the decimal integer: {x} multiplied by {y}.'}],max_tokens=32)
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    answers=list(pool.map(isolated_case,cases))
for (x,y),j in zip(cases,answers):check(f'concurrent-isolation-{x}x{y}',j,str(x*y))
if not all(r['passed'] for r in results):raise SystemExit('Validation failed')
