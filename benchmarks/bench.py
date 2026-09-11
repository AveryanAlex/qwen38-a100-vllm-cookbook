import argparse, concurrent.futures, itertools, json, pathlib, re, statistics, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=19088)
p.add_argument('--output', required=True)
p.add_argument('--tokens', type=int, default=128)
p.add_argument('--wide', action='store_true')
p.add_argument('--prefill', action='store_true')
a = p.parse_args()
base = f'http://127.0.0.1:{a.port}'
model = 'qwen38-flash-next-uncensored'

def post(path, data):
    req = urllib.request.Request(base + path, data=json.dumps(data).encode(), headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req, timeout=600)

def metrics():
    data = urllib.request.urlopen(base + '/metrics').read().decode()
    names = ['generation_tokens_total','inter_token_latency_seconds_count','inter_token_latency_seconds_sum','num_requests_running','num_requests_waiting','spec_decode_num_drafts_total','spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total']
    return {name: next((float(line.rsplit(' ',1)[1]) for line in data.splitlines() if line.startswith('vllm:'+name+'{')), 0) for name in names}

def request(prompt, tokens):
    data = dict(model=model, messages=[dict(role='user',content=prompt)], temperature=0, max_tokens=tokens, ignore_eos=True,
                chat_template_kwargs={'enable_thinking':False}, stream=True, stream_options={'include_usage':True})
    start = time.perf_counter(); first = last = None; chunks=[]; text=''; usage=None
    with post('/v1/chat/completions', data) as r:
        for line in r:
            if not line.startswith(b'data: '): continue
            raw=line[6:].strip()
            if raw==b'[DONE]': break
            j=json.loads(raw)
            if j.get('usage'): usage=j['usage']
            for c in j.get('choices',[]):
                d=c.get('delta',{}); s=(d.get('content') or '')+(d.get('reasoning') or d.get('reasoning_content') or '')
                if s:
                    now=time.perf_counter()
                    if first is None: first=now
                    last=now; chunks.append(now-start); text+=s
    end=time.perf_counter()
    if not usage or first is None: raise RuntimeError(f'No usage or text: {usage} {text}')
    n=usage['completion_tokens']
    return dict(tokens=n,prompt_tokens=usage['prompt_tokens'],ttft=first-start,elapsed=end-start,
                decode_tps=(n-1)/(last-first) if last>first else None,
                e2e_tps=n/(end-start),chunk_times=chunks,text=text,usage=usage)

short='Write a detailed numbered engineering checklist for building a reliable code analysis service. Explain testing, parsing, memory limits, error recovery, deployment, and observability. Give at least forty detailed items.'
medium='Review the following service design notes and write a detailed implementation plan with at least forty numbered steps.\n'+('The service parses source files, validates syntax, tracks dependencies, caches results, reports diagnostics, and isolates plugins.\n'*100)

result={'started':time.time(),'port':a.port,'tokens':a.tokens,'scenarios':[]}
def save(): pathlib.Path(a.output).write_text(json.dumps(result,indent=2))

# Warm batch 1, batch 2 and medium prefill paths before measurement.
request(short,24)
with concurrent.futures.ThreadPoolExecutor(2) as ex: list(ex.map(lambda _: request(short,24),range(2)))
request(medium,16)
specs=[('single_1',1,short),('single_2',1,short),('concurrent_2',2,short),('medium',1,medium)]
if a.wide: specs += [('concurrent_4',4,short),('concurrent_8',8,short)]
if a.prefill: specs += [('cold_long',1,'Run identifier '+str(time.time_ns())+'\n'+medium*4)]
for name, concurrency, prompt in specs:
    before=metrics(); start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as ex:
        rows=list(ex.map(lambda _:request(prompt,a.tokens),range(concurrency)))
    elapsed=time.perf_counter()-start; after=metrics()
    count=after['inter_token_latency_seconds_count']-before['inter_token_latency_seconds_count']
    seconds=after['inter_token_latency_seconds_sum']-before['inter_token_latency_seconds_sum']
    s=dict(name=name,concurrency=concurrency,elapsed=elapsed,aggregate_tps=sum(r['tokens'] for r in rows)/elapsed,
           median_decode_tps=statistics.median(r['decode_tps'] for r in rows),mean_server_itl=seconds/count if count else None,
           before=before,after=after,requests=rows)
    result['scenarios'].append(s); save()
    print(json.dumps({k:v for k,v in s.items() if k not in ['requests','before','after']}),flush=True)

checks=[]
for s in result['scenarios']:
    for index,r in enumerate(s['requests']):
        words=re.findall(r'\w+',r['text'].lower())
        longest=max((sum(1 for _ in g) for _,g in itertools.groupby(words)),default=0)
        checks.append(dict(kind='generation_not_repetitive',scenario=s['name'],request=index,passed=longest<12,longest_repeated_word_run=longest))
for prompt, expected in [('Return only the decimal integer: 17 multiplied by 19.','323'),('Ответь только числом: сколько будет 29 плюс 14?','43')]:
    data=dict(model=model,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=64,chat_template_kwargs={'enable_thinking':False})
    j=json.load(post('/v1/chat/completions',data)); answer=j['choices'][0]['message'].get('content','')
    checks.append(dict(prompt=prompt,answer=answer,passed=expected in answer))
data=dict(model=model,messages=[dict(role='user',content='Use the get_weather tool to look up the current weather in Paris.')],
          tools=[dict(type='function',function=dict(name='get_weather',description='Get the current weather for a city',parameters=dict(type='object',properties=dict(city=dict(type='string')),required=['city'])))],
          tool_choice='auto',temperature=0,max_tokens=96,chat_template_kwargs={'enable_thinking':False})
j=json.load(post('/v1/chat/completions',data)); message=j['choices'][0]['message']; calls=message.get('tool_calls') or []
checks.append(dict(kind='tool',answer=message,passed=bool(calls and calls[0]['function']['name']=='get_weather')))
result['checks']=checks
raw_metrics=urllib.request.urlopen(base+'/metrics').read().decode()
result['speculation_metrics']=[line for line in raw_metrics.splitlines() if line.startswith('vllm:') and 'spec_decode' in line]
result['finished']=time.time(); save()
print(json.dumps({'checks':checks}),flush=True)

if not all(c['passed'] for c in result['checks']):
    raise SystemExit('Benchmark correctness checks failed; inspect JSON')
