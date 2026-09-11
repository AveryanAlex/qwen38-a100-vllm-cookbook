import argparse, json, pathlib, time, urllib.request

p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19088);p.add_argument('--output',required=True);a=p.parse_args()
w=pathlib.Path(__file__).resolve().parents[1]/'data'
book=(w/'tale-of-two-cities-clean.txt').read_text()
markers=['QUARTZ-7193','LANTERN-4286','COPPER-8531','ORCHID-2649']
positions=[book.find('\n',int(len(book)*fraction))+1 for fraction in (.05,.35,.65,.95)]
for pos,code in reversed(list(zip(positions,markers))):
    book=book[:pos]+f'\n[Benchmark margin note: reference code {code}.]\n'+book[pos:]
prompt='Read the complete novel below. The four bracketed benchmark margin notes are annotations, not part of the novel.\n\n'+book+'''

Tasks:
1. Summarize the complete novel in approximately 500 words. Cover the major characters, causal plot developments, the French Revolution, and the ending accurately.
2. Answer briefly: Who marries Lucie? Who goes to the guillotine in whose place? What motivates that sacrifice? What happens to Madame Defarge? What is Jerry Cruncher's secret occupation? What is Dr Manette's prison designation? How is the Marquis connected to Darnay?
3. List the four reference codes from the benchmark margin notes in the order they appear. Do not include those annotations in the summary.
4. Quote one short passage from near the ending and explain how it supports your summary.
'''
body=dict(model='qwen38-flash-next-uncensored',messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=2048,chat_template_kwargs={'enable_thinking':False},stream=True,stream_options={'include_usage':True})
out=pathlib.Path(a.output); result=dict(started=time.time(),source='https://www.gutenberg.org/ebooks/98',book='A Tale of Two Cities',markers=markers,marker_character_positions=positions)
def save():out.write_text(json.dumps(result,indent=2))
save()
req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter();first=last=None;text='';usage=None;finish=None;chunks=[]
with urllib.request.urlopen(req,timeout=1800) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        raw=line[6:].strip()
        if raw==b'[DONE]':break
        j=json.loads(raw)
        if j.get('usage'):usage=j['usage']
        for choice in j.get('choices',[]):
            if choice.get('finish_reason'):finish=choice['finish_reason']
            d=choice.get('delta',{});s=(d.get('content') or '')+(d.get('reasoning') or '')
            if s:
                now=time.perf_counter()
                if first is None:
                    first=now;result['ttft_seconds']=first-start;save();print('First token after',first-start,'seconds',flush=True)
                last=now;chunks.append(now-start);text+=s
end=time.perf_counter()
if usage is None or first is None:raise RuntimeError('No usage or output')
result.update(usage=usage,elapsed_seconds=end-start,decode_tps=(usage['completion_tokens']-1)/(last-first),e2e_tps=usage['completion_tokens']/(end-start),
              approximate_prefill_tps=usage['prompt_tokens']/(first-start),answer=text,chunk_times=chunks,finish_reason=finish,
              all_markers_present=all(marker in text for marker in markers),markers_in_order=all(text.find(x)<text.find(y) for x,y in zip(markers,markers[1:])),finished=time.time())
save();out.with_suffix('.answer.md').write_text(text)
print(json.dumps({k:v for k,v in result.items() if k not in ['answer','chunk_times']}),flush=True)
print(text,flush=True)

if not result['all_markers_present'] or not result['markers_in_order'] or finish != 'stop':
    raise SystemExit('Book checks failed; inspect the saved answer')
