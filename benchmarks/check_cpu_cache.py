"""Verify CPU restores after natural GPU prefix eviction; no development API needed."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--port', type=int, default=19088)
p.add_argument('--book', type=Path, default=Path(__file__).resolve().parents[1] / 'data/tale-of-two-cities-clean.txt')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--churn-requests', type=int, default=6,
               help='Distinct ~190K prompts; default exceeds the tested GPU cache capacity')
a = p.parse_args()
base = f'http://127.0.0.1:{a.port}'
stamp = uuid.uuid4().hex
book = a.book.read_text()
markers = ['QUARTZ-7193', 'LANTERN-4286', 'COPPER-8531', 'ORCHID-2649']
positions = [book.find('\n', int(len(book) * f)) + 1 for f in (.05, .35, .65, .95)]
for pos, code in reversed(list(zip(positions, markers))):
    book = book[:pos] + f'\n[Benchmark margin note: reference code {code}.]\n' + book[pos:]
prompt = f'Benchmark document identity: {stamp}. Read the complete novel below. Margin notes are test annotations, not part of the story.\n\n' + book + '''

Summarize the novel in approximately 500 words, including major characters, the French Revolution and the ending.
Then answer briefly: Who marries Lucie? Who goes to the guillotine in whose place? What motivates that sacrifice? What happens to Madame Defarge? What is Jerry Cruncher's secret occupation? What is Dr Manette's prison designation? How is the Marquis connected to Darnay?
Finally list the four reference codes from the margin notes in their order of appearance.
'''
result = {'started': time.time(), 'document_identity': stamp, 'markers': markers,
          'source': 'https://www.gutenberg.org/ebooks/98', 'churn': []}
a.output.parent.mkdir(parents=True, exist_ok=True)


def save():
    a.output.write_text(json.dumps(result, indent=2) + '\n')


def metrics():
    with urllib.request.urlopen(base + '/metrics', timeout=30) as r:
        raw = r.read().decode()
    # Preserve labels, including TP rank; sum counters only when reporting.
    return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
            for line in raw.splitlines()
            if line.startswith('vllm:') and any(n in line for n in
                ('kv_offload_', 'prefix_cache_', 'generation_tokens_total', 'num_preemptions_total'))}


def counter(snapshot, name):
    return sum(v for k, v in snapshot.items() if k.split('{')[0] == name)


def request(text, max_tokens):
    body = dict(model='qwen38-flash-next-uncensored',
                messages=[dict(role='user', content=text)], temperature=0,
                max_tokens=max_tokens, chat_template_kwargs={'enable_thinking': False},
                stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(base + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    first = last = usage = finish = None
    answer = ''
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            j = json.loads(raw)
            if j.get('error'):
                raise RuntimeError(j['error'])
            if j.get('usage'):
                usage = j['usage']
            for c in j.get('choices', []):
                finish = c.get('finish_reason') or finish
                s = c.get('delta', {}).get('content') or ''
                if s:
                    last = time.perf_counter()
                    if first is None:
                        first = last
                    answer += s
    if usage is None or first is None:
        raise RuntimeError('Missing usage or answer')
    return dict(usage=usage, answer=answer, finish_reason=finish,
                ttft_seconds=first-start, elapsed_seconds=time.perf_counter()-start,
                decode_tps=(usage['completion_tokens']-1)/(last-first) if last > first else None)


result['metrics_before'] = metrics()
result['cold'] = request(prompt, 2048)
save()
print('Cold book:', result['cold']['ttft_seconds'], flush=True)
for i in range(a.churn_requests):
    # Identity differs near the very start, preventing book-prefix reuse.
    row = request(f'Distinct churn document {i} {stamp}.\n' + book + '\nReply OK.', 1)
    result['churn'].append(row)
    save()
    print('Churn', i, row['usage']['prompt_tokens'], row['ttft_seconds'], flush=True)
result['metrics_before_restore'] = metrics()
result['restored'] = request(prompt, 2048)
result['metrics_after_restore'] = metrics()
result['restored_bytes'] = counter(result['metrics_after_restore'], 'vllm:kv_offload_load_bytes_total') - counter(result['metrics_before_restore'], 'vllm:kv_offload_load_bytes_total')
for field, metric in [('gpu_hit_tokens', 'vllm:prefix_cache_hits_total'),
                      ('external_hit_tokens', 'vllm:external_prefix_cache_hits_total'),
                      ('server_generated_tokens_during_restore', 'vllm:generation_tokens_total')]:
    result[field] = counter(result['metrics_after_restore'], metric) - counter(result['metrics_before_restore'], metric)
result['answers_identical'] = result['cold']['answer'] == result['restored']['answer']
result['checks'] = {}
for phase in ['cold', 'restored']:
    answer = result[phase]['answer']
    result['checks'][phase] = (result[phase]['finish_reason'] == 'stop'
                               and all(m in answer for m in markers)
                               and all(answer.find(x) < answer.find(y) for x, y in zip(markers, markers[1:])))
result['checks']['cpu_load_observed'] = result['restored_bytes'] > 0
result['checks']['external_cache_hit_observed'] = result['external_hit_tokens'] > 0
result['checks']['gpu_prefix_evicted'] = result['gpu_hit_tokens'] == 0
result['finished'] = time.time()
save()
print(json.dumps({'checks': result['checks'], 'restored_bytes': result['restored_bytes'],
                  'cold_ttft': result['cold']['ttft_seconds'],
                  'restored_ttft': result['restored']['ttft_seconds']}), flush=True)
if not all(result['checks'].values()):
    raise SystemExit('CPU cache verification failed; inspect results. Book facts require manual review.')
