"""Exact 512K admission and long-book retrieval test, run in the serving image."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--port', type=int, default=19088)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parent / 'fixtures')
p.add_argument('--prepare-only', action='store_true')
p.add_argument('--archive-id', default='512K-20260911', help='Change this to avoid reuse of a previous corpus prefix')
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained('/model')
base = f'http://127.0.0.1:{a.port}'
limit = 524288
# This is a tokenizer warning threshold only; the server limit is checked below.
tok.model_max_length = limit
max_output = 2048
target = limit - max_output
sources = {}
expected_sources = json.loads((a.fixtures / 'extended-context-sources.json').read_text())

def clean(name):
    raw = gzip.decompress((a.fixtures / (name + '.txt.gz')).read_bytes())
    sources[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    assert sources[name]['sha256'] == expected_sources[name]['raw_sha256'], 'Book fixture changed: ' + name
    text = raw.decode('utf-8-sig')
    text = re.split(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n', text, maxsplit=1)[-1]
    return re.split(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', text, maxsplit=1)[0].strip()

moby = tok.encode(clean('moby-dick'), add_special_tokens=False)
tale = clean('tale-of-two-cities')
pride = tok.encode(clean('pride-and-prejudice'), add_special_tokens=False)
front = (f'Validation archive {a.archive_id}. The bracketed ledger notes are benchmark annotations, not part of the books.\n'
         'DOCUMENT A: excerpt of Moby Dick.\n' + tok.decode(moby[:170000]) +
         '\nEND DOCUMENT A.\nDOCUMENT B: Complete A Tale of Two Cities.\n' + tale +
         '\nEND DOCUMENT B.\nDOCUMENT C: excerpt of Pride and Prejudice.\n')
question = '''\nEND DOCUMENT C.
Tasks:
1. List all eight LEDGER reference codes from the archive in the order they appear. Copy them exactly.
2. Summarize only DOCUMENT B, A Tale of Two Cities, in approximately 400 words, covering the main plot, sacrifice and ending. Do not mix in the other novels or benchmark notes.
3. Answer briefly: Who marries Lucie? Who dies at the guillotine in whose place? Why does he sacrifice himself? What happens to Madame Defarge? What is Jerry Cruncher's secret occupation? What is Dr Manette's prison designation? How is the Marquis related to Darnay?
'''
markers = ['LEDGER-' + hashlib.sha256(f'512K-validation-20260911-{i}'.encode()).hexdigest()[:12].upper() for i in range(8)]

def build(tail_tokens):
    corpus = front + tok.decode(pride[:tail_tokens])
    positions = [int(len(corpus) * fraction) for fraction in (.02, .15, .29, .43, .57, .71, .85, .98)]
    for pos, code in reversed(list(zip(positions, markers))):
        # Insert at a nearby newline to preserve words in the source novels.
        pos = corpus.find('\n', pos)
        if pos < 0:
            pos = len(corpus)
        corpus = corpus[:pos] + f'\n[Benchmark ledger note: {code}.]\n' + corpus[pos:]
    return corpus + question


def count(text):
    rendered = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    return len(tok.encode(rendered, add_special_tokens=False))

n = min(len(pride), target - len(tok.encode(front + question, add_special_tokens=False)) - 250)
for _ in range(12):
    prompt = build(n)
    size = count(prompt)
    if size == target:
        break
    n += target - size
    assert 0 < n <= len(pride), (n, len(pride), size)
else:
    raise RuntimeError('Could not construct exact context length')
assert count(prompt) == target
positions = [len(tok.encode(prompt[:prompt.index(code)], add_special_tokens=False)) for code in markers]
result = {'archive_id': a.archive_id, 'sources': sources, 'prompt_tokens_local': target, 'max_output_tokens': max_output,
          'total_token_budget': limit, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
          'markers': markers, 'marker_content_token_offsets': positions,
          'complete_book': 'A Tale of Two Cities', 'book_start_content_token_offset': len(tok.encode(front[:front.index('DOCUMENT B:')], add_special_tokens=False)),
          'book_end_content_token_offset': len(tok.encode(front[:front.index('END DOCUMENT B.')], add_special_tokens=False)),
          'started': time.time()}

def save():
    a.output.write_text(json.dumps(result, indent=2) + '\n')

save()
print('Prepared', target, 'prompt tokens; marker positions', positions, flush=True)
if a.prepare_only:
    raise SystemExit(0)


def metrics():
    raw = urllib.request.urlopen(base + '/metrics', timeout=30).read().decode()
    return {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1]) for line in raw.splitlines()
            if line.startswith('vllm:') and any(n in line for n in ('generation_tokens_total', 'prefix_cache_', 'kv_offload_load_bytes_total', 'num_preemptions_total'))}


def request(text, output, stream=True):
    body = dict(model='qwen38-flash-next-uncensored', messages=[dict(role='user', content=text)],
                temperature=0, max_tokens=output, chat_template_kwargs={'enable_thinking': False}, stream=stream)
    if stream:
        body['stream_options'] = {'include_usage': True}
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            if not stream:
                return dict(http_status=response.status, response=json.load(response))
            first = last = usage = finish = None
            answer = ''
            for line in response:
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
                    text_chunk = c.get('delta', {}).get('content') or ''
                    if text_chunk:
                        last = time.perf_counter()
                        if first is None:
                            first = last
                            print('First token:', first-start, flush=True)
                        answer += text_chunk
            if usage is None or first is None:
                raise RuntimeError('Missing answer or usage')
            return dict(http_status=response.status, usage=usage, answer=answer, finish_reason=finish,
                        ttft_seconds=first-start, elapsed_seconds=time.perf_counter()-start,
                        decode_tps=(usage['completion_tokens']-1)/(last-first) if last>first else None)
    except urllib.error.HTTPError as e:
        return dict(http_status=e.code, error_body=e.read().decode())

result['advertised_model'] = json.load(urllib.request.urlopen(base + '/v1/models'))['data'][0]
assert result['advertised_model']['max_model_len'] == limit
# Exact prompt plus one extra requested output token must be rejected.
result['one_over_limit'] = request(prompt, max_output + 1, stream=False)
save()
assert result['one_over_limit']['http_status'] == 400, result['one_over_limit']
for phase in ['cold', 'cached']:
    result[phase + '_metrics_before'] = metrics()
    result[phase] = request(prompt, max_output)
    result[phase + '_metrics_after'] = metrics()
    save()
    row = result[phase]
    assert row['http_status'] == 200, row
    assert row['usage']['prompt_tokens'] == target, row['usage']
    answer = row['answer']
    row['markers_present'] = all(code in answer for code in markers)
    row['markers_ordered'] = row['markers_present'] and all(answer.index(x) < answer.index(y) for x,y in zip(markers,markers[1:]))
    save()
    print(phase, row['usage'], 'markers:', row['markers_ordered'], flush=True)
    assert row['finish_reason'] == 'stop', 'Full-context generation did not finish normally'
result['answers_identical'] = result['cold']['answer'] == result['cached']['answer']
result['health_after'] = urllib.request.urlopen(base + '/health').status
result['finished'] = time.time()
save()
print('Full-context requests completed; review saved summaries and factual answers.', flush=True)

if not all(result[phase]['markers_ordered'] for phase in ['cold', 'cached']):
    raise SystemExit('Full-context order checks failed; both answers preserved for review')
