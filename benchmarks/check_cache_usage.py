"""Verify cached-token usage fields on Chat Completions and Responses."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--port', type=int, default=19088)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{a.port}'
stamp = str(time.time_ns())
prompt = ('Cache usage field probe ' + stamp + '.\n' +
          'A library stores books, serves readers, and keeps an accurate catalogue.\n' * 140 +
          '\nReply only OK.')
result = {'started': time.time(), 'probe_id': stamp,
          'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'cases': []}


def save():
    a.output.write_text(json.dumps(result, indent=2) + '\n')


def request(endpoint, stream):
    body = dict(model='qwen38-flash-next-uncensored', temperature=0, stream=stream,
                chat_template_kwargs={'enable_thinking': False})
    if endpoint == 'chat/completions':
        body.update(messages=[dict(role='user', content=prompt)], max_tokens=32)
        if stream:
            body['stream_options'] = {'include_usage': True}
    else:
        body.update(input=prompt, max_output_tokens=64, store=False)
    req = urllib.request.Request(base + '/v1/' + endpoint, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        usage = None
        final_event = None
        if not stream:
            j = json.load(response)
            usage = j.get('usage')
        else:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:].strip()
                if raw == b'[DONE]':
                    break
                j = json.loads(raw)
                if j.get('error'):
                    raise RuntimeError(j['error'])
                if endpoint == 'chat/completions' and j.get('usage'):
                    usage = j['usage']
                    final_event = 'usage_chunk'
                if endpoint == 'responses' and j.get('type') in ('response.completed', 'response.incomplete'):
                    final_event = j['type']
                    usage = j['response'].get('usage')
        return dict(http_status=response.status, usage=usage, final_event=final_event,
                    elapsed_seconds=time.perf_counter()-start)


for endpoint in ['chat/completions', 'responses']:
    for name, stream in [('first', False), ('repeat', False), ('stream_repeat', True)]:
        row = request(endpoint, stream)
        row.update(endpoint=endpoint, case=name, stream=stream)
        usage = row['usage'] or {}
        details = usage.get('prompt_tokens_details' if endpoint == 'chat/completions' else 'input_tokens_details') or {}
        row['cached_tokens'] = details.get('cached_tokens')
        row['passed'] = (isinstance(row['cached_tokens'], int) and
                         row['cached_tokens'] >= 0 and
                         (name == 'first' or row['cached_tokens'] > 0))
        result['cases'].append(row)
        save()
        print(json.dumps(row), flush=True)
        if not row['passed']:
            raise SystemExit('Cache usage field check failed')
result['health_after'] = urllib.request.urlopen(base + '/health', timeout=10).status
result['finished'] = time.time()
save()
