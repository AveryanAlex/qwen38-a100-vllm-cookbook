"""Verify current API context limits without changing the server configuration."""
import datetime,hashlib,json,pathlib,time,urllib.error,urllib.request
from transformers import AutoTokenizer

import argparse
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=19088);a=p.parse_args()
out=a.output;out.parent.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained('/model')
base=f'http://127.0.0.1:{a.port}'
model='qwen38-flash-next-uncensored'
def token_ids(messages):
 rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 return tok.encode(rendered,add_special_tokens=False)

result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':model,'changed_server_configuration':False,'tests':[]}

def send(label,text,limit):
 messages=[{'role':'user','content':text}]
 ids=token_ids(messages)
 body={'model':model,'messages':messages,'temperature':0,'max_tokens':limit,'chat_template_kwargs':{'enable_thinking':False}}
 encoded=json.dumps(body).encode()
 row={'case':label,'prompt_tokens_local_template':len(ids),'max_output_tokens':limit,'total_token_budget':len(ids)+limit,'prompt_utf8_bytes':len(text.encode()),'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()}
 start=time.perf_counter()
 try:
  req=urllib.request.Request(base+'/v1/chat/completions',data=encoded,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as response:row.update(http_status=response.status,response=json.load(response))
 except urllib.error.HTTPError as error:
  row.update(http_status=error.code,response=json.loads(error.read()))
 row['elapsed_seconds']=time.perf_counter()-start
 result['tests'].append(row);out.write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
 return row

result['advertised_model']=json.load(urllib.request.urlopen(base+'/v1/models',timeout=10))['data'][0]
# Repeated one-token units create a precisely counted request. This is an API
# admission test, not a retrieval/quality test: the current cap should reject it.
base_messages=[{'role':'user','content':'x'}]
base_count=len(token_ids(base_messages))
probe=len(token_ids([{'role':'user','content':'x'+(' x'*100)}]))
assert probe-base_count==100,(base_count,probe)
for label,target_input,output in [('one_over_native',262144,1),('one_million_total',1048576-64,64)]:
 text='x'+(' x'*(target_input-base_count))
 row=send(label,text,output)
 assert row['prompt_tokens_local_template']==target_input
 assert row['http_status']==400,row
control=send('small_health_control','Return only the integer: 17 multiplied by 19.',32)
assert control['http_status']==200
assert '323' in control['response']['choices'][0]['message']['content']
result['health_after']=urllib.request.urlopen(base+'/health',timeout=10).status
out.write_text(json.dumps(result,indent=2));print('Saved',out,flush=True)
