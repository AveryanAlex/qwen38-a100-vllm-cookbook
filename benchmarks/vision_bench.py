"""Grounded vision checks with original Earthrise and altered-image controls."""
import argparse,base64,concurrent.futures,hashlib,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--qwen3-normalization',action='store_true',help='Override image normalization per request for A/B checks');p.add_argument('--port',type=int,default=19088);p.add_argument('--fixtures',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent/'fixtures/vision');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
source=json.loads((a.fixtures/'source.json').read_text())
for name,expected in source['files'].items():
 if hashlib.sha256((a.fixtures/name).read_bytes()).hexdigest()!=expected:
  raise SystemExit('Fixture hash mismatch: '+name)


def request(name,files,prompt):
 content=[]
 for file in files:
  raw=(a.fixtures/file).read_bytes();content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode()}})
 content.append({'type':'text','text':prompt})
 body={'model':'qwen38-flash-next-uncensored','messages':[{'role':'user','content':content}], 'temperature':0,'max_tokens':600,'chat_template_kwargs':{'enable_thinking':False}}
 if a.qwen3_normalization:body['mm_processor_kwargs']={'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5]}
 req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 start=time.perf_counter()
 try:
  with urllib.request.urlopen(req,timeout=600) as response:r=json.load(response)
  answer=r['choices'][0]['message'].get('content');row={'case':name,'files':files,'prompt':prompt,'answer':answer,'usage':r.get('usage'),'finish_reason':r['choices'][0]['finish_reason'],'elapsed_seconds':time.perf_counter()-start}
 except Exception as e:
  row={'case':name,'files':files,'prompt':prompt,'error':str(e)}
  if hasattr(e,'read'):row['error_body']=e.read().decode(errors='replace')
 return row

question='Describe only what is visible in this image. Where is the gray rocky terrain: top or bottom? Is a blue-and-white planet visible? What colors are the background and the planet? Are any people, spacecraft, trees, or printed labels visible? Be specific and do not invent details.'
cases=[('original',['earthrise-1024.png'],question),('original-repeat',['earthrise-1024.png'],question),('vertical-flip',['earthrise-flipped.png'],question),('lunar-crop',['lunar-crop.png'],question),('marked',['earthrise-marked.png'],'Describe the scene and the added graphic elements. Transcribe the exact printed label. What color and shape is the added object at the upper right? Distinguish artificial overlays from the photograph.'),('two-images',['earthrise-1024.png','lunar-crop.png'],'Compare image 1 and image 2 in that order. Which image shows a blue-and-white planet? Which image is a close crop of gray rocky ground? Are people visible in either? Explain the visible difference.')]
result={'normalization_override':a.qwen3_normalization,'source':json.loads((a.fixtures/'source.json').read_text()),'cases':[]}
for case in cases:
 row=request(*case);result['cases'].append(row);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
# Distinct images processed concurrently, then revisit a prior image to expose
# accidental cache reuse across image identities.
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 rows=list(pool.map(lambda c:request(*c),[('concurrent-original',['earthrise-1024.png'],question),('concurrent-crop',['lunar-crop.png'],question)]))
result['cases'].extend(rows);a.output.write_text(json.dumps(result,indent=2))
for row in rows:print(json.dumps(row),flush=True)
if any('error' in row for row in result['cases']):raise SystemExit('Vision request errors; inspect saved JSON')
print('Requests completed. Manual grounding review is required.',flush=True)
