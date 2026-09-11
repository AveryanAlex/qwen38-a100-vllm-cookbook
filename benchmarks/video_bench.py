"""Check video decoding, temporal order, mixed input and counts above two."""
import argparse,base64,hashlib,json,pathlib,time,urllib.request
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=19088);p.add_argument('--fixtures',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent/'fixtures/vision');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)

def part(file):
 path=a.fixtures/file
 if file=='people-6s.mp4' and not path.exists():path=pathlib.Path(__file__).resolve().parents[1]/'data/video/people-6s.mp4'
 raw=path.read_bytes();kind='video' if file.endswith('.mp4') else 'image';mime='video/mp4' if kind=='video' else 'image/png'
 return {'type':kind+'_url',kind+'_url':{'url':'data:'+mime+';base64,'+base64.b64encode(raw).decode()}}

order='Describe the changes over time, not just the first frame. List the large printed words in their exact chronological order. State the square colors and positions in chronological order. Describe the unchanged background briefly.'
cases=[('forward',['sequence.mp4'],order),('reverse',['sequence-reversed.mp4'],order),('real-people',['people-6s.mp4'],'Describe the visible setting and actions across this video. Are people moving or stationary? Mention the road, grass, buildings and any visible parked vehicles if present. Do not identify individuals or invent events.'),('three-videos',['sequence.mp4','sequence-reversed.mp4','sequence.mp4'],'There are three videos in order. For EACH video list the large printed words in their chronological order. Be explicit about video 1, 2 and 3.'),('three-images',['earthrise-1024.png','lunar-crop.png','earthrise-marked.png'],'For image 1, 2 and 3 in order: say whether Earth is visible, and transcribe any printed label. Which image has the red square?'),('mixed',['sequence.mp4','earthrise-marked.png'],'First describe the sequence of large printed words in the video. Then transcribe the label in the still image and identify its added shape and color.')]
result={'cases':[]}
for name,files,prompt in cases:
 body={'model':'qwen38-flash-next-uncensored','messages':[{'role':'user','content':[*(part(f) for f in files),{'type':'text','text':prompt}]}],'temperature':0,'max_tokens':650,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});start=time.perf_counter();row={'case':name,'files':files,'prompt':prompt}
 try:
  response=json.load(urllib.request.urlopen(req,timeout=600));row.update(answer=response['choices'][0]['message']['content'],usage=response['usage'],finish_reason=response['choices'][0]['finish_reason'],elapsed_seconds=time.perf_counter()-start)
 except Exception as e:
  row.update(error=str(e),body=e.read().decode(errors='replace') if hasattr(e,'read') else '')
 result['cases'].append(row);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(row),flush=True)
 if 'error' in row:break
if any('error' in row for row in result['cases']):raise SystemExit('Video/multi-input check failed; inspect JSON')
print('Requests completed; review temporal grounding.',flush=True)
