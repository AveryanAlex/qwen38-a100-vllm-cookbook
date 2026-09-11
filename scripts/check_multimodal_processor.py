"""Check actual HF image/video processor paths without loading model weights."""
import numpy as np
from PIL import Image
from transformers import Qwen3VLProcessor
import json
from cookbook import create_command
# Derive processor flags from the actual portable launcher, not a second copy.
cfg={'model_dir':'/model','state_dir':'/state','container_name':'processor-check','port':19088,'tuned_all_reduce':False,'vision':True}
args=create_command(cfg)
kwargs=json.loads(args[args.index('--mm-processor-kwargs')+1])
p=Qwen3VLProcessor.from_pretrained('/model',**kwargs)
meta={'total_num_frames':11,'fps':2.0,'duration':5.5,'video_backend':'opencv','frames_indices':list(range(11))}
# Same dimensions and kwargs used by the failed runtime's video dummy warmup.
video=np.full((11,864,864,3),255,dtype=np.uint8)
r=p(text='<|vision_start|><|video_pad|><|vision_end|>',videos=[[video]],video_metadata=[[meta]],do_sample_frames=False,return_tensors='pt',**kwargs)
print('VIDEO',[(k,list(v.shape)) for k,v in r.items() if hasattr(v,'shape')],flush=True)
assert r['pixel_values_videos'].shape[-1]==1536
r=p(text='<|vision_start|><|image_pad|><|vision_end|>',images=[Image.new('RGB',(2400,2400))],return_tensors='pt',**kwargs)
print('IMAGE',[(k,list(v.shape)) for k,v in r.items() if hasattr(v,'shape')],flush=True)
assert r['pixel_values'].shape== (4096,1536)
print('BOTH PROCESSOR PATHS PASSED',flush=True)
