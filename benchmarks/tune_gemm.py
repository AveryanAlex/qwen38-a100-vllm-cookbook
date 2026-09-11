"""Measure SM80 dense decode kernels without changing any serving weights."""
import dataclasses, itertools, json, pathlib, statistics, time
import torch
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import SkinnyGemmConfig, shape_dynamic_skinny_gemm as skinny

torch.manual_seed(719)
assert torch.cuda.get_device_capability()==(8,0)
import argparse
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mtp',type=int,default=1);a=p.parse_args()
out=pathlib.Path(a.output)
flush=torch.empty(32*1024*1024,dtype=torch.float32,device='cuda')
rows=[]
shapes=[(4096,2560),(2560,1536),(24,2560),(3584,2560),(640,2560),(320,2560),(336,10240),(320,10240),(62080,2560)]
depth=a.mtp
token_counts=sorted({1,2,4,depth+1,2*(depth+1)})

def graph(fn, copies=1):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(copies): y=fn()
    return g,y

def measure(g,cold,copies=1):
    events=[]
    for _ in range(30):
        if cold: flush.zero_()
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(1 if cold else 10):g.replay()
        end.record(); events.append((start,end))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e)*1000/(copies*(1 if cold else 10)) for s,e in events)

for n,k in shapes:
    for m in token_counts:
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
        ref=x.float()@w.float().T
        base,base_y=graph(lambda:torch.nn.functional.linear(x,w))
        base_hot,_=graph(lambda:torch.nn.functional.linear(x,w),20)
        baseline={'hot_us':measure(base_hot,False,20),'cold_us':measure(base,True)}
        candidates=[]
        for block,outputs,vector in itertools.product((64,128),(1,2,4),(4,8)):
            if n%outputs or k%(block*vector) or k<2*block*vector:continue
            cfg=SkinnyGemmConfig(m,block,outputs,vector_width=vector,static_k=k)
            entry={'config':dataclasses.asdict(cfg)}
            try:
                fn=lambda: skinny(x,w,cfg)
                y=fn(); torch.cuda.synchronize()
                error=(y.float()-ref).abs(); base_error=(base_y.float()-ref).abs()
                entry['rms_error']=error.square().mean().sqrt().item()
                entry['max_error']=error.max().item()
                entry['base_rms_error']=base_error.square().mean().sqrt().item()
                if entry['rms_error']>max(0.001,entry['base_rms_error']*1.1):
                    entry['error']='Numerical error exceeds BF16 baseline tolerance'
                else:
                    g,_=graph(fn);g_hot,_=graph(fn,20)
                    entry.update(hot_us=measure(g_hot,False,20),cold_us=measure(g,True));del g,g_hot
            except Exception as e: entry['error']=str(e)[:1500]
            candidates.append(entry)
        good=[c for c in candidates if 'hot_us' in c and c['hot_us']<baseline['hot_us']*.9 and c['cold_us']<baseline['cold_us']*.9]
        best=min(good,key=lambda c:c['hot_us']+c['cold_us']) if good else None
        row={'n':n,'k':k,'m':m,'baseline':baseline,'best':best,'candidates':candidates}
        rows.append(row);out.write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
        del base,base_hot,x,w,ref,base_y
        torch.cuda.empty_cache()
