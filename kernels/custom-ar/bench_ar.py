import itertools,json,os,pathlib,statistics,time
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

w=pathlib.Path(__file__).resolve().parent
output=pathlib.Path(os.environ.get('AR_OUTPUT_DIR','/state/ar'));output.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('gloo')
module=load(name='q38_ar_probe',sources=[str(w/'probe.cu')],extra_include_paths=[str(w)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
assert module.meta_size()==ops.meta_size()
maximum=8*1024*1024
comm=CustomAllreduce(dist.group.WORLD,rank,max_size=maximum,max_all_gather_size=maximum,max_reduce_scatter_size=maximum)
assert not comm.disabled
signals=CustomAllreduce.create_shared_buffer(module.meta_size()+maximum,group=dist.group.WORLD)
peers=torch.tensor(comm.buffer_ptrs+[0]*12,dtype=torch.int64,device=rank)
rows=[]

def graph(fn,repeats=20,original=False):
    g=torch.cuda.CUDAGraph()
    torch.cuda.synchronize();dist.barrier()
    if original:
        with comm.capture():
            with torch.cuda.graph(g):
                for _ in range(repeats):fn()
    else:
        with torch.cuda.graph(g):
            for _ in range(repeats):fn()
    for _ in range(3):g.replay()
    torch.cuda.synchronize();dist.barrier()
    return g

def measure(g,repeats=20):
    events=[]
    for _ in range(12):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();g.replay();end.record();events.append((start,end))
    torch.cuda.synchronize()
    local=statistics.median(a.elapsed_time(b)*1000/repeats for a,b in events)
    values=[None]*4;dist.all_gather_object(values,local)
    return max(values)

sizes=[5120,10240,20480,40960,81920,163840,327680,524288,1048576,2097152,4194304]
for size in sizes:
    x=module.alias_buffer(comm.buffer_ptrs[rank],size//2,rank,0)
    values=(torch.arange(x.numel(),device=rank)%17).to(torch.bfloat16)
    x.copy_(values+rank)
    expected=values*4+6
    out=torch.empty_like(x);native=torch.empty_like(x)
    torch.cuda.synchronize();dist.barrier()
    baseline=graph(lambda:comm.all_reduce(x,out=native,registered=True),original=True)
    torch.testing.assert_close(native,expected,rtol=0,atol=0)
    baseline_us=measure(baseline)
    candidates=[]
    for algo,threads,blocks in itertools.product((1,2),(64,128,256,512),(4,8,16,36)):
        g=graph(lambda:module.launch(peers,signals,out,rank,threads,blocks,algo))
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
        us=measure(g)
        candidates.append(dict(algo=algo,threads=threads,block_limit=blocks,us=us))
        del g
    row=dict(bytes=size,baseline_us=baseline_us,best=min(candidates,key=lambda c:c['us']),candidates=candidates)
    rows.append(row)
    if rank==0:
        (output/'kernel-timings.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
    del baseline,out,native,values,expected,x
    torch.cuda.empty_cache()
dist.barrier();comm.close();CustomAllreduce.free_shared_buffer(signals,rank=rank)
dist.destroy_process_group()
