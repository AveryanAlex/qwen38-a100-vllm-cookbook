import json,os,pathlib,statistics
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

w=pathlib.Path(__file__).resolve().parent
output=pathlib.Path(os.environ.get('AR_OUTPUT_DIR','/state/ar'));output.mkdir(parents=True,exist_ok=True)
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('gloo')
probe=load(name='q38_ar_probe',sources=[str(w/'probe.cu')],extra_include_paths=[str(w)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
tuned=load(name='q38_ar_tuned',sources=[str(w/'tuned.cu')],extra_include_paths=[str(w)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)
comm=CustomAllreduce(dist.group.WORLD,rank,max_size=8*1024*1024,max_all_gather_size=8*1024*1024,max_reduce_scatter_size=8*1024*1024)
assert not comm.disabled
tuned.validate_abi(comm._ptr,rank,4,comm.meta_ptrs,comm.rank_data.data_ptr(),comm.rank_data.numel(),comm.buffer_ptrs[rank])
rows=[]
for size in [5120,10240,20480,40960,81920,163840,327680,524288,1048576,2097152,4194304]:
    x=probe.alias_buffer(comm.buffer_ptrs[rank],size//2,rank,0)
    y_ref=torch.empty_like(x);y_new=torch.empty_like(x)
    values=(torch.arange(x.numel(),device=rank)%17).to(torch.bfloat16)
    x.copy_(values+rank);torch.cuda.synchronize();dist.barrier()
    graphs=[]
    for fn in [lambda:comm.all_reduce(x,out=y_ref,registered=True),lambda:tuned.all_reduce(comm._ptr,x,y_new,0,0)]:
        g=torch.cuda.CUDAGraph()
        with comm.capture():
            with torch.cuda.graph(g):
                for _ in range(20):fn()
        graphs.append(g)
    # Mutate inputs between replays and alternate implementations, including
    # different block counts sharing the same synchronization state.
    for phase in range(8):
        x.copy_(values+rank+phase)
        graphs[0].replay();graphs[1].replay()
        expected=values*4+6+phase*4
        torch.testing.assert_close(y_ref,expected,atol=0,rtol=0)
        torch.testing.assert_close(y_new,expected,atol=0,rtol=0)
    events=[[],[]]
    for _ in range(12):
        for index in [0,1,1,0]:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();graphs[index].replay();end.record();events[index].append((start,end))
    torch.cuda.synchronize()
    local=[statistics.median(a.elapsed_time(b)*1000/20 for a,b in pairs) for pairs in events]
    all_values=[None]*4;dist.all_gather_object(all_values,local)
    before=max(v[0] for v in all_values);after=max(v[1] for v in all_values)
    row=dict(bytes=size,stock_us=before,tuned_us=after,ratio=before/after,integer_checks=True)
    rows.append(row)
    if rank==0:print(json.dumps(row),flush=True);(output/'verified-timings.json').write_text(json.dumps(rows,indent=2))
    del graphs,x,y_ref,y_new,values,expected;torch.cuda.empty_cache()

random_checks=[]
for size in [10240,524288,1048576]:
    x=probe.alias_buffer(comm.buffer_ptrs[rank],size//2,rank,0);y_ref=torch.empty_like(x);y_new=torch.empty_like(x)
    for seed in range(4):
        generator=torch.Generator().manual_seed(700+seed)
        inputs=[(torch.randn(x.numel(),generator=generator)*2**(i-2)).to(torch.bfloat16) for i in range(4)]
        expected=sum(t.float() for t in inputs).to(device=rank,dtype=torch.bfloat16)
        x.copy_(inputs[rank]);torch.cuda.synchronize();dist.barrier()
        comm.all_reduce(x,out=y_ref,registered=True)
        tuned.all_reduce(comm._ptr,x,y_new,0,0)
        torch.testing.assert_close(y_ref,expected,rtol=1/128,atol=.001)
        torch.testing.assert_close(y_new,expected,rtol=1/128,atol=.001)
        if size<524288:torch.testing.assert_close(y_ref,y_new,rtol=0,atol=0)
        random_checks.append(dict(bytes=size,seed=seed,passed=True))
if rank==0:(output/'random-correctness.json').write_text(json.dumps(random_checks,indent=2));print('ALL NUMERICAL CHECKS PASSED',flush=True)
torch.cuda.synchronize();dist.barrier();comm.close();dist.destroy_process_group()
