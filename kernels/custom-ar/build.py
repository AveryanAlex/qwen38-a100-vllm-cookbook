import pathlib
from torch.utils.cpp_extension import load
w=pathlib.Path(__file__).resolve().parent
module=load(name='q38_ar_probe',sources=[str(w/'probe.cu')],extra_include_paths=[str(w)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)
print('MODULE',module.__file__,flush=True)
