#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "custom_all_reduce_tuned.cuh"

// This extension is tied to the pinned vLLM image and exact public class header.
// Check the object layout before using its inline capture/registration logic.
void validate_abi(int64_t pointer, int rank, int world,
                  const std::vector<int64_t>& signals, int64_t rank_data,
                  int64_t rank_data_bytes, int64_t registered_buffer) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(pointer);
  TORCH_CHECK(fa->rank_ == rank && fa->world_size_ == world && world == 4);
  TORCH_CHECK(fa->fully_connected_ && signals.size() == 4);
  for (int i=0;i<4;++i) TORCH_CHECK(reinterpret_cast<int64_t>(fa->sg_.signals[i]) == signals[i]);
  TORCH_CHECK(reinterpret_cast<int64_t>(fa->self_sg_) == signals[rank]);
  TORCH_CHECK(reinterpret_cast<int64_t>(fa->d_rank_data_end_) == rank_data + rank_data_bytes);
  TORCH_CHECK(reinterpret_cast<int64_t>(fa->d_rank_data_base_) >= rank_data);
  TORCH_CHECK(reinterpret_cast<int64_t>(fa->d_rank_data_base_) < rank_data + rank_data_bytes);
  auto it=fa->buffers_.find(reinterpret_cast<void*>(registered_buffer));
  TORCH_CHECK(it != fa->buffers_.end());
  auto address=reinterpret_cast<int64_t>(it->second);
  TORCH_CHECK(address >= rank_data && address < rank_data + rank_data_bytes);
}

void tuned_all_reduce(int64_t pointer, torch::Tensor inp, torch::Tensor out,
                      int64_t registered_buffer, int64_t buffer_bytes) {
  c10::cuda::CUDAGuard guard(inp.device());
  TORCH_CHECK(inp.scalar_type()==torch::kBFloat16 && out.scalar_type()==torch::kBFloat16);
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous() && inp.numel()==out.numel());
  auto* fa=reinterpret_cast<vllm::CustomAllreduce*>(pointer);
  TORCH_CHECK(fa->world_size_==4 && fa->fully_connected_);
  auto stream=c10::cuda::getCurrentCUDAStream(inp.get_device()).stream();
  const int64_t bytes=inp.numel()*2;
  void* input=inp.data_ptr();
  if (registered_buffer) {
    TORCH_CHECK(bytes<=buffer_bytes);
    input=reinterpret_cast<void*>(registered_buffer);
    CUDACHECK(cudaMemcpyAsync(input,inp.data_ptr(),bytes,cudaMemcpyDeviceToDevice,stream));
  }
  int threads=512;
  if (bytes<=10240) threads=64;
  else if (bytes<=40960) threads=128;
  else if (bytes<=81920) threads=256;
  else if (bytes<=163840) threads=128;
  else if (bytes<=327680) threads=256;
  else if (bytes<=1048576) threads=128;
  fa->allreduce<nv_bfloat16>(stream,reinterpret_cast<nv_bfloat16*>(input),
                            reinterpret_cast<nv_bfloat16*>(out.data_ptr()),inp.numel(),threads,36);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("validate_abi",&validate_abi);
  m.def("all_reduce",&tuned_all_reduce);
}
