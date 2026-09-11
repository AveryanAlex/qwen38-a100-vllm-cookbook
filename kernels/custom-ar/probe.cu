#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "custom_all_reduce.cuh"

torch::Tensor alias_buffer(int64_t ptr, int64_t count, int64_t device, int64_t dtype) {
  auto scalar = dtype == 0 ? torch::kBFloat16 : (dtype == 1 ? torch::kFloat16 : torch::kFloat32);
  return torch::from_blob(reinterpret_cast<void*>(ptr), {count}, [](void*) {},
                         torch::TensorOptions().dtype(scalar).device(torch::kCUDA, device));
}

template <typename T>
void launch_impl(torch::Tensor peers, const std::vector<int64_t>& signals,
                 torch::Tensor out, int rank, int threads, int block_limit, int algo) {
  TORCH_CHECK(signals.size() == 4);
  TORCH_CHECK(rank >= 0 && rank < 4);
  TORCH_CHECK(threads >= 32 && threads <= 512 && threads % 32 == 0);
  TORCH_CHECK(block_limit >= 1 && block_limit <= vllm::kMaxBlocks);
  constexpr int pack = vllm::packed_t<T>::P::size;
  TORCH_CHECK(out.numel() % pack == 0);
  int size = out.numel() / pack;
  int blocks = std::min(block_limit, (size + threads - 1) / threads);
  vllm::RankSignals sg{};
  for (int i = 0; i < 4; ++i) sg.signals[i] = reinterpret_cast<vllm::Signal*>(signals[i]);
  auto stream = c10::cuda::getCurrentCUDAStream(out.get_device()).stream();
  auto data = reinterpret_cast<vllm::RankData*>(peers.data_ptr());
  auto result = reinterpret_cast<T*>(out.data_ptr());
  if (algo == 1) {
    vllm::cross_device_reduce_1stage<T,4><<<blocks,threads,0,stream>>>(data,sg,sg.signals[rank],result,rank,size);
  } else {
    vllm::cross_device_reduce_2stage<T,4><<<blocks,threads,0,stream>>>(data,sg,sg.signals[rank],result,rank,size);
  }
  auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

void launch(torch::Tensor peers, const std::vector<int64_t>& signals, torch::Tensor out,
            int rank, int threads, int block_limit, int algo) {
  c10::cuda::CUDAGuard guard(out.device());
  TORCH_CHECK(peers.is_cuda() && peers.scalar_type() == torch::kInt64 && peers.numel() >= 16);
  TORCH_CHECK(out.is_contiguous() && out.numel() > 0);
  if (out.scalar_type() == torch::kBFloat16) launch_impl<nv_bfloat16>(peers,signals,out,rank,threads,block_limit,algo);
  else if (out.scalar_type() == torch::kFloat16) launch_impl<half>(peers,signals,out,rank,threads,block_limit,algo);
  else if (out.scalar_type() == torch::kFloat32) launch_impl<float>(peers,signals,out,rank,threads,block_limit,algo);
  else TORCH_CHECK(false,"unsupported dtype");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("alias_buffer", &alias_buffer);
  m.def("launch", &launch);
  m.def("meta_size", [](){return sizeof(vllm::Signal);});
}
