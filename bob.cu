#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/Utils.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdio>
#include <iostream>

using namespace at;

int integer_round(int num, int denom){
  return (num + denom - 1) / denom;
}

template<class T, class U>
__global__ void index_add_deterministic_dim0_cuda_kernel(
  T *const self,
  const U *const index,
  const T *const tensor,
  const int indexN,
  const int stride
){
  const auto tid = blockDim.x*blockIdx.x+threadIdx.x;
  // Which element of `index` are we assigned?
  const auto my_index = tid / stride;
  // Calculate which element we own within a dim0 slice
  const auto my_element = tid % stride;

  //Early exit for smaller datasets
  if(my_index >= indexN){
    return;
  }

  // printf(
  //   "tid=%02d, element=%d, index=%d, self_out=%d\n",
  //   tid, my_element, my_index,
  //   my_index * stride + my_element
  // );

  T running_sum = 0;
  for(int i=0;i<indexN;i++){
    if(index[i] == my_index){
      running_sum += tensor[i * stride + my_element];
    }
  }
  self[my_index * stride + my_element] += running_sum;
}

Tensor index_add_deterministic_dim0_cuda(
  Tensor &self,
  const Tensor &index,
  const Tensor &tensor
) {
  // Stride is product of all non-zero dimensions. It also represents the size
  // of a single slice through dim0.
  const auto stride = at::prod_intlist(self.sizes().begin() + 1, self.sizes().end());
  // The best-case scenario is that each element of index is unique. In this
  // case each index has stride elements associated with it, so we maximize
  // performance by using one thread per element to perform copies. If index
  // has nonunique values the excess threads have an early exit.
  const auto max_useful_threads = stride * self.size(0);
  const auto threads_per_block = 128; // Not chosen for any special reason.
  const auto num_blocks = integer_round(max_useful_threads, threads_per_block);
  AT_DISPATCH_ALL_TYPES(
    self.scalar_type(), "index_add_deterministic_dim0_cuda", [&](){
      AT_DISPATCH_INDEX_TYPES(
        index.scalar_type(), "index_add_deterministic_dim0_cuda", [&](){
          std::cerr<<"Choosing stride="<<stride
                   <<" max_useful_threads="<<max_useful_threads
                   <<" threads_per_block="<<threads_per_block
                   <<" num_blocks="<<num_blocks
                   <<std::endl;
          index_add_deterministic_dim0_cuda_kernel<<<threads_per_block, num_blocks, 0>>>(
            self.data_ptr<scalar_t>(),
            index.data_ptr<index_t>(),
            tensor.data_ptr<scalar_t>(),
            index.numel(),
            stride
          );
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      );
    }
  );

  return self;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("index_add_deterministic(Tensor self, Tensor index, Tensor tensor) -> Tensor");
  m.impl("index_add_deterministic", c10::DispatchKey::CUDA, TORCH_FN(index_add_deterministic_dim0_cuda));
}
