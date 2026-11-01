# official - nsys, 2 prompts, concurrency 1

Total GPU kernel time: 1990.93 ms over 913910 launches.

| group | time ms | % | launches |
|---|---|---|---|
| attention | 62.72 | 3.2 | 12092 |
| gemm | 530.91 | 26.7 | 102288 |
| codec-conv | 11.14 | 0.6 | 302 |
| other | 1386.15 | 69.6 | 799228 |

## Top 25 kernels

| # | group | time ms | % | launches | avg us | kernel |
|---|---|---|---|---|---|---|
| 1 | gemm | 406.272 | 20.4 | 73768 | 5.5 | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` |
| 2 | other | 161.463 | 8.1 | 74364 | 2.2 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c1...` |
| 3 | other | 138.510 | 7.0 | 53780 | 2.6 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` |
| 4 | other | 115.229 | 5.8 | 50168 | 2.3 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned in...` |
| 5 | other | 76.114 | 3.8 | 50168 | 1.5 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, flo...` |
| 6 | other | 63.815 | 3.2 | 50168 | 1.3 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::...` |
| 7 | other | 61.287 | 3.1 | 54360 | 1.1 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda(float) (i...` |
| 8 | other | 59.013 | 3.0 | 50176 | 1.2 | `void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char *, (unsigned long)2>>(in...` |
| 9 | attention | 56.697 | 2.8 | 11368 | 5.0 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_ke...` |
| 10 | other | 55.711 | 2.8 | 50168 | 1.1 | `void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::Tensor...` |
| 11 | other | 55.097 | 2.8 | 47276 | 1.2 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char *, (unsigned long)3>>(...` |
| 12 | other | 49.741 | 2.5 | 25568 | 1.9 | `void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)2>, unsigned int, (int)4, (int)64, (int)6...` |
| 13 | other | 48.600 | 2.4 | 24184 | 2.0 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::neg_kernel_cuda(at::TensorItera...` |
| 14 | other | 47.483 | 2.4 | 36568 | 1.3 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native:...` |
| 15 | gemm | 45.651 | 2.3 | 8700 | 5.2 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 16 | other | 43.615 | 2.2 | 22736 | 1.9 | `void at::native::<unnamed>::CatArrayBatchedCopy_alignedK_contig<at::native::<unnamed>::OpaqueType<(unsigned int)2>, unsigned int, (int)4,...` |
| 17 | other | 38.854 | 2.0 | 1860 | 20.9 | `void at::native::sbtopk::gatherTopK<float, unsigned int, (int)1, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, bool, T2, T...` |
| 18 | gemm | 35.970 | 1.8 | 3248 | 11.1 | `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT` |
| 19 | other | 24.642 | 1.2 | 1740 | 14.2 | `void <unnamed>::softmax_warp_forward<float, float, float, (int)11, (bool)0, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, ...` |
| 20 | other | 21.601 | 1.1 | 16940 | 1.3 | `void at::native::vectorized_elementwise_kernel<(int)2, at::native::CUDAFunctor_add<long>, std::array<char *, (unsigned long)3>>(int, T2, T3)` |
| 21 | other | 21.345 | 1.1 | 13800 | 1.5 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` |
| 22 | other | 20.314 | 1.0 | 13920 | 1.5 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::oper...` |
| 23 | gemm | 20.218 | 1.0 | 12060 | 1.7 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv...` |
| 24 | other | 19.835 | 1.0 | 12108 | 1.6 | `void at::native::vectorized_elementwise_kernel<(int)8, at::native::<unnamed>::silu_kernel(at::TensorIteratorBase &)::[lambda() (instance ...` |
| 25 | other | 18.382 | 0.9 | 8700 | 2.1 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::MinNanFunctor<lo...` |
