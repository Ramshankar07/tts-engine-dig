# Official Qwen3-TTS (PyTorch) - 2 prompts, batch 1 (torch.profiler; record_function ranges excluded)

Total GPU kernel time: 1017.07 ms over 473174 launches.

| group | time ms | % | launches |
|---|---|---|---|
| attention | 31.29 | 3.1 | 6046 |
| gemm | 265.37 | 26.1 | 51144 |
| codec-conv | 5.57 | 0.5 | 151 |
| other | 714.84 | 70.3 | 415833 |

## Top 25 kernels

| # | group | time ms | % | launches | avg us | kernel |
|---|---|---|---|---|---|---|
| 1 | gemm | 203.211 | 20.0 | 36884 | 5.5 | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` |
| 2 | other | 81.159 | 8.0 | 37182 | 2.2 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c1...` |
| 3 | other | 68.967 | 6.8 | 26890 | 2.6 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() con...` |
| 4 | other | 57.574 | 5.7 | 25084 | 2.3 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, ...` |
| 5 | other | 38.470 | 3.8 | 25084 | 1.5 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native:...` |
| 6 | other | 32.030 | 3.1 | 25084 | 1.3 | `void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() cons...` |
| 7 | other | 30.192 | 3.0 | 26948 | 1.1 | `void at::native::vectorized_elementwise_kernel<8, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std:...` |
| 8 | other | 29.538 | 2.9 | 25088 | 1.2 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, std::array<char*, 2ul> >(int, at::native::CUD...` |
| 9 | attention | 28.470 | 2.8 | 5684 | 5.0 | `void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64...` |
| 10 | other | 27.856 | 2.7 | 25084 | 1.1 | `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::Tens...` |
| 11 | other | 27.501 | 2.7 | 23638 | 1.2 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::C...` |
| 12 | other | 24.909 | 2.4 | 12784 | 1.9 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 64, 64>(a...` |
| 13 | other | 24.260 | 2.4 | 12092 | 2.0 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lam...` |
| 14 | other | 23.762 | 2.3 | 18284 | 1.3 | `void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::bina...` |
| 15 | gemm | 22.745 | 2.2 | 4350 | 5.2 | `nvjet_tst_64x8_64x16_4x1_v_bz_splitK_TNT` |
| 16 | other | 21.792 | 2.1 | 11368 | 1.9 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned i...` |
| 17 | other | 19.383 | 1.9 | 930 | 20.8 | `void at::native::sbtopk::gatherTopK<float, unsigned int, 1, false>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int,...` |
| 18 | gemm | 18.004 | 1.8 | 1624 | 11.1 | `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_TNT` |
| 19 | other | 12.302 | 1.2 | 870 | 14.1 | `void (anonymous namespace)::softmax_warp_forward<float, float, float, 11, false, false>(float*, float const*, int, int, int, bool const*,...` |
| 20 | other | 11.547 | 1.1 | 8084 | 1.4 | `Memcpy DtoH (Device -> Pinned)` |
| 21 | other | 10.788 | 1.1 | 8470 | 1.3 | `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul> >(int, at::native::CUDAFuncto...` |
| 22 | other | 10.746 | 1.1 | 6900 | 1.6 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase...` |
| 23 | other | 10.245 | 1.0 | 6054 | 1.7 | `void at::native::vectorized_elementwise_kernel<8, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::...` |
| 24 | other | 10.115 | 1.0 | 6960 | 1.5 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() con...` |
| 25 | gemm | 10.020 | 1.0 | 6030 | 1.7 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, t...` |
