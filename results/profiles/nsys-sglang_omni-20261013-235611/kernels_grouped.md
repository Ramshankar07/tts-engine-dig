# sglang_omni - nsys, 8 prompts, concurrency 1

Total GPU kernel time: 7992.33 ms over 1534746 launches.

| group | time ms | % | launches |
|---|---|---|---|
| attention | 1038.52 | 13.0 | 342586 |
| gemm | 3838.08 | 48.0 | 528830 |
| codec-conv | 118.52 | 1.5 | 12780 |
| other | 2997.20 | 37.5 | 650550 |

## Top 25 kernels

| # | group | time ms | % | launches | avg us | kernel |
|---|---|---|---|---|---|---|
| 1 | gemm | 1718.404 | 21.5 | 174040 | 9.9 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` |
| 2 | gemm | 1478.162 | 18.5 | 251400 | 5.9 | `gemv_add_kernel` |
| 3 | other | 1182.713 | 14.8 | 125700 | 9.4 | `norm_qkv_rope_store_kernel` |
| 4 | other | 1026.481 | 12.8 | 125700 | 8.2 | `norm_gate_up_silu_kernel` |
| 5 | gemm | 516.456 | 6.5 | 46816 | 11.0 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` |
| 6 | attention | 358.439 | 4.5 | 117320 | 3.1 | `cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_2_64x32x128_4x1x1_cga1x1x1_kernel0_0` |
| 7 | attention | 323.848 | 4.1 | 46872 | 6.9 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<(int)2, cute::tuple<cute...` |
| 8 | attention | 189.902 | 2.4 | 93856 | 2.0 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64204820481_tenso...` |
| 9 | other | 95.372 | 1.2 | 25140 | 3.8 | `seeded_top_k_merge_sample_kernel` |
| 10 | other | 89.371 | 1.1 | 46928 | 1.9 | `_triton_mrope_forward_fused` |
| 11 | gemm | 79.309 | 1.0 | 46928 | 1.7 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv...` |
| 12 | other | 65.305 | 0.8 | 46928 | 1.4 | `void sglang::fused_qknorm_warp<(long)128, (bool)1, __nv_bfloat16>(sglang::QKNormParams)` |
| 13 | other | 64.467 | 0.8 | 25140 | 2.6 | `seeded_top_k_chunk_kernel` |
| 14 | other | 62.906 | 0.8 | 46928 | 1.3 | `void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::ActivationKind)0, (bool)1, (bool)0, (bool)0, (bool)0>(sglang::ActivationParams)` |
| 15 | other | 62.167 | 0.8 | 46928 | 1.3 | `void sglang::store_kvcache_kernel<(long)2048, (long)2048, (unsigned int)128, (bool)1, long>(sglang::StoreKVCacheParams)` |
| 16 | attention | 57.810 | 0.7 | 46928 | 1.2 | `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<(int)8>, cute::C<(int)128>>, (int)5, (int)256, (int)1, (bool)0...` |
| 17 | attention | 52.692 | 0.7 | 25140 | 2.1 | `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gmemalign128oi64102410241_tensorptrbf16gmemalign1...` |
| 18 | other | 45.308 | 0.6 | 25140 | 1.8 | `gather_codec_embedding_and_add_kernel` |
| 19 | attention | 43.556 | 0.5 | 8380 | 5.2 | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1x1_kernel0_0` |
| 20 | other | 29.812 | 0.4 | 1676 | 17.8 | `void at::native::radixSortKVInPlace<(int)-2, (int)-1, (int)128, (int)32, c10::BFloat16, long, unsigned int>(at::cuda::detail::TensorInfo<...` |
| 21 | codec-conv | 23.679 | 0.3 | 27 | 877.0 | `void implicit_convolve_sgemm<__nv_bfloat16, __nv_bfloat16, (int)128, (int)6, (int)7, (int)3, (int)3, (int)5, (int)1, (bool)0, (bool)0, (b...` |
| 22 | other | 22.272 | 0.3 | 8785 | 2.5 | `void at::native::<unnamed>::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, (int)2, (int)2, (int)-2>(at::cuda::detail::TensorInf...` |
| 23 | other | 19.607 | 0.2 | 4141 | 4.7 | `void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::Ten...` |
| 24 | other | 17.274 | 0.2 | 4921 | 3.5 | `void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::index_copy_kernel_impl<at::native::OpaqueType<(int)2>>(at::T...` |
| 25 | codec-conv | 14.674 | 0.2 | 3857 | 3.8 | `snake_beta_channels_last_kernel` |
