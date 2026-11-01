# vllm_omni - nsys, 8 prompts, concurrency 1

Total GPU kernel time: 10800.82 ms over 2779148 launches.

| group | time ms | % | launches |
|---|---|---|---|
| attention | 2300.47 | 21.3 | 644488 |
| gemm | 5663.16 | 52.4 | 923150 |
| codec-conv | 350.57 | 3.2 | 7638 |
| other | 2486.62 | 23.0 | 1203872 |

## Top 25 kernels

| # | group | time ms | % | launches | avg us | kernel |
|---|---|---|---|---|---|---|
| 1 | gemm | 1692.714 | 15.7 | 166760 | 10.2 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` |
| 2 | gemm | 1586.278 | 14.7 | 251976 | 6.3 | `nvjet_sm90_tst_64x24_64x16_4x1_v_bz_TNT` |
| 3 | attention | 984.511 | 9.1 | 125400 | 7.9 | `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, ...` |
| 4 | gemm | 700.065 | 6.5 | 127176 | 5.5 | `nvjet_sm90_tst_64x8_64x16_2x4_h_bz_TNT` |
| 5 | gemm | 679.026 | 6.3 | 125400 | 5.4 | `nvjet_sm90_tst_64x16_64x16_4x2_h_bz_splitK_TNT` |
| 6 | gemm | 511.208 | 4.7 | 46200 | 11.1 | `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` |
| 7 | attention | 474.700 | 4.4 | 46648 | 10.2 | `void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMainloopFwdSm90<(int)2, cute::tuple<cute...` |
| 8 | other | 306.455 | 2.8 | 25080 | 12.2 | `void at::native::sbtopk::gatherTopK<c10::BFloat16, unsigned int, (int)1, (bool)0>(at::cuda::detail::TensorInfo<const T1, T2>, T2, T2, boo...` |
| 9 | gemm | 300.257 | 2.8 | 172216 | 1.7 | `void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv...` |
| 10 | attention | 263.249 | 2.4 | 125400 | 2.1 | `triton_per_fused__scaled_dot_product_flash_attention__to_copy__unsafe_view_add_cat_clone_index_mean_mm_mul_neg_pow_rsqrt_slice_split_tran...` |
| 11 | attention | 223.442 | 2.1 | 125400 | 1.8 | `triton_per_fused__scaled_dot_product_flash_attention__to_copy__unsafe_view_add_cat_clone_index_mean_mm_mul_neg_pow_rsqrt_slice_split_tran...` |
| 12 | codec-conv | 207.297 | 1.9 | 252 | 822.6 | `void implicit_convolve_sgemm<__nv_bfloat16, __nv_bfloat16, (int)128, (int)6, (int)7, (int)3, (int)3, (int)5, (int)1, (bool)0, (bool)0, (b...` |
| 13 | other | 160.965 | 1.5 | 125400 | 1.3 | `triton_poi_fused__unsafe_view_mul_silu_split_with_sizes_6` |
| 14 | gemm | 152.174 | 1.4 | 26840 | 5.7 | `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` |
| 15 | attention | 150.373 | 1.4 | 125400 | 1.2 | `triton_poi_fused__scaled_dot_product_flash_attention__to_copy__unsafe_view_add_clone_index_mean_mm_mul_pow_rsqrt_slice_split_transpose_un...` |
| 16 | other | 149.483 | 1.4 | 25080 | 6.0 | `void at::native::warpMergeSortKVInPlace<(int)-2, (int)-1, (int)128, (int)16, c10::BFloat16, long, at::native::GTOp<c10::BFloat16, (bool)1...` |
| 17 | other | 119.515 | 1.1 | 26752 | 4.5 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, (int)4, (i...` |
| 18 | other | 117.142 | 1.1 | 125400 | 0.9 | `triton_poi_fused__to_copy_add_mean_mm_mul_pow_rsqrt_view_1` |
| 19 | attention | 105.863 | 1.0 | 46816 | 2.3 | `void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0>(const T1 *, const T1 *, T2 *, T2 *, ...` |
| 20 | other | 104.219 | 1.0 | 75240 | 1.4 | `triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_5` |
| 21 | other | 99.375 | 0.9 | 46816 | 2.1 | `triton_red_fused_fused_add_rms_norm_2` |
| 22 | other | 89.197 | 0.8 | 46816 | 1.9 | `triton_red_fused_fused_add_rms_norm_0` |
| 23 | other | 88.702 | 0.8 | 45144 | 2.0 | `triton_per_fused_4` |
| 24 | attention | 87.739 | 0.8 | 46816 | 1.9 | `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cute::tuple<cute::C<(int)8>, cute::C<(int)128>>, (int)5, (int)256, (int)1, (bool)0...` |
| 25 | other | 87.399 | 0.8 | 45144 | 1.9 | `triton_poi_fused_cat_index_select_split_split_with_sizes_3` |
