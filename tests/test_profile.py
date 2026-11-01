import pytest

from tts_bench.profile import KernelStat, classify_kernel, group_totals, kernels_from_nsys_csv, render_kernel_markdown


@pytest.mark.parametrize(
    "name,group",
    [
        ("void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4>>", "attention"),
        ("fmha_cutlassF_bf16_aligned_64x64_rf_sm80", "attention"),
        ("flashinfer::BatchDecodeWithPagedKVCacheKernel", "attention"),
        ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", "gemm"),
        ("ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn", "gemm"),
        ("nvjet_tst_128x64_64x8_2x1_v_bz_coopA_TNN", "gemm"),
        ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>", "gemm"),
        ("sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc", "codec-conv"),
        ("void cudnn::ops::nchwToNhwcKernel<__nv_bfloat16>", "codec-conv"),
        ("void at::native::conv_depthwise2d_forward_kernel<float>", "codec-conv"),
        ("void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda>", "other"),
        ("void at::native::(anonymous namespace)::convert_kernel<float>", "other"),  # "convert" is not conv
        ("void at::native::reduce_kernel<512, 1>", "other"),
    ],
)
def test_classify(name, group):
    assert classify_kernel(name) == group


NSYS_CSV = """Generating SQLite file report.sqlite
Processing [report.sqlite] with [cuda_gpu_kern_sum.py]...
"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
"55.0","5500000","100","55000.0","55000.0","50000","60000","1000.0","void flash_fwd_kernel<X>"
"30.0","3000000","300","10000.0","10000.0","9000","11000","100.0","nvjet_tst_128x64_TNN"
"10.0","1000000","50","20000.0","20000.0","19000","21000","100.0","sm80_xmma_fprop_implicit_gemm_bf16"
"5.0","500000","1000","500.0","500.0","400","600","10.0","void at::native::reduce_kernel<512, 1>"
"""


def test_nsys_parse_and_group():
    stats = kernels_from_nsys_csv(NSYS_CSV)
    assert len(stats) == 4
    assert stats[0].total_us == pytest.approx(5500.0) and stats[0].count == 100
    g = group_totals(stats)
    assert g["attention"]["pct"] == pytest.approx(55.0)
    assert g["gemm"]["pct"] == pytest.approx(30.0)
    assert g["codec-conv"]["pct"] == pytest.approx(10.0)
    assert g["other"]["pct"] == pytest.approx(5.0)
    md = render_kernel_markdown(stats, "test")
    assert "| attention | 5.50 | 55.0 | 100 |" in md
    assert "flash_fwd_kernel" in md


def test_nsys_parse_rejects_garbage():
    with pytest.raises(ValueError):
        kernels_from_nsys_csv("nothing here")


def test_pipe_in_kernel_name_is_escaped():
    md = render_kernel_markdown([KernelStat("a|b", 1.0, 1)], "t")
    assert "a\\|b" in md
