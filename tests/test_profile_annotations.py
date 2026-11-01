from tts_bench.profile import is_annotation, stats_from_kernels_csv, write_kernel_outputs, KernelStat


def test_annotation_rows_are_not_kernels(tmp_path):
    stats = [
        KernelStat("lm_generate", 1_000_000.0, 2),
        KernelStat("request:p1-001", 500_000.0, 1),
        KernelStat("nvjet_tst_64x8_64x16_4x1_v_bz_TNT", 200.0, 10),
    ]
    write_kernel_outputs(stats, tmp_path, "t")
    kept = stats_from_kernels_csv(tmp_path / "kernels.csv")
    assert [k.name for k in kept] == ["nvjet_tst_64x8_64x16_4x1_v_bz_TNT"]
    assert is_annotation("codec_decode") and not is_annotation("flash_fwd_kernel")
