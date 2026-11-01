from tts_bench.cli import main
from tts_bench.config import load_config


def test_pinned_config(repo):
    cfg = load_config(repo / "configs" / "p1_run.toml")
    assert cfg.model.id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert cfg.model.voice == "Ryan" and cfg.model.language == "English"
    assert cfg.model.precision == "bfloat16"
    assert cfg.run.concurrency == [1, 4, 8]
    assert cfg.sampling.temperature == 0.9 and cfg.sampling.top_k == 50 and cfg.sampling.max_new_tokens == 2048
    assert -1 in cfg.streaming.initial_codec_chunk_frames
    assert cfg.resolve(cfg.run.prompts).exists()
    assert set(cfg.engines) == {"official", "vllm_omni", "sglang_omni", "fake"}


def test_cli_validate_prompts(capsys):
    assert main(["validate-prompts"]) == 0
    assert "OK:" in capsys.readouterr().out


def test_cli_bench_fake(repo, tmp_path, capsys):
    rc = main(["bench", "--config", str(repo / "configs" / "fake_cpu.toml"), "--engine", "fake",
               "--concurrency", "1", "--modes", "full", "--limit", "2", "--out", str(tmp_path)])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1 and (runs[0] / "results.jsonl").exists()
