"""Configuration: precedence, coercion, interpolation, path anchoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doccat.cli import build_parser, build_llm_config, build_pipeline_options, main
from doccat.config import CONFIG_TEMPLATE, ConfigError, Settings, discover_config_file


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "doccat.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_defaults_when_there_is_no_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCCAT_CONFIG", raising=False)
    settings = Settings.load()
    assert settings.config_file is None
    assert settings.indexing.llm_workers == 8
    assert settings.llm.base_url == "http://localhost:8080/v1"


def test_config_file_is_read_and_typed(tmp_path):
    path = write_config(tmp_path, """
llm:
  base_url: http://gpu-box:8000/v1
  model: my-model
  max_tokens: "8192"
  think: "yes"
  requests_per_second: 12
indexing:
  llm_workers: 24
  include: ["contracts/**"]
server:
  port: "9000"
""")
    s = Settings.load(path, use_env=False)
    assert s.llm.base_url == "http://gpu-box:8000/v1"
    assert s.llm.max_tokens == 8192 and isinstance(s.llm.max_tokens, int)
    assert s.llm.think is True
    assert s.llm.requests_per_second == 12.0
    assert s.indexing.llm_workers == 24
    assert s.indexing.include == ["contracts/**"]
    assert s.server.port == 9000


def test_environment_overrides_the_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, "llm:\n  model: from-file\nindexing:\n  llm_workers: 4\n")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    s = Settings.load(path)
    assert s.llm.model == "from-env"
    assert s.indexing.llm_workers == 4          # untouched by the env
    assert "llm.model <- $LLM_MODEL" in s.env_overrides


def test_flags_override_the_environment_and_the_file(tmp_path, monkeypatch):
    path = write_config(tmp_path, "llm:\n  model: from-file\nindexing:\n  llm_workers: 4\n")
    monkeypatch.setenv("DOCCAT_LLM_WORKERS", "16")
    settings = Settings.load(path)
    parser = build_parser(settings)

    args = parser.parse_args(["index", "--root", str(tmp_path)])
    assert args.model == "from-file" and args.llm_workers == 16     # env beat the file

    args = parser.parse_args(["index", "--root", str(tmp_path), "-j", "64",
                              "--model", "from-flag"])
    assert args.llm_workers == 64 and args.model == "from-flag"     # flags beat both


def test_env_interpolation_with_and_without_a_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-value")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    path = write_config(tmp_path, """
llm:
  api_key: ${MY_KEY}
  model: ${NOT_SET:-fallback-model}
""")
    s = Settings.load(path)
    assert s.llm.api_key == "secret-value"
    assert s.llm.model == "fallback-model"


def test_relative_paths_anchor_to_the_config_file_not_the_cwd(tmp_path, monkeypatch):
    nested = tmp_path / "deploy"
    nested.mkdir()
    path = nested / "doccat.yaml"
    path.write_text("paths:\n  docs_root: ./documents\n  catalog: ./cat.jsonl\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)          # run from somewhere else entirely
    s = Settings.load(path)
    assert s.paths.docs_root == nested / "documents"
    assert s.paths.catalog == nested / "cat.jsonl"


def test_unknown_section_and_key_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown section"):
        Settings.load(write_config(tmp_path, "nonsense:\n  a: 1\n"), use_env=False)
    with pytest.raises(ConfigError, match="unknown key"):
        Settings.load(write_config(tmp_path, "llm:\n  modle: typo\n"), use_env=False)


def test_bad_type_is_rejected_with_a_useful_message(tmp_path):
    with pytest.raises(ConfigError, match="indexing.llm_workers: expected an integer"):
        Settings.load(write_config(tmp_path, "indexing:\n  llm_workers: many\n"), use_env=False)


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        Settings.load(tmp_path / "nope.yaml")


def test_discovery_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCCAT_CONFIG", raising=False)
    assert discover_config_file() is None
    (tmp_path / "doccat.yaml").write_text("llm: {}\n", encoding="utf-8")
    assert discover_config_file() == Path("doccat.yaml")

    explicit = tmp_path / "other.yaml"
    explicit.write_text("llm: {}\n", encoding="utf-8")
    monkeypatch.setenv("DOCCAT_CONFIG", str(explicit))
    assert discover_config_file() == explicit
    assert discover_config_file(tmp_path / "doccat.yaml") == tmp_path / "doccat.yaml"


def test_shipped_template_loads_and_is_complete(tmp_path):
    path = write_config(tmp_path, CONFIG_TEMPLATE)
    s = Settings.load(path, use_env=False)
    assert s.llm.model == "qwen2.5-14b-instruct"
    assert s.indexing.mode == "auto"
    assert s.optimize.sample_size == 12


def test_secrets_are_masked_unless_asked_for(tmp_path):
    path = write_config(tmp_path, "llm:\n  api_key: sk-abcdefghijklmnop\n")
    s = Settings.load(path, use_env=False)
    assert "abcdefghijklmnop" not in json.dumps(s.as_dict())
    assert s.as_dict(mask_secrets=False)["llm"]["api_key"] == "sk-abcdefghijklmnop"


def test_headers_from_the_config_reach_the_llm_client(tmp_path):
    path = write_config(tmp_path, """
llm:
  bill_to: BSC-LT
  headers:
    X-Team: applied-intelligence
""")
    settings = Settings.load(path, use_env=False)
    args = build_parser(settings).parse_args(["index", "--root", str(tmp_path),
                                              "--header", "X-Run:nightly"])
    cfg = build_llm_config(args)
    assert cfg.extra_headers["X-Team"] == "applied-intelligence"
    assert cfg.extra_headers["X-HF-Bill-To"] == "BSC-LT"
    assert cfg.extra_headers["X-Run"] == "nightly"


def test_paths_from_the_config_drive_a_run(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = write_config(tmp_path, f"""
paths:
  docs_root: {docs}
  catalog: {tmp_path}/out/catalog.jsonl
  profiles_dir: [{tmp_path}/profiles]
indexing:
  llm_workers: 12
  checkpoint_every: 7
""")
    settings = Settings.load(path, use_env=False)
    args = build_parser(settings).parse_args(["index"])     # no flags at all
    assert args.root == docs
    opts = build_pipeline_options(args)
    assert opts.llm_workers == 12
    assert opts.checkpoint_every == 7
    assert opts.text_cache == tmp_path / "out" / "extracted_text"
    assert args.profiles_dir == [f"{tmp_path}/profiles"]


def test_config_commands(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCCAT_CONFIG", raising=False)

    assert main(["config", "init"]) == 0
    assert (tmp_path / "doccat.yaml").is_file()

    assert main(["config", "path"]) == 0
    assert "doccat.yaml" in capsys.readouterr().out

    assert main(["config", "show", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["indexing"]["llm_workers"] == 8
    assert report["_source"]["config_file"].endswith("doccat.yaml")

    with pytest.raises(SystemExit):
        main(["config", "init"])                # refuses to clobber


def test_config_init_then_index_needs_no_flags(tmp_path, docs_tree, mock_llm, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCCAT_CONFIG", raising=False)
    write_config(tmp_path, f"""
paths:
  docs_root: {docs_tree}
  catalog: {tmp_path}/catalog.jsonl
llm:
  base_url: {mock_llm.base_url}
  model: mock
indexing:
  llm_workers: 4
""")
    assert main(["index"]) == 0                 # every setting came from the file
    records = (tmp_path / "catalog.jsonl").read_text().strip().splitlines()
    assert len(records) == 5
