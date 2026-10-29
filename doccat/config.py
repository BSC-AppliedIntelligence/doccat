"""One place to configure everything: endpoints, directories, workers, serving.

Settings reach the program from four places. Later ones win:

    1. the defaults in this module
    2. a config file — `--config FILE`, `$DOCCAT_CONFIG`, or the first of
       ./doccat.yaml, ./doccat.yml, ./.doccat.yaml,
       ~/.config/doccat/config.yaml, /etc/doccat/config.yaml
    3. environment variables (the names the container image and systemd units use)
    4. explicit command-line flags

So a deployment can put everything in one YAML file, override one value with an
environment variable in a container, and still pass a flag for a one-off run —
without any of those three mechanisms knowing about the others.

    from doccat.config import Settings

    settings = Settings.load()                  # discover the config file
    settings = Settings.load("prod.yaml")       # or name one
    print(settings.llm.model, settings.indexing.llm_workers)

`doccat config init` writes a commented starter file; `doccat config show`
prints the fully resolved settings, secrets masked, with the origin of each
section — which is the fastest way to find out why a deployment is not using
the endpoint you thought it was.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("doccat.config")

__all__ = ["Settings", "PathSettings", "LLMSettings", "IndexSettings",
           "ServerSettings", "OptimizeSettings", "GraphBuildSettings", "ConfigError",
           "CONFIG_TEMPLATE", "discover_config_file", "ENV_MAP"]


class ConfigError(Exception):
    """The config file is missing, malformed, or contains unknown keys."""


#: Where `Settings.load()` looks when no path is given, in order.
CONFIG_SEARCH_PATH = (
    Path("doccat.yaml"),
    Path("doccat.yml"),
    Path(".doccat.yaml"),
    Path.home() / ".config" / "doccat" / "config.yaml",
    Path("/etc/doccat/config.yaml"),
)

#: `section.field` → environment variable. The single source of truth for env
#: names: the CLI, the container image and the systemd units all read these.
ENV_MAP: dict[str, str] = {
    "paths.docs_root": "DOCS_ROOT",
    "paths.catalog": "CATALOG_PATH",
    "paths.text_cache": "TEXT_CACHE",
    "paths.error_log": "DOCCAT_ERROR_LOG",
    "paths.profiles_dir": "DOCCAT_PROFILES_DIR",

    "llm.base_url": "LLM_BASE_URL",
    "llm.model": "LLM_MODEL",
    "llm.api_key": "LLM_API_KEY",
    "llm.temperature": "LLM_TEMPERATURE",
    "llm.max_tokens": "LLM_MAX_TOKENS",
    "llm.timeout": "LLM_TIMEOUT",
    "llm.think": "LLM_THINK",
    "llm.requests_per_second": "LLM_RPS",
    "llm.bill_to": "HF_BILL_TO",

    "indexing.llm_workers": "DOCCAT_LLM_WORKERS",
    "indexing.extract_workers": "DOCCAT_EXTRACT_WORKERS",
    "indexing.mode": "DOCCAT_MODE",
    "indexing.profile": "DOCCAT_PROFILE",
    "indexing.checkpoint_every": "DOCCAT_CHECKPOINT_EVERY",
    "indexing.progress_every": "DOCCAT_PROGRESS_EVERY",
    "indexing.change_detection": "DOCCAT_CHANGE_DETECTION",
    "indexing.shard": "DOCCAT_SHARD",

    "server.host": "MCP_HOST",
    "server.port": "MCP_PORT",
    "server.stateless": "MCP_STATELESS",

    "optimize.sample_size": "DOCCAT_SAMPLE_SIZE",
    "optimize.refine_rounds": "DOCCAT_REFINE_ROUNDS",

    "graph.out_dir": "DOCCAT_GRAPH_DIR",
    "graph.detect_communities": "DOCCAT_GRAPH_COMMUNITIES",
    "graph.min_community_size": "DOCCAT_GRAPH_MIN_COMMUNITY_SIZE",
    "graph.summarize": "DOCCAT_GRAPH_SUMMARIZE",
}

#: Fields whose value is never printed in full.
_SECRET_FIELDS = {"llm.api_key"}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class PathSettings:
    """Where the documents, the catalog and its side files live."""

    docs_root: Path = Path(".")
    catalog: Path = Path("catalog.jsonl")
    text_cache: Path | None = None      # default: <catalog dir>/extracted_text
    error_log: Path | None = None       # default: <catalog dir>/llm_errors
    profiles_dir: list[Path] = field(default_factory=list)

    def resolved_text_cache(self) -> Path:
        return self.text_cache or self.catalog.resolve().parent / "extracted_text"

    def resolved_error_log(self) -> Path:
        return self.error_log or self.catalog.resolve().parent / "llm_errors"


@dataclass
class LLMSettings:
    """The OpenAI-compatible endpoint used for cataloguing."""

    base_url: str = "http://localhost:8080/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 300.0
    think: bool = False
    requests_per_second: float = 0.0
    bill_to: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class IndexSettings:
    """How a cataloguing run behaves."""

    llm_workers: int = 8
    extract_workers: int = 0            # 0 = auto (min(cpu, 8))
    mode: str = "auto"                  # parse | upload | auto
    profile: str | None = None          # force one profile
    checkpoint_every: int = 50
    progress_every: int = 20
    change_detection: str = "hash"      # hash | mtime
    shard: str | None = None            # "i/n"
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class ServerSettings:
    """How the MCP server is exposed."""

    host: str = "0.0.0.0"
    port: int = 8000
    stateless: bool = True


@dataclass
class OptimizeSettings:
    """Defaults for the profile-designing agent (`doccat optimize`)."""

    sample_size: int = 12
    refine_rounds: int = 1
    sample_chars: int = 6000


@dataclass
class GraphBuildSettings:
    """Defaults for corpus-level graph consolidation (`doccat graph build`).

    Distinct from a profile's own `graph:` YAML block (doccat/profiles.py),
    which controls what gets *extracted* per document — this section controls
    how the per-document results get *consolidated* into one corpus graph.
    See docs/GRAPH.md.
    """

    out_dir: Path | None = None        # default: <catalog dir>/graph
    detect_communities: bool = True
    min_community_size: int = 2
    summarize: bool = False             # one LLM call per community; off by default
    community_seed: int = 0

    def resolved_out_dir(self, catalog_path: Path) -> Path:
        return self.out_dir or catalog_path.resolve().parent / "graph"


_SECTIONS = {
    "paths": PathSettings,
    "llm": LLMSettings,
    "indexing": IndexSettings,
    "server": ServerSettings,
    "optimize": OptimizeSettings,
    "graph": GraphBuildSettings,
}


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _coerce(value: Any, target_type: Any, where: str) -> Any:
    """Turn a YAML/env value into the type the dataclass field declares."""
    if value is None:
        return None
    type_name = getattr(target_type, "__name__", str(target_type))

    if "Path" in type_name or target_type is Path:
        return Path(os.path.expanduser(str(value)))
    if target_type is bool or type_name == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    if target_type is int or type_name == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected an integer, got {value!r}") from None
    if target_type is float or type_name == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected a number, got {value!r}") from None
    return value


def _coerce_field(section_name: str, field_name: str, declared: str, value: Any) -> Any:
    """Coerce one value against a dataclass field's (string) type annotation."""
    where = f"{section_name}.{field_name}"
    if value is None:
        return None

    if declared.startswith("list["):
        if isinstance(value, str):
            # Env vars and single-value YAML entries: ':' or ',' separated.
            value = [v.strip() for v in re.split(r"[:,]", value) if v.strip()]
        elif not isinstance(value, list):
            value = [value]
        inner = declared[len("list["):-1]
        return [_coerce(v, Path if "Path" in inner else str, where) for v in value]

    if declared.startswith("dict["):
        if isinstance(value, str):
            out: dict[str, str] = {}
            for part in value.split(","):
                name, sep, val = part.partition(":")
                if not sep:
                    raise ConfigError(f"{where}: expected NAME:VALUE pairs, got {part!r}")
                out[name.strip()] = val.strip()
            return out
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return {str(k): str(v) for k, v in value.items()}

    base = declared.split("|")[0].strip()
    simple = {"int": int, "float": float, "bool": bool, "str": str, "Path": Path}
    return _coerce(value, simple.get(base, str), where)


# ---------------------------------------------------------------------------
# Env interpolation in the YAML file
# ---------------------------------------------------------------------------

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` inside config strings.

    Keeps secrets out of the config file: `api_key: ${LLM_API_KEY}` reads the
    key from the environment at load time, and an unset variable with no
    default is left as-is so the problem is visible rather than silently empty.
    """
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            env = os.getenv(name)
            if env is not None:
                return env
            if default is not None:
                return default
            log.warning("Config references ${%s}, which is not set — left unexpanded", name)
            return match.group(0)
        return _ENV_REF.sub(replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def discover_config_file(explicit: str | Path | None = None) -> Path | None:
    """Find the config file to use, or None when there is none."""
    if explicit:
        path = Path(os.path.expanduser(str(explicit)))
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path
    from_env = os.getenv("DOCCAT_CONFIG")
    if from_env:
        path = Path(os.path.expanduser(from_env))
        if not path.is_file():
            raise ConfigError(f"DOCCAT_CONFIG points at a missing file: {path}")
        return path
    for candidate in CONFIG_SEARCH_PATH:
        if candidate.is_file():
            return candidate
    return None


@dataclass
class Settings:
    """The complete, resolved configuration."""

    paths: PathSettings = field(default_factory=PathSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    indexing: IndexSettings = field(default_factory=IndexSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    optimize: OptimizeSettings = field(default_factory=OptimizeSettings)
    graph: GraphBuildSettings = field(default_factory=GraphBuildSettings)

    #: Provenance, for `doccat config show`.
    config_file: Path | None = None
    env_overrides: list[str] = field(default_factory=list)

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, config_file: str | Path | None = None, *,
             use_env: bool = True) -> "Settings":
        """Build settings from defaults, then the config file, then the environment."""
        settings = cls()
        path = discover_config_file(config_file)
        if path is not None:
            settings._apply_file(path)
            settings.config_file = path
        if use_env:
            settings._apply_env()
        return settings

    def _apply_file(self, path: Path) -> None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: invalid YAML — {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")

        unknown = set(raw) - set(_SECTIONS)
        if unknown:
            raise ConfigError(f"{path}: unknown section(s) {sorted(unknown)}; "
                              f"allowed: {sorted(_SECTIONS)}")

        for section_name, values in raw.items():
            if values is None:
                continue
            if not isinstance(values, dict):
                raise ConfigError(f"{path}: section {section_name!r} must be a mapping")
            section = getattr(self, section_name)
            known = {f.name: f.type for f in fields(section)}
            bad = set(values) - set(known)
            if bad:
                raise ConfigError(f"{path}: unknown key(s) in {section_name}: {sorted(bad)}; "
                                  f"allowed: {sorted(known)}")
            for key, value in values.items():
                declared = known[key] if isinstance(known[key], str) else str(known[key])
                coerced = _coerce_field(section_name, key, declared, _interpolate(value))
                setattr(section, key, _anchor(coerced, path.resolve().parent))

    def _apply_env(self) -> None:
        for dotted, env_name in ENV_MAP.items():
            raw = os.getenv(env_name)
            if raw is None or raw == "":
                continue
            section_name, key = dotted.split(".")
            section = getattr(self, section_name)
            declared = {f.name: f.type for f in fields(section)}[key]
            declared = declared if isinstance(declared, str) else str(declared)
            setattr(section, key, _coerce_field(section_name, key, declared, raw))
            self.env_overrides.append(f"{dotted} <- ${env_name}")

    # -- reporting ---------------------------------------------------------
    def as_dict(self, mask_secrets: bool = True) -> dict:
        """The resolved settings, ready to print. Secrets masked by default."""
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if is_dataclass(value):
                return {k: convert(v) for k, v in asdict(value).items()}
            return value

        out: dict[str, Any] = {}
        for section_name in _SECTIONS:
            section = getattr(self, section_name)
            values = {f.name: convert(getattr(section, f.name)) for f in fields(section)}
            if mask_secrets:
                for f_name in list(values):
                    if f"{section_name}.{f_name}" in _SECRET_FIELDS and values[f_name]:
                        values[f_name] = _mask(str(values[f_name]))
                if section_name == "llm" and values.get("headers"):
                    values["headers"] = {k: _mask(str(v)) for k, v in values["headers"].items()}
            out[section_name] = values

        # Derived values people actually need to see when debugging a deployment.
        out["paths"]["text_cache (effective)"] = str(self.paths.resolved_text_cache())
        out["paths"]["error_log (effective)"] = str(self.paths.resolved_error_log())
        out["_source"] = {
            "config_file": str(self.config_file) if self.config_file else None,
            "env_overrides": self.env_overrides or None,
        }
        return out

    def extra_headers(self) -> dict[str, str]:
        """HTTP headers for LLM requests, including the HF billing header."""
        headers = dict(self.llm.headers)
        if self.llm.bill_to:
            headers["X-HF-Bill-To"] = self.llm.bill_to
        return headers


def _anchor(value: Any, base: Path) -> Any:
    """Make relative paths from a config file relative to *that file*, not the cwd.

    A config at /etc/doccat/config.yaml saying `docs_root: ./documents` means
    /etc/doccat/documents, whatever directory the command happens to run from.
    """
    if isinstance(value, Path):
        return value if value.is_absolute() else (base / value).resolve()
    if isinstance(value, list):
        return [_anchor(v, base) for v in value]
    return value


def _mask(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-2:]} ({len(value)} chars)"


CONFIG_TEMPLATE = """\
# doccat configuration. Every key is optional — what you leave out keeps its
# default. Environment variables override this file, and command-line flags
# override both, so a container can change one value without editing anything.
#
# `${VAR}` and `${VAR:-fallback}` are expanded from the environment at load
# time: keep secrets out of this file and in the environment.
#
#   doccat config show          # the fully resolved settings, secrets masked
#   doccat index                # uses everything below, no flags needed

paths:
  docs_root: ./documents           # the corpus to catalogue
  catalog: ./catalog.jsonl         # the catalog to build and serve
  # text_cache: ./extracted_text   # default: next to the catalog
  # error_log: ./llm_errors        # default: next to the catalog
  profiles_dir:                    # your custom cataloguing profiles
    - ./profiles

llm:
  base_url: http://localhost:8080/v1
  model: qwen2.5-14b-instruct
  api_key: ${LLM_API_KEY:-not-needed}
  temperature: 0.1
  max_tokens: 4096
  timeout: 300
  think: false                     # keep the model's reasoning mode enabled
  requests_per_second: 0           # 0 = unlimited
  # bill_to: BSC-LT                # Hugging Face router: sends X-HF-Bill-To
  # headers:
  #   X-Custom-Header: value

indexing:
  llm_workers: 8                   # concurrent LLM requests
  extract_workers: 0               # 0 = auto (min(cpu, 8)); 1 = extract inline
  mode: auto                       # parse | upload | auto
  # profile: legal                 # force one profile instead of auto-routing
  checkpoint_every: 50
  progress_every: 20
  change_detection: hash           # hash | mtime (cheaper on huge trees)
  # shard: 0/4                     # this machine's slice of the corpus
  include: []                      # glob patterns, relative to docs_root
  exclude: []

server:
  host: 0.0.0.0
  port: 8000
  stateless: true                  # robust behind proxies and restarts

optimize:
  sample_size: 12                  # documents `doccat optimize` reads
  refine_rounds: 1                 # critique-and-improve passes
  sample_chars: 6000               # characters shown per sampled document

# Corpus-level graph consolidation (`doccat graph build`) — distinct from a
# profile's own `graph:` block, which controls what gets *extracted* per
# document. This section controls how those results get *consolidated* into
# one corpus graph. See docs/GRAPH.md.
graph:
  # out_dir: ./graph                # default: 'graph' next to the catalog
  detect_communities: true
  min_community_size: 2
  summarize: false                  # one LLM call per community if true
  community_seed: 0
"""
