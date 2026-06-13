"""
Open registry of GPU / model / cost / runtime / anchor profiles.

The catalog directory (catalog/) is the source of truth for seeded entries.
Users add entries by editing those YAMLs or by calling register_model() /
register_gpu(), which persist to a writable user-catalog dir.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_CATALOG = Path(__file__).parent.parent / "catalog"
_USER_CATALOG = Path.home() / ".llm-inference-bench" / "catalog"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CatalogError(ValueError):
    """Raised for missing / invalid catalog entries."""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PeakFlops(BaseModel):
    fp16: Optional[float] = None
    bf16: Optional[float] = None
    fp8: Optional[float] = None
    mxfp4: Optional[float] = None

    def get(self, dtype: str) -> float:
        value = getattr(self, dtype, None)
        if value is None:
            raise CatalogError(f"dtype '{dtype}' not available for this GPU")
        return value


class GpuProfile(BaseModel):
    name: str  # injected by loader from the dict key
    display_name: str
    mem_gb: float
    hbm_bandwidth_gbps: float
    peak_flops: PeakFlops
    default_mfu_prefill: float = 0.40
    default_bw_efficiency_decode: float = 0.70


class ModelProfile(BaseModel):
    name: str
    display_name: str
    is_moe: bool = False
    total_params: int
    active_params: int
    num_layers: int
    d_model: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    native_dtype: str
    weight_bytes_per_param: float
    kv_dtype_bytes: int = 1
    geometry_source: Literal["known", "estimated"] = "known"
    context_len: int = 131072
    # Optional: explicit checkpoint size overrides weight_bytes_per_param * total_params
    resident_weights_gb: Optional[float] = None
    # MoE extras (informational)
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None

    @property
    def resident_weights_bytes(self) -> float:
        if self.resident_weights_gb is not None:
            return self.resident_weights_gb * 1e9
        return self.total_params * self.weight_bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes


class CostProfile(BaseModel):
    gpu_name: str
    on_demand_usd_per_hour: float
    reserved_usd_per_hour: float


class RuntimeProfile(BaseModel):
    name: str
    display_name: str
    openai_compatible: bool = True
    flags: list[str] = []
    notes: str = ""


class Anchor(BaseModel):
    model: str
    gpu: str
    dtype: str
    isl: int
    osl: int
    concurrency: int
    measured_ttft_p50_ms: Optional[float] = None
    measured_ttft_p95_ms: Optional[float] = None
    measured_throughput_tok_s: Optional[float] = None
    derived_mfu_prefill: Optional[float] = None
    source: str = ""
    # Optional field present in some anchor rows (mns sweep)
    max_num_seqs: Optional[int] = None


# ---------------------------------------------------------------------------
# Catalog container
# ---------------------------------------------------------------------------


class Catalog(BaseModel):
    gpus: dict[str, GpuProfile]
    models: dict[str, ModelProfile]
    costs: dict[str, CostProfile]
    runtimes: dict[str, RuntimeProfile]
    anchors: list[Anchor]


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    with path.open() as f:
        return yaml.safe_load(f)


def _merge_yaml(filename: str) -> Any:
    """Merge repo catalog + user catalog (user entries override repo entries)."""
    data = {}
    repo_path = _REPO_CATALOG / filename
    user_path = _USER_CATALOG / filename

    if repo_path.exists():
        loaded = _load_yaml(repo_path)
        if isinstance(loaded, dict):
            data.update(loaded)

    if user_path.exists():
        loaded = _load_yaml(user_path)
        if isinstance(loaded, dict):
            data.update(loaded)  # user entries win

    return data


def _merge_yaml_list(filename: str) -> list:
    """Merge lists from repo + user catalog."""
    items = []
    repo_path = _REPO_CATALOG / filename
    user_path = _USER_CATALOG / filename

    if repo_path.exists():
        loaded = _load_yaml(repo_path)
        if isinstance(loaded, list):
            items.extend(loaded)

    if user_path.exists():
        loaded = _load_yaml(user_path)
        if isinstance(loaded, list):
            items.extend(loaded)

    return items


def load_catalog() -> Catalog:
    """Parse all catalog YAMLs and return a validated Catalog."""
    raw_gpus = _merge_yaml("gpus.yaml")
    raw_models = _merge_yaml("models.yaml")
    raw_costs = _merge_yaml("costs.yaml")
    raw_runtimes = _merge_yaml("runtimes.yaml")
    raw_anchors = _merge_yaml_list("anchors.yaml")

    gpus = {
        name: GpuProfile(name=name, **data)
        for name, data in raw_gpus.items()
    }

    models = {
        name: ModelProfile(name=name, **data)
        for name, data in raw_models.items()
    }

    costs = {
        name: CostProfile(gpu_name=name, **data)
        for name, data in raw_costs.items()
    }

    runtimes = {
        name: RuntimeProfile(name=name, **data)
        for name, data in raw_runtimes.items()
    }

    anchors = [Anchor(**row) for row in raw_anchors if row is not None]

    return Catalog(
        gpus=gpus,
        models=models,
        costs=costs,
        runtimes=runtimes,
        anchors=anchors,
    )


# Module-level singleton — loaded once per process.
_catalog: Optional[Catalog] = None


def _get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = load_catalog()
    return _catalog


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_gpu(name: str) -> GpuProfile:
    cat = _get_catalog()
    if name not in cat.gpus:
        available = ", ".join(sorted(cat.gpus))
        raise CatalogError(
            f"Unknown GPU '{name}'. Available: {available}. "
            "Add it to catalog/gpus.yaml or pass --gpu-spec."
        )
    return cat.gpus[name]


def get_model(name: str) -> ModelProfile:
    cat = _get_catalog()
    if name not in cat.models:
        available = ", ".join(sorted(cat.models))
        raise CatalogError(
            f"Unknown model '{name}'. Available: {available}. "
            "Add it to catalog/models.yaml or use --model-spec / --model-params."
        )
    return cat.models[name]


def find_anchors(
    model: str,
    gpu: str,
    dtype: str,
    isl: int,
    osl: int,
    concurrency: int,
    isl_tolerance: float = 0.20,
) -> list[Anchor]:
    """Return anchors matching (model, gpu, dtype) within ±isl_tolerance of isl.

    The concurrency band match is loose — within ±50% is close enough to be
    informative for confidence scoring.
    """
    cat = _get_catalog()
    results = []
    for a in cat.anchors:
        if a.model != model or a.gpu != gpu or a.dtype != dtype:
            continue
        isl_match = abs(a.isl - isl) / max(isl, 1) <= isl_tolerance
        if isl_match:
            results.append(a)
    return results


# ---------------------------------------------------------------------------
# Geometry estimation (param-only rough spec)
# ---------------------------------------------------------------------------


DTYPE_BYTES: dict[str, float] = {
    "fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "mxfp4": 0.5, "int4": 0.5,
}


def _estimate_geometry(total_params: int, active_params: int, dtype: str) -> dict:
    """Estimate transformer geometry from param count.

    Uses an empirical lookup table anchored to known dense/MoE models.
    Applied only when full geometry is not provided.
    """
    p = total_params
    if p < 2e9:
        num_layers, d_model = 24, 2048
    elif p < 10e9:
        num_layers, d_model = 32, 4096
    elif p < 20e9:
        num_layers, d_model = 40, 5120
    elif p < 40e9:
        num_layers, d_model = 48, 6144
    elif p < 80e9:
        num_layers, d_model = 80, 8192
    else:
        num_layers, d_model = 96, 8192

    num_q_heads = d_model // 128
    num_kv_heads = max(8, num_q_heads // 8)
    head_dim = 128

    weight_bytes = DTYPE_BYTES.get(dtype, 2.0)

    return {
        "num_layers": num_layers,
        "d_model": d_model,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "weight_bytes_per_param": weight_bytes,
    }


# ---------------------------------------------------------------------------
# Open-registry surface
# ---------------------------------------------------------------------------


def resolve_model(name_or_spec: str | dict) -> ModelProfile:
    """Resolve a model from: (a) catalog name, (b) full inline spec dict,
    (c) param-only rough spec dict with total_params + dtype.

    Rough specs return geometry_source='estimated' and emit a warning.
    """
    if isinstance(name_or_spec, str):
        return get_model(name_or_spec)

    spec = dict(name_or_spec)
    name = spec.setdefault("name", spec.get("display_name", "custom-model"))
    spec.setdefault("display_name", name)

    _FULL_GEOMETRY = {"num_layers", "d_model", "num_q_heads", "num_kv_heads", "head_dim"}
    has_full_geometry = _FULL_GEOMETRY.issubset(spec.keys())

    if not has_full_geometry:
        if "total_params" not in spec:
            raise CatalogError(
                "Custom model spec must supply either full geometry fields "
                "(num_layers, d_model, num_q_heads, num_kv_heads, head_dim) "
                "or at least total_params + dtype."
            )
        total_params = int(float(spec["total_params"]))
        active_params = int(float(spec.get("active_params", total_params)))
        dtype = spec.get("native_dtype", "bf16")
        estimated = _estimate_geometry(total_params, active_params, dtype)
        for k, v in estimated.items():
            spec.setdefault(k, v)
        spec["active_params"] = active_params
        spec["total_params"] = total_params
        spec["geometry_source"] = "estimated"
    else:
        spec.setdefault("geometry_source", "known")

    spec.setdefault("is_moe", False)
    if "active_params" not in spec:
        spec["active_params"] = spec["total_params"]
    spec.setdefault("native_dtype", "bf16")
    spec.setdefault("weight_bytes_per_param", 2.0)
    spec.setdefault("kv_dtype_bytes", 2)

    return ModelProfile(**spec)


def resolve_gpu(name_or_spec: str | dict) -> GpuProfile:
    """Resolve a GPU from: (a) catalog name, (b) full inline spec dict.

    No rough-spec path for GPUs (entering bandwidth / FLOPs is a correctness
    risk; the catalog is the right place for new cards).
    """
    if isinstance(name_or_spec, str):
        return get_gpu(name_or_spec)

    spec = dict(name_or_spec)
    name = spec.setdefault("name", spec.get("display_name", "custom-gpu"))
    spec.setdefault("display_name", name)

    required = {"mem_gb", "hbm_bandwidth_gbps", "peak_flops"}
    missing = required - spec.keys()
    if missing:
        raise CatalogError(
            f"Custom GPU spec missing required fields: {missing}. "
            "See catalog/gpus.yaml for the full schema."
        )

    return GpuProfile(**spec)


def register_model(spec: dict) -> ModelProfile:
    """Validate spec and persist it to the user-catalog dir."""
    profile = resolve_model(spec)
    _USER_CATALOG.mkdir(parents=True, exist_ok=True)
    path = _USER_CATALOG / "models.yaml"
    existing = {}
    if path.exists():
        loaded = _load_yaml(path)
        if isinstance(loaded, dict):
            existing = loaded

    row = profile.model_dump(exclude={"name"})
    existing[profile.name] = row

    with path.open("w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

    # Invalidate singleton so next call picks up the new entry.
    global _catalog
    _catalog = None

    return profile


def register_gpu(spec: dict) -> GpuProfile:
    """Validate spec and persist it to the user-catalog dir."""
    profile = resolve_gpu(spec)
    _USER_CATALOG.mkdir(parents=True, exist_ok=True)
    path = _USER_CATALOG / "gpus.yaml"
    existing = {}
    if path.exists():
        loaded = _load_yaml(path)
        if isinstance(loaded, dict):
            existing = loaded

    row = profile.model_dump(exclude={"name"})
    existing[profile.name] = row

    with path.open("w") as f:
        yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)

    global _catalog
    _catalog = None

    return profile
