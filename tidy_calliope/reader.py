"""Reading Calliope 0.7 result files (NetCDF with `inputs` and `results` groups) without importing Calliope."""

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tidy_calliope.timeinfo import TimeInfo


@dataclass
class ModelResults:
    path: Path
    scenario: str
    inputs: xr.Dataset
    results: xr.Dataset
    time: TimeInfo = field(init=False)

    def __post_init__(self):
        self.time = TimeInfo(self.inputs)

    def close(self):
        self.inputs.close()
        self.results.close()


def find_result_files(paths: list[str | Path]) -> list[Path]:
    """NetCDF files from a list of files, folders (all `*.nc` inside) and glob patterns, in a stable order."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.glob("*.nc")))
        elif p.is_file():
            files.append(p)
        else:
            matches = sorted(Path(m) for m in glob.glob(str(p)))
            if not matches:
                raise FileNotFoundError(f"No result file or folder found for `{p}`")
            files.extend(m for m in matches if m.suffix == ".nc")
    unique = list(dict.fromkeys(f.resolve() for f in files))
    if not unique:
        raise FileNotFoundError(f"No .nc files in {[str(p) for p in paths]}")
    return unique


class ResultFileError(ValueError):
    """A file that cannot be processed: not a Calliope 0.7 result file, or a model without a solution."""


def open_results(path: str | Path) -> ModelResults:
    path = Path(path)
    try:
        tree = xr.open_datatree(path)
        inputs, results = tree["inputs"].to_dataset(), tree["results"].to_dataset()
    except (KeyError, OSError, ValueError) as err:
        raise ResultFileError(f"{path}: not a Calliope 0.7 result file (needs `inputs` and `results` groups): {err}")
    if not results.data_vars:
        termination = results.attrs.get("termination_condition")
        detail = f" (termination condition: {termination})" if termination else ""
        raise ResultFileError(f"{path}: no results{detail}: the model was not solved to optimality")
    return ModelResults(path=path, scenario=path.stem, inputs=inputs, results=results)


def to_long(da: xr.DataArray, zero_threshold: float = 0.0, chunk_size: float = 2e8, techs=None) -> pd.Series:
    """Non-empty values of `da` (optionally only `techs`) as a Series with one index level per dimension.

    Large arrays (e.g. flows with a time dimension) are read in contiguous blocks of techs aligned with the file's
    compression chunks (each chunk is decompressed once), so the dense array never has to fit in memory at once.
    """
    keep = None if techs is None else set(techs)

    def _clean(sub: xr.DataArray) -> pd.Series:
        if keep is not None:
            sub = sub.isel(techs=[i for i, t in enumerate(sub.indexes["techs"]) if t in keep])
            if sub.sizes["techs"] == 0:
                return pd.Series(dtype=float)
        values = sub.values
        mask = ~np.isnan(values)
        if zero_threshold > 0:
            mask &= np.abs(values) > zero_threshold
        if not mask.any():
            return pd.Series(dtype=float)
        idx = np.nonzero(mask)
        index = pd.MultiIndex.from_arrays(
            [sub.indexes[dim][i] for dim, i in zip(sub.dims, idx)], names=list(sub.dims)
        )
        return pd.Series(values[mask].astype(float), index=index)

    if "techs" not in da.dims or da.size <= chunk_size:
        return _clean(da.load())
    per_tech = da.size / da.sizes["techs"]
    file_chunks = da.encoding.get("chunksizes")
    block = file_chunks[da.dims.index("techs")] if file_chunks else 1
    step = block * max(1, int(chunk_size // (per_tech * block)))
    parts = [_clean(da.isel(techs=slice(i, i + step)).load()) for i in range(0, da.sizes["techs"], step)]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts) if parts else pd.Series(dtype=float)
