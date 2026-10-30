"""Reading Calliope 0.7 result files (NetCDF with `inputs` and `results` groups) without importing Calliope."""

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tidy_calliope.timeinfo import TimeInfo


SPORE_FILE = re.compile(r"^spore_(\d+)$")  # Calliope's `save_per_spore_path` files: spore_<n>.nc


@dataclass
class ModelResults:
    """The inputs and results of one model run, or of one SPORE (`spore`) of a SPORES run."""

    path: Path
    scenario: str
    inputs: xr.Dataset
    results: xr.Dataset
    spore: int | None = None
    time: TimeInfo | None = field(default=None)
    # the opened files (DataTrees): kept until `close`, so they are never closed by the garbage collector while
    # their (lazily read) data is still in use
    handles: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.time is None:
            self.time = TimeInfo(self.inputs)

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles.clear()


@dataclass
class Run:
    """Files processed together into one output folder: one result file, or the per-SPORE files of one folder."""

    scenario: str
    files: list[Path]


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


def _groups(path: Path) -> tuple[xr.DataTree, xr.Dataset | None, xr.Dataset]:
    """The opened file and its `inputs` (None if absent) and `results` groups. Close the file when done."""
    try:
        tree = xr.open_datatree(path)
    except (OSError, ValueError) as err:
        raise ResultFileError(f"{path}: not a Calliope 0.7 result file (needs `inputs` and `results` groups): {err}")
    groups = {node.path for node in tree.subtree}
    if "/results" not in groups:
        tree.close()
        raise ResultFileError(f"{path}: not a Calliope 0.7 result file (needs `inputs` and `results` groups)")
    inputs = tree["inputs"].to_dataset() if "/inputs" in groups else None
    return tree, inputs, tree["results"].to_dataset()


def open_results(path: str | Path, inputs: xr.Dataset | None = None) -> ModelResults:
    """One result file. `inputs`: for per-SPORE files without inputs (Calliope writes them in `spore_0.nc` only)."""
    path = Path(path)
    tree, own_inputs, results = _groups(path)
    if own_inputs is None and inputs is None:
        tree.close()
        raise ResultFileError(f"{path}: not a Calliope 0.7 result file (needs `inputs` and `results` groups)")
    inputs = own_inputs if own_inputs is not None else inputs
    if "spores" in inputs.dims:  # per-SPORE files: inputs of spore_0 carry a `spores` dimension of size 1
        inputs = inputs.isel(spores=0, drop=True)
    if not results.data_vars:
        termination = results.attrs.get("termination_condition")
        detail = f" (termination condition: {termination})" if termination else ""
        tree.close()
        raise ResultFileError(f"{path}: no results{detail}: the model was not solved to optimality")
    return ModelResults(path=path, scenario=path.stem, inputs=inputs, results=results, handles=[tree])


def spore_views(model: ModelResults) -> list[ModelResults]:
    """One ModelResults per SPORE if the results have a `spores` dimension (sharing inputs and time), else [model]."""
    if "spores" not in model.results.dims:
        return [model]
    return [
        ModelResults(path=model.path, scenario=model.scenario, inputs=model.inputs,
                     results=model.results.sel(spores=s, drop=True), spore=int(s), time=model.time,
                     handles=model.handles)
        for s in model.results["spores"].values
    ]


def group_runs(files: list[Path]) -> list[Run]:
    """Per-SPORE files (`spore_<n>.nc`) of one folder form one run named after the folder; other files are runs."""
    runs: dict[str, Run] = {}
    for f in files:
        if SPORE_FILE.match(f.stem):
            key = str(f.parent)
            runs.setdefault(key, Run(scenario=f.parent.name, files=[])).files.append(f)
        else:
            runs[str(f)] = Run(scenario=f.stem, files=[f])
    for run in runs.values():
        run.files.sort(key=lambda f: int(SPORE_FILE.match(f.stem).group(1)) if SPORE_FILE.match(f.stem) else -1)
    return list(runs.values())


def open_run(run: Run) -> list[ModelResults]:
    """The models (one per SPORE for SPORES runs) of a run."""
    if len(run.files) == 1 and not SPORE_FILE.match(run.files[0].stem):
        return spore_views(open_results(run.files[0]))
    # per-SPORE files: the inputs are in spore_0.nc of the same folder (also when only some SPOREs are given)
    folder = run.files[0].parent
    inputs, inputs_tree = None, None
    for candidate in [folder / "spore_0.nc", *run.files]:
        if candidate.exists():
            tree, own_inputs, _ = _groups(candidate)
            if own_inputs is not None:
                inputs, inputs_tree = own_inputs, tree
                break
            tree.close()
    if inputs is None:
        raise ResultFileError(f"{folder}: per-SPORE files without inputs (Calliope writes them in spore_0.nc)")
    views = []
    for f in run.files:
        model = open_results(f, inputs=inputs)
        model.scenario = run.scenario
        model.handles.append(inputs_tree)  # closed with the run (DataTree.close is idempotent)
        views.extend(spore_views(model))
    return views


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
