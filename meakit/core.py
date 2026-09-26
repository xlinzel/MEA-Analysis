"""

"""

from __future__ import annotations

# std library imports
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field, replace, asdict
import sys
import os
import configparser
from pathlib import Path

# maxwell plugin path setup
def _configure_maxwell_plugin():
    plugin_path = Path.home() / "hdf5_plugin_path_maxwell"

    lib = {
        "win32": "compression.dll",
        "darwin": "libcompression.dylib",
    }.get(sys.platform, "libcompression.so")

    if not (plugin_path / lib).exists():
        from neo.rawio.maxwellrawio import (
            auto_install_maxwell_hdf5_compression_plugin,
        )

        auto_install_maxwell_hdf5_compression_plugin()

    if not (plugin_path / lib).exists():
        raise RuntimeError(
            f"Maxwell compression plugin was not found at {plugin_path}"
        )

    os.environ["HDF5_PLUGIN_PATH"] = str(plugin_path)

_configure_maxwell_plugin()

# third-party imports
import numpy as np
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# logger
logger = logging.getLogger(__name__)

# type alias
ProgressFn = Callable[[str, float], None]

# default analyzer extensions
DEFAULT_EXTENSIONS = {
    "random_spikes": {},
    "waveforms": {},
    "templates": {},
    "noise_levels": {},
    "spike_amplitudes": {},
    "spike_locations": {},
    "unit_locations": {"method": "center_of_mass"},
    "correlograms": {},
    "template_similarity": {"method": "cosine", "max_lag_ms": 1.0},
    "quality_metrics": {},
    "template_metrics": {},
}

# dataclasses
@dataclass(frozen=True)
class Params:
    freq_min: float = 300.0
    freq_max: float = 6000.0
    common_reference: str = "median"
    detect_mad: float = 5.0
    merge_thresh: float = 0.25
    min_snr: float = 3.0
    max_isi_viol: float = 0.5
    sorter: str = "herdingspikes"
    sorter_params: dict = field(default_factory=dict)
    extensions: dict = field(default_factory=lambda: dict(DEFAULT_EXTENSIONS))

@dataclass(frozen=True)
class Project:
    dir: Path
    title: str = ""


@dataclass(frozen=True)
class Data:
    name: str
    project_title: str
    tag: str
    recording: si.BaseRecording
    locations: np.ndarray
    pitch_um: float
    sorting: si.BaseSorting | None = None
    analyzer: si.SortingAnalyzer | None = None
    peaks: np.ndarray | None = None


# public functions
def configure_runtime(n_jobs=4, chunk_duration="1s") -> None:
    global_job_kwargs = dict(
        n_jobs=n_jobs, 
        chunk_duration=chunk_duration
    )
    si.set_global_job_kwargs(**global_job_kwargs)


def load(path: Path) -> Data:

    logger.info("Loading %s", path)

    if path.suffixes[-2:] == [".raw", ".h5"]:
        try:
            recording = si.read_maxwell(path)
        except Exception as e:
            logger.exception("Failed to load %s as data file", path)
            raise RuntimeError(
                f"{path.name}: could not open. If this is an HDF5 filter error, "
                f"the MaxWell decompression plugin is missing."
            ) from e

    else:
        try:
            recording = si.load(path)
        except Exception as e:
            logger.exception("Failed to load %s from spikeinterface", path)
            raise RuntimeError(
                f"{path.name}: could not open. The given file was not "
                f"a Maxwell file or any other loadable file."
            ) from e

    locations = recording.get_channel_locations()
    if locations is None or not np.any(locations):
        raise ValueError(f"{path.name}: no probe geometry")

    xs = np.unique(locations[:, 0])
    if len(xs) < 2:
        raise ValueError(f"{path.name}: all electrodes in one column")
    pitch = float(np.min(np.diff(xs)))

    metadata = _read_md(path.parent / "mxassay.metadata")

    logger.info(
        "%s: %d ch, %.1f s, %.0f Hz, pitch %.1f µm", 
        path.name, 
        recording.get_num_channels(), 
        recording.get_total_duration(), 
        recording.get_sampling_frequency(), 
        pitch
    )

    return Data(
        name=path.parent.name.split('.')[0],
        project_title=metadata["project_title"],
        tag=metadata["tag"],
        recording=recording, 
        locations=locations, 
        pitch_um=pitch
    )


def preprocess(data: Data, params: Params, project: Project, save: bool = False, force: bool = True) -> Data:

    logger.info("Preprocessing %s", data.name)

    save_path = result_dir(project, data.name) / "preprocessed"

    def pb():
        logger.info(
            "Bandpass: %.0f–%.0f Hz, reference=%s",
            params.freq_min,
            params.freq_max,
            params.common_reference,
        )
        
        rec_sign = si.unsigned_to_signed(data.recording, bit_depth=10)

        rec_uv = si.scale_to_uV(rec_sign)

        rec_filt = si.bandpass_filter(
            rec_uv, 
            freq_min=params.freq_min, 
            freq_max=params.freq_max, 
            dtype="float32"
        )
        rec_ref = si.common_reference(
            rec_filt, 
            reference="global", 
            operator=params.common_reference
        )

        if save:
            rec_ref.save_to_folder(
                folder = save_path, 
                overwrite=True, 
                format='binary'
            )

            logger.info("Saved preprocessed recording to %s", save_path)

        return rec_ref

    if force or not save_path.is_dir():
        rec_ref = pb()
    else:
        try:
            logger.info("Loading cached preprocessing from %s", save_path)
            rec_ref = si.load(file_or_folder_or_dict=save_path)
        except Exception:
            
            logger.exception("Failed to load cached preprocessing from %s", save_path)

            rec_ref = pb()

    logger.info("Preprocessing complete: %s", data.name)

    return replace(data, recording=rec_ref)

def detect(data: Data, params: Params, project: Project, save: bool = False, force: bool = True) -> Data:

    logger.info("Starting peak detection %s", data.name)
    
    save_path = result_dir(project, data.name) / "peaks.npy"

    def db():
        noise = si.get_noise_levels(recording=data.recording, method="mad", return_in_uV=True)

        peaks = detect_peaks(
            data.recording,
            method="by_channel",
            method_kwargs=dict(
                peak_sign="neg",
                detect_threshold=params.detect_mad,
                exclude_sweep_ms=0.1,
                noise_levels=noise,
            ),
        )

        if save:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, peaks)

            logger.info("Saved peaks to %s", save_path)

        return peaks

    if force or not save_path.is_file():
        peaks = db()
    else:
        try:
            logger.info("Loading cached peaks from %s", save_path)

            peaks = np.load(save_path)
        except Exception:
            logger.exception("Failed to load cached peaks from %s", save_path)

            peaks = db()

    logger.info("Peak detection complete: %s, %d peaks", data.name, len(peaks))

    return replace(data, peaks=peaks)


def sort(data: Data, params: Params, project: Project, force: bool = True) -> Data:

    logger.info("Sorting %s with %s", data.name, params.sorter)

    save_path = result_dir(project, data.name) / f"sorted_{params.sorter}"

    def sb():
        merged = si.get_default_sorter_params(params.sorter) | params.sorter_params
        return si.run_sorter(
            sorter_name=params.sorter,
            recording=data.recording,
            folder=save_path,
            remove_existing_folder=True,
            **merged,
        )

    if force or not save_path.is_dir():
        sorted_data = sb()
    else:
        try:
            logger.info("Loading cached sorting from %s", save_path)

            sorted_data = si.load(file_or_folder_or_dict=save_path)
        except Exception:
            logger.exception("Failed to load cached sorting data from %s", save_path)

            sorted_data = sb()

    logger.info(
        "Sorting complete: %d units found",
        sorted_data.get_num_units(),
    )

    return replace(data, sorting=sorted_data)


def analyze(data: Data, params: Params,  project: Project, force: bool = True) -> Data:

    logger.info("Analyzing %s", data.name)

    save_path = result_dir(project, data.name) / f"analyzer_{params.sorter}"

    def _build_analyzer(data: Data, save_path: Path) -> si.SortingAnalyzer:
        logger.info("Creating sorting analyzer at %s", save_path)

        sp = si.estimate_sparsity(
            data.sorting, data.recording,
            method="snr", threshold=2,
            noise_levels=si.get_noise_levels(
                data.recording, method="mad", return_in_uV=True
            ),
        )
        best = si.estimate_sparsity(
            data.sorting, data.recording,
            method="best_channels", num_channels=1,
        )
        sp.mask |= best.mask

        return si.create_sorting_analyzer(
            sorting=data.sorting,
            recording=data.recording,
            format='binary_folder',
            folder=save_path,
            overwrite=True,
            sparsity=sp
        )

    if force or not save_path.is_dir():
        logger.info("Creating sorting analyzer at %s", save_path)

        analyzer = _build_analyzer(data, save_path)
        
    else:
        analyzer = None
        try:
            logger.info("Loading cached analyzer from %s", save_path)

            analyzer = si.load(file_or_folder_or_dict=save_path)

            if analyzer.get_num_units() != data.sorting.get_num_units():
                logger.warning(
                    "Cached analyzer has %d units, sorting has %d — rebuilding",
                    analyzer.get_num_units(), data.sorting.get_num_units(),
                )
                analyzer = None
            else:
                for ext in analyzer.get_saved_extension_names():
                    if analyzer.get_extension(ext) is None:
                        logger.warning(
                            "Cached analyzer has broken extension %s — rebuilding", ext
                        )
                        analyzer = None
                        break

        except Exception:
            logger.exception("Failed to load cached analyzing data from %s", save_path)

        if analyzer is None:
            analyzer = _build_analyzer(data, save_path)

    logger.info("Analysis complete: %s", data.name)

    return replace(data, analyzer=analyzer)


def compute(data: Data, params: Params, force: bool = False) -> Data:
    have = set(data.analyzer.get_saved_extension_names())
    todo = params.extensions if force else {
        k: v for k, v in params.extensions.items() if k not in have
    }

    if not todo:
        logger.info("All extensions already computed for %s", data.name)
        return data

    logger.info("Computing %d extensions for %s", len(todo), data.name)
    data.analyzer.compute(todo)
    logger.info("Compute complete: %s", data.name)
    return data


def merge(data: Data, params: Params, project: Project, force: bool = False) -> Data:

    logger.info("Merging units for %s (thresh=%.2f)", data.name, params.merge_thresh)

    save_path = result_dir(project, data.name) / f"analyzer_{params.sorter}_merged"

    def mb():
        sim = data.analyzer.get_extension("template_similarity").get_data()
        uids = list(data.sorting.unit_ids)
        thresh = 1.0 - params.merge_thresh
        fs = data.recording.get_sampling_frequency()
        dur = data.recording.get_total_duration()

        def isi_ratio(times, refrac_ms=1.5):
            """SI-style violation ratio for a merged spike train."""
            n = len(times)
            if n < 2:
                return 0.0
            t = np.sort(times)
            viol = np.count_nonzero(np.diff(t) < refrac_ms / 1000)
            rate = n / dur
            expected = 2 * (refrac_ms / 1000) * rate * n
            return viol / expected if expected > 0 else 0.0

        trains = {u: data.sorting.get_unit_spike_train(u, return_times=True)
                  for u in uids}

        # candidate pairs, most similar first
        pairs = [(sim[i, j], uids[i], uids[j])
                 for i in range(len(uids)) for j in range(i + 1, len(uids))
                 if sim[i, j] >= thresh]
        pairs.sort(reverse=True)

        groups = {u: [u] for u in uids}        # representative -> members
        owner = {u: u for u in uids}

        for s, a, b in pairs:
            ra, rb = owner[a], owner[b]
            if ra == rb:
                continue
            combined = np.concatenate([trains[u] for u in groups[ra] + groups[rb]])
            r = isi_ratio(combined)
            if r <= params.max_isi_viol:
                groups[ra] = groups[ra] + groups[rb]
                for u in groups[rb]:
                    owner[u] = ra
                del groups[rb]
                logger.info("  merged %s + %s (sim %.3f, isi %.2f)",
                            a, b, s, r)
            else:
                logger.info("  rejected %s + %s (sim %.3f, isi %.2f)",
                            a, b, s, r)

        merge_groups = [g for g in groups.values() if len(g) > 1]
        logger.info("%s: %d merge groups (sim >= %.2f, isi <= %g)",
                    data.name, len(merge_groups), thresh, params.max_isi_viol)

        if not merge_groups:
            merged = data.analyzer
        else:
            merged = data.analyzer.merge_units(
                merge_unit_groups=merge_groups, merging_mode="hard")

        if save_path.exists():
            shutil.rmtree(save_path)
        merged.save_as(format="binary_folder", folder=save_path)
        return si.load_sorting_analyzer(save_path)

    if force or not save_path.is_dir():
        analyzer = mb()
    else:
        try:
            logger.info("Loading cached merge from %s", save_path)
            analyzer = si.load_sorting_analyzer(save_path)
        except Exception:
            logger.exception("Failed to load cached merge from %s", save_path)
            analyzer = mb()

    logger.info("Merge complete: %d -> %d units",
                data.analyzer.get_num_units(), analyzer.get_num_units())

    return replace(data, analyzer=analyzer, sorting=analyzer.sorting)


def curate(data: Data, params: Params, project: Project, force: bool = False) -> Data:
    """Drop units failing quality thresholds."""

    save_path = result_dir(project, data.name) / f"analyzer_{params.sorter}_curated"

    def cb():
        qm = data.analyzer.get_extension("quality_metrics").get_data()

        logger.info(
            "%s pre-curate:\n%s",
            data.name,
            qm[["snr", "isi_violations_ratio", "firing_rate", "num_spikes"]]
              .sort_values("snr", ascending=False)
              .to_string(),
        )

        keep = qm[(qm["snr"] >= params.min_snr)
                  & (qm["isi_violations_ratio"] <= params.max_isi_viol)].index.to_list()

        logger.info("%s: keeping %d of %d units (snr >= %g, isi <= %g)",
                    data.name, len(keep), len(qm),
                    params.min_snr, params.max_isi_viol)

        curated = data.analyzer.select_units(keep)

        if save_path.exists():
            shutil.rmtree(save_path)
        curated.save_as(format="binary_folder", folder=save_path)
        return si.load_sorting_analyzer(save_path)

    if force or not save_path.is_dir():
        analyzer = cb()
    else:
        try:
            logger.info("Loading cached curation from %s", save_path)
            analyzer = si.load_sorting_analyzer(save_path)
        except Exception:
            logger.exception("Failed to load cached curation from %s", save_path)
            analyzer = cb()

    logger.info("Curation complete: %d -> %d units",
                data.analyzer.get_num_units(), analyzer.get_num_units())

    return replace(data, analyzer=analyzer, sorting=analyzer.sorting)



def save_params(params: Params, path: Path) -> None:
    logger.info("Saving parameters to %s", path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=4)
        

def load_params(path: Path) -> Params:
    if not path.is_file():
        logger.info("No parameters at %s, using defaults", path)
        return Params()

    try:
        with open(path, "r") as f:
            return Params(**json.load(f))
    except Exception:
        logger.exception("Failed loading parameters from %s, using defaults", path)
        return Params()


def register(project: Project, data: Data, raw_path: Path) -> None:

    reg_path = project.dir / "recordings.json"
    reg = registered(project)

    reg[data.name] = {
        "path": str(raw_path.resolve()),
        "project_title": data.project_title,
        "tag": data.tag,
    }

    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=4)

    logger.info("Registered %s", data.name)


def registered(project: Project) -> dict:
    reg_path = project.dir / "recordings.json"
    if not reg_path.is_file():
        return {}

    try:
        with open(reg_path, "r") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed reading registry at %s", reg_path)
        return {}


def result_dir(project: Project, name: str) -> Path:
    return project.dir / "results" / name

# private helpers

def _configure_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _read_md(path: Path) -> dict:

    cp = configparser.RawConfigParser()
    cp.optionxform = str

    if not cp.read(path):
        logger.warning("No metadata at %s", path)
        return {"project_title": "", "tag": "", "chipid": ""}

    metadata = {}
    for sec in cp.sections():
        metadata.update(cp[sec])

    return {k: metadata.get(k, "") for k in ("project_title", "tag", "chipid")}


    




