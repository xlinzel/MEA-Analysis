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
import pandas as pd
import spikeinterface.full as si
from spikeinterface.curation.curation_tools import resolve_merging_graph
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# logger
logger = logging.getLogger(__name__)

# type alias
ProgressFn = Callable[[str, float], None]

# default analyzer extensions
DEFAULT_EXTENSIONS = {
    "random_spikes": {},
    "waveforms": {"ms_before": 1.5, "ms_after": 2.5},
    "templates": {"ms_before": 1.5, "ms_after": 2.5},
    "noise_levels": {},
    "spike_amplitudes": {"peak_sign": "both"},
    "spike_locations": {},
    "unit_locations": {"method": "monopolar_triangulation"},
    "correlograms": {},
    "template_similarity": {"method": "l1"},   # the metric auto_merge_units uses
    "quality_metrics": {},
    "template_metrics": {},
}

# sorter settings for an already filtered + referenced 2D HD-MEA recording
SORTER_DEFAULTS = {
    "herdingspikes": {"common_reference": "none"},
    "spykingcircus2": {"apply_preprocessing": False},   # keeps its whitening
    "kilosort4": {"do_CAR": False, "do_correction": False},  # no 1D probe drift in a slice
}

# dataclasses
@dataclass(frozen=True)
class Params:
    freq_min: float = 300.0
    freq_max: float = 6000.0
    common_reference: str = "median"
    detect_mad: float = 5.0
    sparsity_radius_um: float = 60.0
    exclude_channels: tuple = ()
    merge_min_spikes: int = 30
    duplicate_fraction: float = 0.5   # shared spikes (±0.4 ms) that mark a duplicate
    min_spike_snr: float = 2.0   # spikes below this x noise on the unit's peak electrode are template fits to noise
    # curation
    min_snr: float = 4.5
    max_amp_cutoff: float = 0.1
    min_presence: float = 0.5   # slice activity runs down / TTX washes in over minutes
    min_rate_hz: float = 0.05   # a rate, so the cutoff does not depend on recording length
    fast_rate_hz: float = 2.0
    max_rp_contamination: float = 0.2
    max_isi_count_slow: int = 1
    sorter: str = "spykingcircus2"
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
    parts: tuple = ()   # the recordings a concatenated Data was built from (see concatenate)


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

        # drop noisy channels before they enter the median reference
        bad, _labels = si.detect_bad_channels(rec_filt, method="mad")
        drop = sorted(set(bad) | set(params.exclude_channels))
        logger.info("%s: dropping %d channels %s", data.name, len(drop), drop)
        rec_filt = rec_filt.remove_channels(drop)

        rec_ref = si.common_reference(
            rec_filt,
            reference="global", 
            operator=params.common_reference
        )

        if save:
            rec_ref = rec_ref.save_to_folder(
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

    # channels may have been dropped, so take geometry from the preprocessed recording
    return replace(data, recording=rec_ref, locations=rec_ref.get_channel_locations())

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
        extra = SORTER_DEFAULTS.get(params.sorter, {})
        if params.sorter == "kilosort4":
            # template grid at the electrode pitch, x centres every ~300 µm (2D array)
            x_span = np.ptp(data.recording.get_channel_locations()[:, 0])
            extra = extra | {"dmin": data.pitch_um, "dminx": data.pitch_um,
                             "x_centers": max(1, round(x_span / 300))}
        merged = si.get_default_sorter_params(params.sorter) | extra | params.sorter_params
        logger.info("Sorter params: %s", merged)
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

    if force or not save_path.is_dir():
        analyzer = _build_analyzer(data.sorting, data.recording, params, save_path)
        
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
            analyzer = _build_analyzer(data.sorting, data.recording, params, save_path)

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
    """Realign spike trains, merge duplicate units, then auto-merge oversplit units."""

    base = result_dir(project, data.name) / f"analyzer_{params.sorter}"

    def mb():
        an = data.analyzer
        # template matching also fits templates onto noise: drop spikes with no signal
        # on the unit's own peak electrode (SI amplitudes and noise levels)
        amps = an.get_extension("spike_amplitudes").get_data(outputs="by_unit", concatenated=True)
        noise = dict(zip(an.channel_ids, an.get_extension("noise_levels").get_data()))
        peak = si.get_template_extremum_channel(an, peak_sign="both")
        trains = {u: an.sorting.get_unit_spike_train(u)[np.abs(amps[u]) >= params.min_spike_snr * noise[peak[u]]]
                  for u in an.unit_ids}
        logger.info("%s: dropped %d of %d spikes below %g x noise", data.name,
                    an.sorting.to_spike_vector().size - sum(map(len, trains.values())),
                    an.sorting.to_spike_vector().size, params.min_spike_snr)
        sorting = si.NumpySorting.from_unit_dict(trains, an.sampling_frequency)

        # peak_shift returns (peak - nbefore) but align_sorting subtracts its shift,
        # so the sign must be flipped or misaligned units move further apart (SI 0.104)
        shifts = {u: -s for u, s in
                  si.get_template_extremum_channel_peak_shift(an, peak_sign="both").items()}
        sorting = si.remove_excess_spikes(si.align_sorting(sorting, shifts), data.recording)

        # templates and metrics must be rebuilt on the aligned spike trains
        an = _build_analyzer(sorting, data.recording, params, Path(f"{base}_dedup"))
        an.compute(params.extensions)

        # duplicates: one cell emitted under two templates. Merge rather than drop so
        # spikes only one copy caught are kept; the censor removes the double counts
        pairs = si.find_redundant_units(sorting, delta_time=0.4,
                                        duplicate_threshold=params.duplicate_fraction)
        groups = resolve_merging_graph(sorting, pairs)
        logger.info("%s: duplicate groups %s", data.name, groups)
        if groups:
            an = an.merge_units(groups, merging_mode="hard", censor_ms=0.4,
                                new_id_strategy="take_first")

        # correlogram presets need hundreds of spikes; the cross-contamination test
        # (same template + location, no refractory conflict) works at 50-150
        an, merges, _, _ = si.auto_merge_units(
            an, presets=["x_contaminations"],
            steps_params=[{"num_spikes": {"min_spikes": params.merge_min_spikes}}],
            censor_ms=0.3, merging_mode="hard", extra_outputs=True,
        )
        logger.info("%s: auto-merges %s", data.name, merges)
        return an

    analyzer = _cached_analyzer(Path(f"{base}_merged"), force, mb)

    logger.info("Merge complete: %d -> %d units",
                data.analyzer.get_num_units(), analyzer.get_num_units())

    return replace(data, analyzer=analyzer, sorting=analyzer.sorting)


def curate(data: Data, params: Params, project: Project, force: bool = False) -> Data:
    """Keep units passing every rule; log and save which rules rejected the rest."""

    base = result_dir(project, data.name) / f"analyzer_{params.sorter}"

    def cb():
        units = _curation_table(data.analyzer, params)
        logger.info("%s curation:\n%s", data.name, units[
            ["snr", "firing_rate", "num_spikes", "footprint_n", "keep", "rejected_by"]
        ].sort_values("snr", ascending=False).to_string())
        units.to_csv(f"{base}_curation.csv", index_label="unit_id")

        return data.analyzer.select_units(units.index[units["keep"]].to_list())

    analyzer = _cached_analyzer(Path(f"{base}_curated"), force, cb)

    logger.info("Curation complete: %d -> %d units",
                data.analyzer.get_num_units(), analyzer.get_num_units())

    return replace(data, analyzer=analyzer, sorting=analyzer.sorting)


def concatenate(datas: list[Data], name: str) -> Data:
    """Join preprocessed recordings of one slice end to end so they are sorted as one.

    Recordings are matched by electrode, not channel: a MaxWell channel id is a readout
    channel that each configuration can route to a different electrode. Only electrodes
    present in every recording are kept, and channels are renamed to their electrode id
    so all parts share ids. After sort/analyze/compute/merge on the result, split() cuts
    it back into the original recordings with shared unit ids.
    """
    def electrodes(rec):   # electrode id per channel; position when the file has none
        e = rec.get_property("electrode")
        return ([str(x) for x in e] if e is not None else
                [f"{x:g}_{y:g}" for x, y in rec.get_channel_locations()])

    ids = [dict(zip(electrodes(d.recording), d.recording.channel_ids)) for d in datas]
    common = [e for e in ids[0] if all(e in m for m in ids[1:])]
    if not common:
        raise ValueError(f"{name}: {[d.name for d in datas]} share no electrodes "
                         "(different configurations), so they cannot be sorted together")
    smallest = min(d.recording.get_num_channels() for d in datas)
    if len(common) < 0.5 * smallest:
        logger.warning("%s: only %d of %d electrodes are shared", name, len(common), smallest)

    parts = []
    for d, m in zip(datas, ids):
        rec = d.recording.select_channels([m[e] for e in common]).rename_channels(common)
        parts.append(replace(d, recording=rec, locations=rec.get_channel_locations(), peaks=None))
    rec = si.concatenate_recordings([p.recording for p in parts])
    logger.info("%s: %s concatenated, %d shared electrodes, %.0f s",
                name, [d.name for d in datas], len(common), rec.get_total_duration())
    return replace(parts[0], name=name, tag=" + ".join(d.tag or d.name for d in datas),
                   recording=rec, parts=tuple(parts))


def split(data: Data, params: Params, project: Project, force: bool = True) -> list[Data]:
    """Cut a concatenated sorting back into its recordings, each with its own analyzer."""
    sortings = si.split_sorting(data.sorting, data.recording)
    out = []
    for i, part in enumerate(data.parts):
        p = replace(part, sorting=si.select_segment_sorting(sortings, i))
        out.append(compute(analyze(p, params, project, force), params))
    return out


def curate_together(parts: list[Data], params: Params, project: Project, force: bool = False) -> list[Data]:
    """Curate split recordings as one: a unit passing the rules in any recording is kept in all.

    A cell near a cutoff then cannot appear in one condition and vanish in the next.
    Per recording, a kept unit that fails there is still counted when its rate is below
    min_rate_hz (a silenced cell, so means are not inflated) or it fails only SNR, presence
    or amplitude cutoff (a weaker real cell). It is left out of that recording when it fails
    footprint or refractory at a real rate: that is noise the template absorbed.
    """
    tables = [_curation_table(p.analyzer, params) for p in parts]
    keep = sorted(set().union(*(t.index[t["keep"]] for t in tables)))
    logger.info("Joint curation of %s: keeping %s", [p.name for p in parts], keep)

    out = []
    for p, units in zip(parts, tables):
        base = result_dir(project, p.name) / f"analyzer_{params.sorter}"
        noise = (units["rejected_by"].str.contains("footprint|refractory")
                 & (units["firing_rate"].fillna(0) >= params.min_rate_hz))
        units["keep_joint"] = units.index.isin(keep)
        units["in_recording"] = units["keep_joint"] & ~noise
        units.to_csv(f"{base}_curation.csv", index_label="unit_id")
        ids = units.index[units["in_recording"]].to_list()
        logger.info("%s: units %s, left out as noise here %s", p.name, ids, sorted(set(keep) - set(ids)))
        an = _cached_analyzer(Path(f"{base}_curated"), force, lambda: p.analyzer.select_units(ids))
        out.append(replace(p, analyzer=an, sorting=an.sorting))
    return out


def unit_footprints(an: si.SortingAnalyzer, k: float = 3.0, neighbour_um: float = 35.0) -> pd.DataFrame:
    """Channels above k x noise per unit; flags units seen on one electrode despite routed neighbours."""
    tmpl = an.get_extension("templates").get_data()
    noise = an.get_extension("noise_levels").get_data()
    loc = an.get_channel_locations()

    amp = np.abs(tmpl).max(axis=1)                          # (units, channels)
    pk = amp.argmax(axis=1)
    dist = np.linalg.norm(loc[pk][:, None] - loc[None], axis=2)
    df = pd.DataFrame({
        "footprint_n": (amp > k * noise).sum(axis=1),
        "routed_neighbours": ((dist > 0) & (dist <= neighbour_um)).sum(axis=1),
    }, index=an.unit_ids)
    df["single_electrode"] = (df.footprint_n <= 1) & (df.routed_neighbours >= 4)
    return df



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

def _build_analyzer(sorting, recording, params: Params, folder: Path) -> si.SortingAnalyzer:
    logger.info("Creating sorting analyzer at %s", folder)

    # fixed radius: SNR masks shrink to one channel for weak units, which
    # collapses unit localization onto the electrode
    sp = si.estimate_sparsity(sorting, recording, method="radius",
                              radius_um=params.sparsity_radius_um, peak_sign="both")

    return si.create_sorting_analyzer(sorting=sorting, recording=recording, format="binary_folder",
                                      folder=folder, overwrite=True, sparsity=sp)


def _curation_table(an: si.SortingAnalyzer, params: Params) -> pd.DataFrame:
    """Quality metrics + footprint per unit, with the rule outcome in keep / rejected_by."""
    units = an.get_extension("quality_metrics").get_data().join(unit_footprints(an))
    fast = units["firing_rate"] >= params.fast_rate_hz
    rules = pd.DataFrame({
        "snr": units["snr"] >= params.min_snr,
        "amplitude_cutoff": units["amplitude_cutoff"].isna()
                            | (units["amplitude_cutoff"] <= params.max_amp_cutoff),
        "presence_ratio": units["presence_ratio"] >= params.min_presence,
        "firing_rate": units["firing_rate"] >= params.min_rate_hz,
        # contamination is only measurable for fast units; slow units get a count
        "refractory": np.where(fast, units["rp_contamination"] <= params.max_rp_contamination,
                               units["isi_violations_count"] <= params.max_isi_count_slow),
        "footprint": ~units["single_electrode"],
    })
    units["keep"] = rules.all(axis=1)
    units["rejected_by"] = rules.apply(lambda r: ",".join(r.index[~r.to_numpy(bool)]), axis=1)
    return units


def _cached_analyzer(path: Path, force: bool, build: Callable[[], si.SortingAnalyzer]) -> si.SortingAnalyzer:
    if not force and path.is_dir():
        try:
            logger.info("Loading cached analyzer from %s", path)
            return si.load_sorting_analyzer(path)
        except Exception:
            logger.exception("Failed to load cached analyzer from %s", path)

    analyzer = build()
    if path.exists():
        shutil.rmtree(path)
    return analyzer.save_as(format="binary_folder", folder=path)


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


    




