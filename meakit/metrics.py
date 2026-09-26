"""

"""

from __future__ import annotations

# stdlib
import logging
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import neo
import quantities as pq
from elephant.statistics import cv, cv2, isi
import spikeinterface.full as si

# local
from meakit.core import Data, Params, Project, result_dir

# logger
logger = logging.getLogger(__name__)


# public functions

def unit_stats(data: Data) -> pd.DataFrame:
    logger.info("Obtaining unit statistics for %s", data.name)

    an = data.analyzer
    if an is None:
        raise ValueError(f"{data.name}: no analyzer, run analyze() first")

    frames = []
    for name in ("quality_metrics", "template_metrics"):
        if an.has_extension(name):
            frames.append(an.get_extension(name).get_data())
        else:
            logger.warning("Extension %s not computed, skipping", name)

    if not frames:
        raise ValueError(f"{data.name}: no metric extensions computed")

    df = pd.concat(frames, axis=1)
    df.index.name = "unit_id"
    df = df.reset_index()

    if an.has_extension("unit_locations"):
        loc = an.get_extension("unit_locations").get_data()
        df["x_um"] = loc[:, 0]
        df["y_um"] = loc[:, 1]

    ext = si.get_template_extremum_channel(an)
    df["peak_channel"] = df["unit_id"].map(ext)

    dur = data.recording.get_total_duration()
    trains = _to_spiketrains(data.sorting, dur)
    df["cv"] = [float(cv(isi(t))) for t in trains]
    df["cv2"] = [float(cv2(isi(t))) for t in trains]

    return df


def unit_spike_times(data: Data) -> list[np.ndarray]:
    """Spike times in seconds, one array per unit, in sorting.unit_ids order."""
    return [
        data.sorting.get_unit_spike_train(uid, return_times=True)
        for uid in data.sorting.unit_ids
    ]


def unit_channels(data: Data) -> dict:
    """Unit id -> the channel ids that unit appears on."""
    an = data.analyzer
    if an is None or an.sparsity is None:
        logger.warning("%s: no sparsity, falling back to peak channel", data.name)
        ext = si.get_template_extremum_channel(an)
        return {uid: [ch] for uid, ch in ext.items()}

    return an.sparsity.unit_id_to_channel_ids



def rate_over_time(data: Data, bin_s: float = 10.0) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean firing rate across units, binned over the recording.

    One row per bin: t_s, mean_hz, sem_hz, n_units.
    """
    logger.info("Binning unit firing rate for %s at %.0f s", data.name, bin_s)

    dur = data.recording.get_total_duration()
    times = unit_spike_times(data)

    if not times:
        logger.warning("%s: no units", data.name)
        return (pd.DataFrame(columns=["t_s", "mean_hz", "sem_hz",
                                      "total_hz", "n_units"]),
                np.empty((0, 0)))

    edges = np.arange(0, dur + bin_s, bin_s)
    if edges[-1] > dur:
        edges = edges[:-1]

    per_unit = np.array([np.histogram(t, bins=edges)[0] / bin_s for t in times])
    n = per_unit.shape[0]

    summary = pd.DataFrame({
        "t_s": edges[:-1] + bin_s / 2,
        "mean_hz": per_unit.mean(axis=0),
        "sem_hz": per_unit.std(axis=0) / np.sqrt(n),
        "total_hz": per_unit.sum(axis=0),
        "n_units": n,
    })
    return summary, per_unit


def trace_segment(data: Data, units: pd.DataFrame, n_channels: int = 8,
                  seconds: float = 1.0, start_s: float | None = None,
                  channel_ids: list | None = None
                  ) -> tuple[np.ndarray, float, float]:
    """Raw traces in µV from the peak channels of the busiest units.

    start_s picks the window; None finds the one with the most unit spikes.
    Returns (traces, start_s, fs), traces shaped (n_samples, n_channels).
    """
    fs = data.recording.get_sampling_frequency()
    dur = data.recording.get_total_duration()
    all_ids = list(data.recording.channel_ids)

    if channel_ids is None:
        top = units.sort_values("firing_rate", ascending=False)
        chans = unit_channels(data)
        channel_ids = []
        for uid in top["unit_id"]:
            for c in chans.get(uid, []):
                if c not in channel_ids:
                    channel_ids.append(c)
        channel_ids = channel_ids[:n_channels]
        n_unit = len(channel_ids)

        if n_unit < n_channels and data.peaks is not None:
            counts = np.bincount(data.peaks["channel_index"],
                                 minlength=len(all_ids))
            for i in np.argsort(counts)[::-1]:
                if len(channel_ids) >= n_channels:
                    break
                if all_ids[i] not in channel_ids:
                    channel_ids.append(all_ids[i])
    else:
        n_unit = len(channel_ids)

    if start_s is None:
        uids = list(data.sorting.unit_ids)
        times = unit_spike_times(data)
        on_shown = [
            times[uids.index(row.unit_id)]
            for row in units.itertuples()
            if row.peak_channel in channel_ids
        ]
        start_s = _best_window(on_shown, dur, seconds) if on_shown else dur / 2

    start_s = float(np.clip(start_s, 0.0, max(dur - seconds, 0.0)))

    seg = data.recording.get_traces(
        start_frame=int(start_s * fs),
        end_frame=int((start_s + seconds) * fs),
        channel_ids=channel_ids,
    )

    logger.info("Traces: %d ch, %.2f s from t=%.2f s",
                seg.shape[1], seconds, start_s)

    return seg, start_s, fs, n_unit


# loading and saving

def write_tables(data: Data, project: Project, units: pd.DataFrame) -> None:
    out = result_dir(project, data.name)
    out.mkdir(parents=True, exist_ok=True)

    units.to_parquet(out / "units.parquet", index=False)
    logger.info("Wrote %d unit rows to %s", len(units), out)



def load_tables(data: Data, project: Project) -> pd.DataFrame | None:
    u_path = result_dir(project, data.name) / "units.parquet"
    if u_path.is_file():
        return pd.read_parquet(u_path)
    return None



# private helper functions

def _to_spiketrains(sorting, duration: float) -> list[neo.SpikeTrain]:
    return [
        neo.SpikeTrain(
            sorting.get_unit_spike_train(uid, return_times=True) * pq.s,
            t_stop=duration * pq.s,
        )
        for uid in sorting.unit_ids
    ]


def _best_window(times_subset: list[np.ndarray], duration: float,
                 seconds: float, step_s: float = 0.01) -> float:
    """Start time whose window holds the most spikes across the given channels."""
    nonempty = [t for t in times_subset if len(t)]
    if not nonempty:
        return max(duration / 2 - seconds / 2, 0.0)

    all_t = np.sort(np.concatenate(nonempty))
    latest = max(duration - seconds, 0.0)
    if latest <= 0:
        return 0.0

    starts = np.arange(0.0, latest, step_s)
    counts = (np.searchsorted(all_t, starts + seconds)
              - np.searchsorted(all_t, starts))

    return float(starts[counts.argmax()])
    