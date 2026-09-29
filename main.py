"""Manual testing entry point."""

from __future__ import annotations

from pathlib import Path
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

from meakit.core import (
    Params, Project, configure_runtime, _configure_logging,
    load, preprocess, detect, sort, analyze, compute, merge, concatenate, split,
    curate_together, result_dir
)
from meakit import metrics
from meakit import plots



MAKE_VIDEOS = False   # electrode timelapse per recording (~10 min each to render)

# recordings of one slice are sorted together so units keep their id across conditions;
# they must share electrodes (concatenate raises if they don't)
SLICES = {
    #"MMarm_Lumbar_S1": ("000002", "000003", "000004"),
    "FMarm_SNI_S1": ("000004", "000008", "000011", "000012"),
    #"Thoracic_S3": ("000012", "000013", "000014"),
    #"Lumbar_S1": ("000015", "000018", "000019"),
}


def figures(data, params, project, silent_chans=(), band="range"):
    """band: spread shaded around the mean rate in the summary, "range" (min-max),
    "iqr" (interquartile), "sem" (± standard error) or "none"."""
    units = metrics.unit_stats(data)
    chans = metrics.unit_channels(data)

    seg, t0, fs, n_unit = metrics.trace_segment(data, units, seconds=1.0)
    zseg, _, _, _ = metrics.trace_segment(data, units, seconds=0.1, start_s=t0 + 0.45)

    metrics.write_tables(data, project, units)

    figdir = result_dir(project, data.name) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    stem = f"{data.tag.replace(' ', '_') if data.tag else data.name}_{params.sorter}"

    t_mid, per_unit, bin_s = metrics.rate_over_time(data)
    templates = data.analyzer.get_extension("templates").get_data()
    figs = {
        "summary": plots.summary(units, data.locations, data.recording.channel_ids,
                                 chans, seg, zseg, fs, params, data.name, data.tag, n_unit,
                                 t_mid, per_unit, bin_s, band=band,
                                 silent_chans=silent_chans),
        "rates": plots.rates(t_mid, per_unit, bin_s, metrics.unit_spike_times(data),
                             data.recording.get_total_duration(), data.name),
        "waveforms": plots.waveforms(templates, list(data.sorting.unit_ids), units, fs),
        "diagnostics": plots.diagnostics(metrics.isi_amplitudes(data)),
    }
    for kind, fig in figs.items():
        fig.savefig(figdir / f"{stem}_{kind}.png")
        plt.close(fig)


def comparison(parts, slice_name, params, project, band="sem"):
    """All recordings of a slice in one figure, rates on a shared window."""
    window_s = max(metrics.rate_over_time(d)[2] for d in parts)
    recs = []
    for d in parts:
        t, per_unit, _ = metrics.rate_over_time(d, window_s=window_s)
        dur = d.recording.get_total_duration()
        recs.append(dict(label=d.tag or d.name, unit_chans=metrics.unit_channels(d), t=t,
                         per_unit=per_unit, rates={u: len(s) / dur for u, s in
                                                   zip(d.sorting.unit_ids, metrics.unit_spike_times(d))}))
    fig = plots.comparison(recs, parts[0].locations, parts[0].recording.channel_ids,
                           window_s, band, slice_name)
    figdir = result_dir(project, slice_name) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{slice_name}_{params.sorter}_comparison.png")
    plt.close(fig)


def video(raw, part, params, project):
    """Electrode timelapse of one recording; raw carries the pipeline's detect() peaks."""
    ids = list(part.analyzer.channel_ids)
    on = [ids.index(c) for cs in metrics.unit_channels(part).values() for c in cs]
    rec = raw.recording
    figdir = result_dir(project, raw.name) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    stem = f"{raw.tag.replace(' ', '_') if raw.tag else raw.name}_electrodes.mp4"
    plots.electrode_video(raw.peaks, raw.locations, part.analyzer.get_channel_locations()[on],
                          rec.get_sampling_frequency(), rec.get_total_duration(),
                          figdir / stem, raw.tag or raw.name, params.detect_mad)


if __name__ == "__main__":
    freeze_support()
    _configure_logging()

    project = Project(Path("tests/project"))
    # per-sorter settings live in core.SORTER_DEFAULTS; override with Params(sorter_params=...)
    # SC2 is the validated choice; "herdingspikes" and "kilosort4" (GPU) also run
    params_dict = tuple(Params(sorter=s, sorter_params={"job_kwargs": {"chunk_duration": "100ms"}})
                    for s in ("spykingcircus2",))

    configure_runtime()

    for params in params_dict:
        for slice_name, names in SLICES.items():
            datas = []
            for name in names:
                data = load(Path(f"tests/project/data/{name}/data.raw.h5"))
                datas.append(preprocess(data, params, project, save=True, force=False))

            joint = concatenate(datas, slice_name)
            joint = sort(joint, params, project, force=False)
            joint = analyze(joint, params, project, force=False)
            joint = compute(joint, params, force=False)
            joint = merge(joint, params, project, force=False)

            parts = split(joint, params, project, force=False)
            parts = curate_together(parts, params, project, force=False)

            if MAKE_VIDEOS:
                for raw, part in zip(datas, parts):
                    video(detect(raw, params, project, save=True, force=False), part, params, project)

            # electrodes seen in each recording, so a map can show those gone silent
            seen = [{c for cs in metrics.unit_channels(d).values() for c in cs} for d in parts]
            if not any(d.analyzer.get_num_units() for d in parts):
                print(f"{slice_name}: no units passed curation in any recording, no figures")
                continue
            for data, here in zip(parts, seen):
                figures(data, params, project, set().union(*seen) - here, band="sem")
            comparison(parts, slice_name, params, project, band="sem")
