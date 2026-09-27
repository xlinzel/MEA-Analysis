"""Manual testing entry point."""

from __future__ import annotations

from pathlib import Path
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

from meakit.core import (
    Params, Project, configure_runtime, _configure_logging,
    load, preprocess, sort, analyze, compute, merge, concatenate, split,
    curate_together, result_dir
)
from meakit import metrics
from meakit import plots



# recordings of one slice are sorted together so units keep their id across conditions
SLICES = {
    "Lumbar_S2": ("000005", "000007", "000008"),
    "Thoracic_S3": ("000012", "000013", "000014"),
    "Lumbar_S1": ("000015", "000018", "000019"),
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


if __name__ == "__main__":
    freeze_support()
    _configure_logging()

    project = Project(Path("tests/project"))
    # per-sorter settings live in core.SORTER_DEFAULTS; override with Params(sorter_params=...)
    # SC2 is the validated choice; "herdingspikes" and "kilosort4" (GPU) also run
    params_dict = tuple(Params(sorter=s) for s in ("spykingcircus2",))

    configure_runtime()

    for params in params_dict:
        for slice_name, names in SLICES.items():
            datas = []
            for name in names:
                data = load(Path(f"tests/project/data/{name}/data.raw.h5"))
                datas.append(preprocess(data, params, project, save=True, force=True))

            joint = concatenate(datas, slice_name)
            joint = sort(joint, params, project, force=True)
            joint = analyze(joint, params, project, force=True)
            joint = compute(joint, params)
            joint = merge(joint, params, project, force=True)

            parts = split(joint, params, project, force=True)
            parts = curate_together(parts, params, project, force=True)

            # electrodes seen in each recording, so a map can show those gone silent
            seen = [{c for cs in metrics.unit_channels(d).values() for c in cs} for d in parts]
            for data, here in zip(parts, seen):
                figures(data, params, project, set().union(*seen) - here, band="range")
