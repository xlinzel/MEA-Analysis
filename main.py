"""Manual testing entry point."""

from __future__ import annotations

from pathlib import Path
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

from meakit.core import (
    Params, Project, configure_runtime, _configure_logging,
    load, preprocess, detect, sort, analyze, compute, merge, curate, result_dir
)
from meakit import metrics
from meakit import plots



if __name__ == "__main__":
    freeze_support()
    _configure_logging()

    project = Project(Path("tests/project"))
    params_dict = (
        Params(sorter="herdingspikes"),
        # SC2 keeps its whitening but must not re-filter/re-reference our recording
        Params(sorter="spykingcircus2", sorter_params={"apply_preprocessing": False}),
    )

    configure_runtime()

    for params in params_dict:
        for name in ("000005", "000007", "000008", "000012", "000013", "000014", "000015", "000018", "000019",):
            data = load(Path(f"tests/project/data/{name}/data.raw.h5"))

            data = preprocess(data, params, project, save=True, force=True)
            data = detect(data, params, project, save=True, force=True)

            data = sort(data, params, project, force=True)
            data = analyze(data, params, project, force=True)
            data = compute(data, params)
            data = merge(data, params, project, force=True)
            data = curate(data, params, project, force=True)

            units = metrics.unit_stats(data)
            rates, per_unit = metrics.rate_over_time(data)
            chans = metrics.unit_channels(data)

            seg, t0, fs, n_unit = metrics.trace_segment(data, units, seconds=1.0)
            zseg, _, _, _ = metrics.trace_segment(data, units, seconds=0.1,
                                                    start_s=t0 + 0.45)

            metrics.write_tables(data, project, units)

            fig = plots.summary(units, data.locations, data.recording.channel_ids,
                                chans, seg, zseg, fs, rates, params,
                                data.name, data.tag, n_unit, per_unit)

            figdir = result_dir(project, data.name) / "figures"
            figdir.mkdir(parents=True, exist_ok=True)
            stem = data.tag.replace(" ", "_") if data.tag else data.name
            fig.savefig(figdir / f"{stem}_{params.sorter}.png")
            plt.close(fig)

            templates = data.analyzer.get_extension("templates").get_data()
            uids = list(data.sorting.unit_ids)
            fig2 = plots.waveforms(templates, uids, units, fs)
            fig2.savefig(figdir / f"{stem}_waveforms__{params.sorter}.png")
            plt.close(fig2)