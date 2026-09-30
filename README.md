# MEA Analysis (meakit)

Spike sorting and analysis for MaxWell HD-MEA recordings, built on SpikeInterface.
The recordings of one slice (e.g. Baseline → E2 → TTX) are sorted **together**, so each
cell keeps the same unit ID across conditions and you can follow it appearing, changing
or dying.

Pipeline per slice: preprocess each recording → join them → sort → merge duplicate and
split units → split back per recording → curate → figures.

## Install

```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -e .                  # add [kilosort] for Kilosort 4 (needs a GPU)
```

## Data

One folder per project; each recording in its own folder with MaxWell's metadata file:

```
<project>/data/000012/data.raw.h5
<project>/data/000012/mxassay.metadata     # its "tag" is used to title figures
<project>/results/                         # created by the pipeline
```

## Running `main.py`

Edit the top of `main.py`, then run `python main.py`.

**`PROJECTS`**: which project folders to run, and in each, which recordings to sort
together as one slice. Every project and slice listed is run in turn.
```python
PROJECTS = {
    "D:/MEA/2026-09-12_MSDR_Thoracic": {
        "Thoracic_S3": ("000012", "000013", "000014"),   # Baseline, E2, TTX, in order
    },
    "D:/MEA/2026-09-04_MMarm_Lumbar": {
        "Lumbar_S1": ("000002", "000003", "000004"),
    },
}
```
- Only group recordings of the **same slice**. Recordings are matched by electrode
  position, and only electrodes recorded in all of them are kept. The log warns
  `only N of M electrodes are shared`; if N is small, a recording is from another slice
  or another electrode configuration.
- If the recordings share no electrodes at all, the slice stops with an error. Sort
  them as separate slices.
- The order sets the colours and legend order in the comparison figure.

**`MAKE_VIDEOS`**: `True` makes an electrode timelapse video per recording
(about 10 min each to render). Default `False`.

**`RATE_WINDOW_S`**: sliding window for all firing-rate plots, in seconds (default 40).
Shorter shows faster changes but is noisier; `None` picks a window from the firing rates.

**`force=`** on each stage: `True` redoes the stage, `False` reuses saved results from
`<project>/results/`. After changing a parameter, set `force=True` from that
stage onwards:

| You changed | Redo from |
|---|---|
| filter / reference / `exclude_channels` | `preprocess` (everything after) |
| sorter or `sorter_params` | `sort` |
| `min_spike_snr`, `duplicate_fraction`, `merge_min_spikes` | `merge` |
| curation thresholds (`min_snr`, `min_rate_hz`, …) | `curate_together` |
| figure options (`band`, `RATE_WINDOW_S`, colours) | nothing: figures are always redrawn |

**`band=`** in `figures(...)` and `comparison(...)`: shading around the mean firing
rate: `"range"` (min–max), `"iqr"`, `"sem"` or `"none"`.

**Sorter**: `Params(sorter="spykingcircus2")` (default, validated). `"kilosort4"` and
`"herdingspikes"` also run but are less tested.

## Outputs (`<project>/results/`)

| Where | What |
|---|---|
| `<recording>/figures/` | summary (map, traces, mean rate), rates, waveforms, diagnostics; `_electrodes.mp4` if videos are on |
| `<recording>/analyzer_<sorter>_curation.csv` | every unit's metrics; `keep` / `rejected_by` (this recording), `keep_joint` (kept for the slice), `zeroed_as_noise` (only noise here, counted as 0 Hz) |
| `<recording>/units.parquet` | table behind the figures |
| `<slice>/figures/<slice>_<sorter>_comparison.png` | all recordings of the slice: electrode map, mean rates overlaid, per-unit rates |

**Curation rule:** a unit passing the rules in *any* recording is kept in *all* of them.
Every recording therefore averages over the same units. Where a unit's spikes in a
recording are only noise (fails footprint or refractory at a real rate, e.g. a cell
silenced by TTX), they are removed and it counts as 0 Hz (× in the comparison figure).

## Parameters (`Params(...)` in `main.py`)

Pass any of these, e.g. `Params(sorter="spykingcircus2", min_snr=5.0)`.

| Parameter | Default | What it does |
|---|---|---|
| `freq_min`, `freq_max` | 300, 6000 Hz | band-pass filter |
| `common_reference` | `"median"` | global reference subtracted from every electrode |
| `exclude_channels` | `()` | readout channel IDs to drop (e.g. a known artifact channel) |
| `detect_mad` | 5.0 | spike threshold (× noise) for the per-electrode detection used by the videos |
| `sorter_params` | `{}` | passed straight to the sorter, overriding its defaults |
| `min_spike_snr` | 2.0 | drops single spikes smaller than this × noise on the unit's peak electrode (template fits to noise) |
| `duplicate_fraction` | 0.5 | two units sharing this fraction of spikes (±0.4 ms) are merged as one cell |
| `merge_min_spikes` | 30 | units need this many spikes to be considered for auto-merging |
| `min_snr` | 4.5 | curation: template peak ÷ noise |
| `min_rate_hz` | 0.05 | curation: minimum firing rate (15 spikes in 300 s) |
| `min_presence` | 0.5 | curation: fraction of 60 s bins with spikes |
| `max_amp_cutoff` | 0.1 | curation: estimated fraction of spikes lost below threshold |
| `fast_rate_hz` | 2.0 | above this rate, refractory is judged by contamination; below, by violation count |
| `max_rp_contamination` | 0.2 | curation (fast units): refractory contamination |
| `max_isi_count_slow` | 1 | curation (slow units): spikes closer than the refractory period |
| `sparsity_radius_um` | 60 µm | electrodes around each unit used for its waveform and metrics |

## If things look wrong

| Symptom | Try |
|---|---|
| Sorting crashes with `Unable to allocate … GiB` | `sorter_params={"job_kwargs": {"chunk_duration": "100ms"}}` |
| `Recording too large to be preloaded in RAM` | harmless, just slower. With ≥ 64 GB RAM: `sorter_params={"cache_preprocessing": {"mode": "memory", "memory_limit": 0.75}}` |
| No units pass curation | open the `curation.csv`s and read `rejected_by`. Mostly `snr` + `footprint` + `refractory` means noise, not a missed cell (check the electrode video) |
| A real-looking cell is rejected on `snr` only | lower `min_snr` (e.g. 4.0); check it drops in TTX |
| Noise units kept (flat across TTX, one electrode, very high rate) | raise `min_snr`; add the channel to `exclude_channels` if it's the same readout channel each time |
| Pile of spike amplitudes near 0 µV | raise `min_spike_snr` (e.g. 2.5–3) |
| One cell appears as two units | lower `duplicate_fraction` (e.g. 0.3) |
| Two cells merged into one | raise `duplicate_fraction` |
| Cell active in only one condition is missing | lower SC2's own cutoff, which applies to the joined recording: `sorter_params={"min_firing_rate": 0.02}` |
| Few shared electrodes / error joining recordings | recordings are from different slices or electrode configurations; split them in `PROJECTS` |
| A failed sort is found on the next run (`Failed to load cached sorting`) | harmless, it re-sorts; or delete `results/<slice>/sorted_<sorter>` |
