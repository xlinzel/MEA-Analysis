# MEA Analysis Software — Build Plan

Python software for MaxWell HD-MEA data analysis. Core deliverables: active-electrode
analysis, spike sorting, and spatial overlays on a static plate image.

**Stack:** SpikeInterface (IO, preprocessing, detection, sorting, QC) · Elephant
(spike-train statistics) · pandas/pyarrow (tables) · matplotlib (figures) ·
imgui-bundle (GUI) · Typer (CLI)

---

## 1. Storage tree

```
project/                            # a "project" = one lab study, user picks the folder
├── recordings.csv                  # manifest: path, well, condition, div, image
│
├── raw/                            # ← the .raw.h5 files live here
│   ├── div07_wellA.raw.h5
│   ├── div14_wellA.raw.h5
│   └── div21_wellA.raw.h5
│
├── images/
│   ├── div14_plate.png             # photo of array + sample
│   └── div14_plate.reg.json        # electrode µm → pixel transform
│
├── results/
│   └── div14_wellA/                # one folder per raw file, named by stem
│       ├── params.json             # exactly what was run
│       ├── preprocessed/           # SI binary folder (large, deletable)
│       ├── peaks.npy
│       ├── sorting/
│       ├── analyzer/
│       ├── electrodes.parquet
│       ├── units.parquet
│       └── figures/
│           ├── overlay.png
│           ├── activity_map.png
│           └── raster.png
│
└── exports/
    └── all_metrics.parquet         # every recording concatenated
```

### Storage rules

- **`raw/` is read-only.** Nothing in the codebase ever writes there. All signal data
  stays inside the `.raw.h5` files and is read lazily — you never load a full recording
  into memory.
- **One results folder per raw file**, named by the file stem. This is the only mapping
  between input and output, and it's computed by a single function so it can't drift.
- **Registration lives with the image**, not in `results/`. One plate photo typically
  serves every recording from that DIV, so tying the transform to a recording would
  duplicate it.
- **`preprocessed/` is the only large derived artifact** and is safe to delete.
  Everything downstream reads `peaks.npy`, `analyzer/`, and the parquet files.
- **`params.json` records what actually ran**, written at the end of a successful run.
  When a result looks wrong six weeks later, this is what tells you why.

---

## 2. What `recordings.csv` is for

It is an **input manifest**, not a metrics store. It answers two questions the `.h5`
files can't:

1. **Which raw files belong to this project** — an explicit index, so the software
   never has to glob a directory and guess.
2. **The experimental metadata that isn't in the recording** — well, condition, DIV,
   and which plate image goes with it. The `.h5` knows sample rate and electrode
   geometry; it does not know that well A is the knockout at DIV 14.

| Column      | Meaning                                          |
|-------------|--------------------------------------------------|
| `path`      | Filename relative to `raw/`                       |
| `well`      | Well identifier                                   |
| `condition` | Genotype / treatment / control                    |
| `div`       | Days in vitro (integer)                           |
| `image`     | Filename relative to `images/`, or blank          |
| `notes`     | Free text, ignored by code                        |

CSV specifically because lab members can edit it in Excel, it diffs cleanly in git, and
it needs no database. The GUI reads and writes it as a table; hand-editing stays valid.

**Metrics never go here.** Computed output goes to parquet (`electrodes.parquet`,
`units.parquet`, `all_metrics.parquet`) — typed, columnar, fast, and it preserves
dtypes across a save/load round trip in a way CSV does not.

---

## 3. Code tree

```
mea-analysis/
├── pyproject.toml
├── README.md
└── meakit/
    ├── __init__.py
    ├── core.py
    ├── metrics.py
    ├── cli.py
    ├── plots.py
    └── gui.py
```

### Module connections

```
        core.py          (SpikeInterface only — no pandas, no matplotlib)
           │
           ├──────────► metrics.py     (SI objects → pandas DataFrames)
           │                 │
        plots.py             │         (DataFrames → matplotlib Figures)
           │                 │
           └────────┬────────┘
                    │
             ┌──────┴──────┐
          cli.py        gui.py         (clients — orchestration only)
```

**Import rules, enforced by hand:**

- `core.py` imports nothing from the package. It is the bottom of the stack.
- `metrics.py` imports `core` only for type annotations and the results-path helper.
- `plots.py` imports nothing from the package. It takes DataFrames and arrays, so it's
  testable with synthetic data and never triggers a recording load.
- `cli.py` and `gui.py` import the other three. **Nothing imports them.**
- Neither `cli.py` nor `gui.py` contains analysis logic. If you delete `gui.py` the
  software still works fully from the command line.

### Data flow

```
raw/*.raw.h5 ──core.load──► Recording ──core.preprocess──► preprocessed/
                                │
                                ├──core.detect────► peaks.npy
                                │                      │
                                │                      ├─metrics.active_electrodes─► electrodes.parquet
                                │                      │
                                └──core.sort──► sorting/ ──core.analyze──► analyzer/
                                                                              │
                                                                  metrics.unit_stats ──► units.parquet
                                                                              │
recordings.csv ──────────────────────────────► metrics.tidy ◄─────────────────┘
                                                    │
                                            exports/all_metrics.parquet
                                                    │
images/*.png + *.reg.json ──────────► plots.* ──► figures/*.png
```

### Convention applied throughout

**Compute functions return; write functions write. Never both.** A function that
returns a DataFrame does not touch disk. This keeps everything unit-testable and lets
the CLI and GUI decide where output lands.

**Every stage is idempotent.** Each checks whether its output folder already exists and
returns early unless `force=True`. Re-running a Kilosort4 sort by accident costs hours.

---

## 4. `core.py` — write first

**Role:** the pipeline. Thin wrappers over SpikeInterface, one function per stage.

**Depends on:** nothing internal.
**Writes to disk:** `results/<stem>/preprocessed/`, `peaks.npy`, `sorting/`,
`analyzer/`, `params.json`.

| Function | Returns | Writes |
|---|---|---|
| `Params` (frozen dataclass) | — | — |
| `read_manifest(project_root)` | DataFrame of `recordings.csv`, paths resolved to absolute | — |
| `result_dir(raw_path, project_root)` | `Path` to `results/<stem>/`, created if absent | creates folder |
| `load(raw_path)` | `BaseRecording`, probe attached, lazy | — |
| `preprocess(rec, out_dir, params)` | `BaseRecording` (filtered, referenced) | `preprocessed/` |
| `detect(rec, out_dir, params)` | structured `np.ndarray` — `sample_index`, `channel_index`, `amplitude` | `peaks.npy` |
| `sort(rec, out_dir, params)` | `BaseSorting` | `sorting/` |
| `analyze(rec, sorting, out_dir)` | `SortingAnalyzer` with extensions computed | `analyzer/` |
| `run_all(raw_path, params, progress)` | `Path` to the results folder | `params.json` + all of the above |

`Params` fields: highpass Hz, lowpass Hz, common-reference flag, detection MAD
threshold, sorter name, job count. Frozen so it can be hashed and compared against a
previous `params.json`.

`run_all` takes a **progress callback** — a function of `(stage_name, fraction)`. This
is the single integration point for both clients: the CLI passes a printer, the GUI
passes something that writes into its state object. `core.py` therefore has no
dependency on Typer, imgui, or threading.

### Order of work inside the file

1. `Params` and `result_dir`. Trivial, but everything else takes them as arguments, so
   settle their shape before writing anything that uses them.
2. `load`. Confirm a real `.raw.h5` opens. MaxWell files need the vendor HDF5
   decompression plugin — SpikeInterface ships an auto-installer, otherwise set
   `HDF5_PLUGIN_PATH` yourself. Get this working before anything else; it's the most
   likely environment problem in the whole project.
3. `read_manifest`. Small, and you need it to find files by anything other than a
   hardcoded path.
4. `preprocess`. Decide deliberately whether common median referencing is on — it
   changes detection results, and the lab may want to match vendor software.
5. `detect`. Threshold detection only, no sorting. This is the fastest path to a real
   number and it unblocks the whole active-electrode deliverable.
6. `run_all` with stages 1–5 wired and the progress callback threaded through.
7. `sort` and `analyze` last. They're slow, GPU-dependent, and not needed for the
   primary deliverable, so don't let them block progress on everything else.

**Done when:** a real `.raw.h5` opens and `get_channel_locations()` returns sane
micrometre coordinates matching the known array pitch.

---

## 5. `metrics.py` — write second

**Role:** convert SpikeInterface objects into pandas tables. The analysis layer.

**Depends on:** `core` (types and the path helper).
**Writes to disk:** `electrodes.parquet`, `units.parquet`, `exports/all_metrics.parquet`.

| Function | Returns | Writes |
|---|---|---|
| `active_electrodes(peaks, rec, params)` | DataFrame, one row per channel | — |
| `unit_stats(analyzer)` | DataFrame, one row per unit | — |
| `tidy(table, meta)` | DataFrame in long format | — |
| `write_tables(result_dir, elec_df, unit_df)` | `None` | both parquets |
| `load_all(project_root)` | DataFrame, all recordings concatenated | — |
| `export(project_root, df)` | `None` | `exports/all_metrics.parquet` |

`active_electrodes` columns: `channel_id`, `x_um`, `y_um`, `n_spikes`, `rate_hz`,
`median_amp_uv`, `presence_ratio`, `active`. Three criteria combine into the boolean —
a rate floor, an amplitude floor, and a presence ratio. The presence ratio is what
rejects an artifact channel that fires 500 times in two seconds and then goes silent;
a bare rate threshold passes it.

`unit_stats` columns: `unit_id`, `peak_channel`, `x_um`, `y_um`, `n_spikes`, `rate_hz`,
`snr`, `isi_violation_ratio`, `presence_ratio`, `amplitude_cutoff`, `cv`, `cv2`. The
first group comes from SpikeInterface quality metrics, the last from Elephant. The
handoff between the two libraries is a `neo.SpikeTrain` built from the sorting's spike
times — that's the whole integration.

`tidy` produces long format: `recording, well, condition, div, channel_id, unit_id,
metric, value`. Everything downstream — group statistics, comparison plots, export —
reads only this shape. It costs a little effort now and saves a great deal of array
reshaping later.

### Order of work inside the file

1. `active_electrodes`, with the rate criterion only. Get a number out of a real
   recording and sanity-check it against what the lab expects.
2. Add amplitude and presence-ratio criteria. Compare the resulting counts against
   vendor software output on the same recording — this is the validation step that
   makes the lab trust the tool.
3. `write_tables` and `load_all`. Now the CLI has something to call.
4. `tidy`. Design the column set once and don't revise it casually; plots and exports
   both bind to it.
5. `unit_stats` last, after sorting works in `core.py`.

**Done when:** you can produce an active-electrode count from a real recording with no
sorting involved, and defend it against the vendor number.

---

## 6. `cli.py` — write third, before the GUI

**Role:** batch entry point and the primary interface during development.

**Depends on:** `core`, `metrics`, `plots`.
**Writes to disk:** nothing directly — it calls the write functions in the other modules.

| Command | Does |
|---|---|
| `run <recording>` | One recording end to end. `--force` re-runs completed stages, `--sort` includes sorting. |
| `batch` | Every row in the manifest, or a filtered subset. |
| `export` | Concatenate all results into `exports/all_metrics.parquet`. |
| `figures <recording>` | Regenerate figures without re-running analysis. |

Writing this before `gui.py` matters for three reasons: it makes every stage testable
without clicking, it's what you'll actually use for overnight batches, and it forces
the library/UI separation to be real rather than aspirational. If the CLI can do
everything, the GUI is guaranteed to be a thin shell.

### Order of work inside the file

1. `run` for a single hardcoded recording. Crude is fine; you want the round trip.
2. Manifest loading and `batch`, with a filter expression so you can target one DIV.
3. `export`.
4. `figures` once `plots.py` exists.

**Done when:** one command takes a raw file to `electrodes.parquet`.

---

## 7. `plots.py` — write fourth

**Role:** all matplotlib, plus the small amount of geometry only plotting needs.

**Depends on:** nothing internal — it takes DataFrames and arrays.
**Writes to disk:** `images/<name>.reg.json` (registration only). Figures are returned,
not saved; the caller decides the path.

| Function | Returns | Writes |
|---|---|---|
| `apply_style()` | `None` — sets rcParams | — |
| `fit_transform(src_um, dst_px)` | 3×3 `np.ndarray` affine | — |
| `apply_transform(T, pts)` | `np.ndarray` (N, 2) in pixels | — |
| `save_registration(image_path, T)` | `None` | `<image>.reg.json` |
| `load_registration(image_path)` | 3×3 `np.ndarray` | — |
| `activity_map(elec_df)` | `Figure` | — |
| `overlay(image, elec_df, T, metric)` | `Figure` | — |
| `raster(sorting, t0, t1)` | `Figure` | — |
| `qc_panel(unit_df)` | `Figure` | — |

`fit_transform` is a least-squares solve for six affine parameters — three
correspondences minimum, four for stability. A square-on plate photo needs nothing
more, so no OpenCV dependency. Only escalate to a projective transform if you see real
perspective distortion in the images.

`apply_style` is called once at import. Font, sizes, DPI, and default save format live
there and nowhere else. It's the difference between figures that look like one coherent
paper and figures that look like six people made them.

### Order of work inside the file

1. `apply_style`. Thirty seconds, and it applies retroactively to everything you build
   after it.
2. `activity_map`. No image, no transform, no registration — just electrode positions
   coloured by rate. It validates your geometry independently of the overlay, so when
   the overlay looks wrong later you'll know which half is at fault.
3. `fit_transform` and `apply_transform`, tested against synthetic points where you know
   the answer before you point them at a real photo.
4. `save_registration` / `load_registration`.
5. `overlay`. Now it's `activity_map` drawn on top of an image through a transform you
   already trust.
6. `raster` and `qc_panel` after sorting works.

**Performance note:** rasters with hundreds of thousands of spikes will stall matplotlib
if drawn with `scatter`. Use `eventplot` or a `LineCollection`, and rasterize the axes
while keeping text vector so exported PDFs stay small and editable.

**Done when:** `overlay()` produces a publication-ready PNG from a saved `.reg.json`.

---

## 8. `gui.py` — write last

**Role:** imgui-bundle desktop app. Orchestration and display only.

**Depends on:** `core`, `metrics`, `plots`.
**Writes to disk:** only through the other modules, plus `recordings.csv` when the user
edits the manifest table.

Immediate mode means the frame function runs at 60 Hz and widgets hold no state of their
own. That single fact determines the file's shape.

### State

One `AppState` dataclass holds everything mutable: the manifest DataFrame, the set of
selected row indices, the current `Params`, a running flag, current stage name, progress
fraction, log lines, cached figure textures, and a dirty flag per cached figure. Because
imgui keeps nothing itself, this object *is* the application. Get its fields right and
the rest follows.

### Frame sections

| Section | Contents |
|---|---|
| Manifest table | Sortable, multi-select rows from `recordings.csv`; editable condition/DIV/image cells |
| Parameters | Widgets bound directly to `state.params` fields |
| Run controls | Submit button, progress bar, cancel |
| Log | Scrolling text from the worker |
| Overlay view | Plate image with electrodes drawn on top |

### Three constraints that shape the implementation

**Never block the frame.** Run `core.run_all` in a `threading.Thread` and let the
progress callback write into `AppState`. Python threads are fine here — SpikeInterface
and NumPy release the GIL during the heavy work. Push log lines through a `queue.Queue`
rather than mutating a list across threads.

**Cache figure textures.** Uploading a matplotlib Figure to the GPU every frame will
drop you to single-digit framerates. Re-render only when the corresponding dirty flag
flips — on new results, on a parameter change, on a new selection.

**Draw the overlay with the ImGui draw list, not matplotlib.** Display the plate image,
then draw filled circles at transformed electrode coordinates. You get hover tooltips
and click-to-pick fiducials nearly free, since fiducial picking is just a mouse position
converted into image coordinates. Matplotlib stays for the exported PNG, so the figure
that goes in a paper and the thing on screen are produced by different code paths — a
worthwhile trade for interaction quality.

### Order of work inside the file

1. `AppState` and an empty window that runs. Confirm the event loop and `hello_imgui`
   app runner work on your setup before adding anything.
2. Manifest table, read-only. Load `recordings.csv`, display it, make rows selectable.
3. Parameter widgets bound to `state.params`.
4. Run button calling `core.run_all` **synchronously**, blocking the UI. Ugly, but it
   proves the wiring end to end with far less to debug.
5. Move it to a thread; add the progress callback, progress bar, and log pane.
6. Figure display with texture caching — start with `activity_map`.
7. Interactive overlay with the draw list.
8. Manifest editing and write-back to CSV.
9. Cancellation and error surfacing. Left until last because a crash in a worker thread
   is easy to debug from the console during development, and error handling is fiddly to
   write before you know what fails.

Use `hello_imgui` as the app runner — docking layout and window geometry persist between
sessions for free. `portable_file_dialogs` gives you native file pickers for selecting
the project folder.

**Done when:** a lab member can pick a project folder, select recordings, set thresholds,
run, and see the overlay without touching a terminal.

---

## 9. Suggested milestones

| # | Deliverable | Spans |
|---|---|---|
| 1 | A real `.raw.h5` opens, geometry verified | `core.load` |
| 2 | Active-electrode count from the command line | `core.detect`, `metrics.active_electrodes`, `cli.run` |
| 3 | Activity map and overlay PNGs | `plots.*` |
| 4 | Batch processing across a manifest | `cli.batch`, `cli.export` |
| 5 | GUI wrapping all of the above | `gui.py` |
| 6 | Sorting and unit statistics | `core.sort`, `core.analyze`, `metrics.unit_stats` |

Milestones 1–3 are demonstrable quickly and cover the lab's primary ask. Sorting is
deliberately last — it's the slowest, the most GPU-dependent, and nothing else waits on
it.