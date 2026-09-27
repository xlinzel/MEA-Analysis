"""

"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# local
from meakit.core import Params

logger = logging.getLogger(__name__)

# palette (from the style guide)
PURPLE = "#6a3d9a"
BLUE   = "#20548a"
RED    = "#c03030"
GREY_L = "#dcdcdc"
RED_L = "#f1a3a3"   # electrodes of units active only in other recordings of the slice
GREY_M = "#cfcfcf"
BG     = "#f0f0f0"
CONDITIONS = ["#2d1650", PURPLE, "#a47fd6", "#d4c2ef"]   # purple ramp, one per recording, in order

def apply_style() -> None:

    mpl.rcParams.update({
        # figure
        "figure.figsize": (15, 9),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": False,

        # saving (§19)
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "savefig.facecolor": "white",

        # font hierarchy (§14)
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # panel labels (§3)
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlepad": 10,

        # spines and ticks (§13)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # no colour cycling (§1)
        "axes.prop_cycle": mpl.cycler(color=[PURPLE]),

        # legends (§15)
        "legend.frameon": False,

        # lines (short version, point 5)
        "lines.linewidth": 1.0,
        "lines.solid_capstyle": "butt",

        # misc
        "image.cmap": "viridis",
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def unit_map(units: pd.DataFrame, locations: np.ndarray, channel_ids,
             unit_chans: dict, params: Params, name: str = "",
             ax=None, silent_chans=()) -> Figure:
    """Electrodes in grey; those belonging to a unit in purple, unit centres black.

    silent_chans: electrodes of units seen in other recordings of the slice but not
    this one, drawn light red.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(11, 10))
    fig = ax.figure

    pos = {cid: i for i, cid in enumerate(channel_ids)}
    n_units = len(units)

    ax.scatter(locations[:, 0], locations[:, 1],
               s=5, c=GREY_L, edgecolors="none", zorder=1)

    used = {pos[c] for chans in unit_chans.values() for c in chans if c in pos}
    used = sorted(used)
    silent = sorted({pos[c] for c in silent_chans if c in pos} - set(used))
    ax.scatter(locations[silent, 0], locations[silent, 1],
               s=18, c=RED_L, edgecolors="none", zorder=2, label="Active in other recordings")
    ax.scatter(locations[used, 0], locations[used, 1],
               s=18, c=PURPLE, edgecolors="none", zorder=2, label="Active here (> 3× noise)")
    if silent:
        ax.legend(loc="lower right", fontsize=7, markerscale=1.2, handletextpad=0.2)

    ax.scatter(units["x_um"], units["y_um"],
               s=12, c="k", edgecolors="none", zorder=3)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x position (µm)")
    ax.set_ylabel("y position (µm)")

    if standalone:
        ax.set_title(name, loc="left")
        ax.text(0.5, 1.02,
                f"{n_units} units across {len(used)} electrodes",
                transform=ax.transAxes, ha="center", fontsize=8)
        fig.text(0.5, 0.015,
                f"{params.sorter}  ·  "
                f"{params.freq_min:.0f}–{params.freq_max:.0f} Hz  ·  "
                f"{params.common_reference} reference  ·  "
                f"black dots mark estimated unit centres",
                ha="center", fontsize=7, color="#555555")

    return fig


def traces(seg, fs, ax=None, offset_uv=120.0, lw=0.5, ms=False, n_unit=None):
    """Stacked raw traces. seg is (n_samples, n_channels) in µV."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    t = np.arange(seg.shape[0]) / fs
    if ms:
        t = t * 1000

    n_unit = seg.shape[1] if n_unit is None else n_unit

    for k in range(seg.shape[1]):
        colour = PURPLE if k < n_unit else "k"
        ax.plot(t, seg[:, k] - k * offset_uv, lw=lw, color=colour)

    if 0 < n_unit < seg.shape[1]:
        ax.plot([], [], color=PURPLE, lw=1.2, label="unit electrode")
        ax.plot([], [], color="k", lw=1.2, label="most active")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    elif n_unit == 0:
        ax.plot([], [], color="k", lw=1.2, label="most active (no units)")
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    ax.set_xlim(0, t[-1] * 1.03)          # right-hand padding so the traces aren't clipped
    ax.axis("off")

    span = t[-1]
    dx = span * 0.2
    label = f"{dx:.0f} ms" if ms else f"{dx*1000:.0f} ms"
    _scale_bar(ax, span * 0.02, -seg.shape[1] * offset_uv + 20,
               dx, 50, label, "50 µV")
    return ax



def rates(t, per_unit, window_s, spike_times, duration, name="") -> Figure:
    """Rate over time without letting one fast unit dominate.

    A: each unit normalised to its own mean, median and IQR across units.
    B: every unit's sliding-window rate. C: cumulative spike count (no smoothing).
    """
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{name}  ·  {len(per_unit)} units  ·  {window_s:.0f} s sliding-window average", fontsize=10)

    if not len(per_unit):
        axA.text(0.5, 0.5, "no units", transform=axA.transAxes, ha="center", color="#555555")
        return fig

    m = per_unit.mean(axis=1, keepdims=True)
    norm = np.divide(per_unit, m, out=np.zeros_like(per_unit, dtype=float), where=m > 0)
    q25, q50, q75 = np.percentile(norm, [25, 50, 75], axis=0)
    axA.plot(t, q50, color=PURPLE, lw=1.6, label="Median across units")
    axA.fill_between(t, q25, q75, color=PURPLE, alpha=0.25, linewidth=0, label="Interquartile range")
    axA.axhline(1, color="#555555", lw=0.6, ls=":")
    axA.set_ylabel("Normalised firing rate\n(× unit mean)")
    axA.legend(loc="upper right")

    for row in per_unit:
        axB.plot(t, row, color=PURPLE, lw=1.0, alpha=0.7)
    axB.set_yscale("symlog", linthresh=0.1)
    axB.set_ylim(0, None)
    axB.set_ylabel("Firing rate per unit (Hz)")

    for st in spike_times:
        axC.step(np.r_[0, st, duration], np.r_[0, np.arange(1, len(st) + 1), len(st)],
                 where="post", color=PURPLE, lw=1.0, alpha=0.7)
    axC.set_yscale("symlog", linthresh=10)
    axC.set_ylabel("Cumulative spike count")

    for ax, letter in ((axA, "A"), (axB, "B"), (axC, "C")):
        ax.set_title(letter)
        ax.set_xlabel("Time (s)")
        ax.tick_params(labelbottom=True)
        if ax.get_yscale() == "symlog":
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axC.set_xlim(0, duration)
    fig.tight_layout()
    return fig


def diagnostics(isi_amps: dict, ncols: int = 4) -> Figure:
    """Per unit: ISI histogram (0-50 ms) and amplitude histogram.

    Real cell: empty ISIs below ~1.5 ms and a full bell of amplitudes.
    Noise/multi-unit: ISIs near 0 or amplitudes cut off at threshold.
    Mains: ISI peaks at 16.7 ms and multiples.
    """
    n = max(len(isi_amps), 1)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, 2 * ncols, figsize=(3.2 * ncols, 1.8 * nrows), squeeze=False)

    for k, (uid, (isi, amp)) in enumerate(isi_amps.items()):
        a_isi, a_amp = axes[k // ncols][2 * (k % ncols)], axes[k // ncols][2 * (k % ncols) + 1]
        a_isi.hist(isi[isi < 50], bins=np.arange(0, 50.25, 0.25), color=PURPLE)
        a_isi.axvspan(0, 1.5, color=RED, alpha=0.15, linewidth=0)
        for f in (16.7, 33.3):
            a_isi.axvline(f, color="#555555", lw=0.5, ls=":")
        a_isi.set_title(f"unit {uid}  ·  {len(amp)} spikes", loc="left", fontsize=7)
        a_amp.hist(amp, bins=40, color=BLUE)
        for a, xl in ((a_isi, "Inter-spike interval (ms)"), (a_amp, "Spike amplitude (µV)")):
            a.tick_params(labelsize=6)
            a.set_xlabel(xl, fontsize=7)
            a.set_ylabel("Spike count", fontsize=7)

    for k in range(len(isi_amps), nrows * ncols):
        for j in (0, 1):
            axes[k // ncols][2 * (k % ncols) + j].axis("off")

    fig.tight_layout()
    return fig


def waveforms(templates, unit_ids, units: pd.DataFrame, fs: float,
              ax=None, ncols: int = 6) -> Figure:
    """Grid of unit templates, each on its own peak channel."""
    n = len(unit_ids)

    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no units", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#555555")
        ax.axis("off")
        return fig

    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows),
                             sharey=True, squeeze=False)

    cols = ["firing_rate", "snr", "isi_violations_ratio",
            "num_spikes", "x_um", "y_um", "peak_channel",
            "peak_to_trough_duration", "trough_half_width"]
    info = units.set_index("unit_id")[
        [c for c in cols if c in units.columns]
    ].to_dict("index")

    for k, uid in enumerate(unit_ids):
        a = axes[k // ncols][k % ncols]
        t = templates[k]
        ch = np.abs(t).max(axis=0).argmax()
        wf = t[:, ch]
        tt = np.arange(len(wf)) / fs * 1000

        a.plot(tt, wf, color=PURPLE, lw=1.0)

        u = info.get(uid, {})
        a.set_title(f"unit {uid}  ·  ch {u.get('peak_channel', '?')}",
                    loc="left", fontsize=7, fontweight="bold")
        a.text(
            0.97, 0.05,
            f"{u.get('firing_rate', float('nan')):.2f} Hz\n"
            f"{int(u.get('num_spikes', 0))} spikes\n"
            f"snr {u.get('snr', float('nan')):.1f}\n"
            f"p2t {u.get('peak_to_trough_duration', float('nan'))*1000:.2f} ms\n"
            f"hw {u.get('trough_half_width', float('nan'))*1000:.2f} ms\n"
            f"isi {u.get('isi_violations_ratio', float('nan')):.2f}\n"
            f"({u.get('x_um', float('nan')):.0f}, {u.get('y_um', float('nan')):.0f}) µm",
            transform=a.transAxes, ha="right", va="bottom",
            fontsize=6, color="#555555", linespacing=1.4,
        )

        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=6)
        if k // ncols == nrows - 1:
            a.set_xlabel("Time (ms)", fontsize=8)
        if k % ncols == 0:
            a.set_ylabel("Amplitude (µV)", fontsize=8)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.tight_layout()
    return fig


def summary(units, locations, channel_ids, unit_chans, seg, zseg, fs,
            params, name="", tag="", n_unit=None,
            t=None, per_unit=None, window_s=None, band: str = "range",
            silent_chans=()) -> Figure:
    """A: unit map, B: traces (+ zoom), C: mean firing rate across units over time.

    band: spread shown around the mean in C: "range" (min-max across units),
    "iqr" (25th-75th percentile), "sem" (± standard error of the mean) or "none".
    """
    fig = plt.figure(figsize=(15, 8.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1.5, 0.8], height_ratios=[1.3, 0.7],
                  wspace=0.28, hspace=0.35)

    if tag:
        fig.suptitle(tag, fontsize=12, y=0.99)

    axA = fig.add_subplot(gs[0, 0])
    unit_map(units, locations, channel_ids, unit_chans, params, ax=axA, silent_chans=silent_chans)
    axA.set_title("A", loc="left")
    axA.text(0.5, 1.02, f"{len(units)} units, purple = electrodes > 3× noise, black = centres",
             transform=axA.transAxes, ha="center", fontsize=8)

    axB = fig.add_subplot(gs[0, 1])
    traces(seg, fs, ax=axB, n_unit=n_unit)
    axB.set_title("B", loc="left")
    axB.text(0.5, 1.02, f"{seg.shape[1]} electrodes, {seg.shape[0]/fs:.0f} s",
             transform=axB.transAxes, ha="center", fontsize=8)

    axZ = fig.add_subplot(gs[0, 2])
    traces(zseg, fs, ax=axZ, lw=0.7, ms=True, n_unit=n_unit)
    axZ.text(0.5, 1.02, f"zoom: {zseg.shape[0]/fs*1000:.0f} ms",
             transform=axZ.transAxes, ha="center", fontsize=8)

    axC = fig.add_subplot(gs[1, :])
    axC.set_title("C", loc="left")
    if per_unit is not None and len(per_unit):
        mean = per_unit.mean(axis=0)
        sem = per_unit.std(axis=0) / np.sqrt(len(per_unit))
        if band != "none":
            lo, hi, band_label = {
                "range": (per_unit.min(axis=0), per_unit.max(axis=0), "Min–max across units"),
                "iqr": (*np.percentile(per_unit, [25, 75], axis=0), "Interquartile range"),
                "sem": (mean - sem, mean + sem, "± SEM"),
            }[band]
            axC.fill_between(t, lo, hi, color=PURPLE, alpha=0.2, linewidth=0, label=band_label)
        axC.plot(t, mean, color=PURPLE, lw=1.4, label=f"Mean ({len(per_unit)} units)")
        axC.legend(loc="upper right")
        axC.set_xlim(0, t[-1] + (t[1] - t[0]) / 2 if len(t) > 1 else t[-1])
        axC.set_ylim(0, None)
        axC.text(0.5, 1.02, f"mean of {len(per_unit)} units, {window_s:.0f} s sliding window",
                 transform=axC.transAxes, ha="center", fontsize=8)
    else:
        axC.text(0.5, 0.5, "no units", transform=axC.transAxes, ha="center", va="center",
                 fontsize=9, color="#555555")
    axC.set_xlabel("Time (s)")
    axC.set_ylabel("Mean firing rate per unit (Hz)")

    fig.text(
        0.5, 0.01,
        f"{name}  ·  {params.sorter}  ·  "
        f"{params.freq_min:.0f}–{params.freq_max:.0f} Hz band-pass  ·  "
        f"{params.common_reference} reference  ·  "
        f"snr ≥ {params.min_snr:g}, presence ≥ {params.min_presence:g}, "
        f"rate ≥ {params.min_rate_hz:g} Hz",
        ha="center", fontsize=7, color="#555555",
    )
    return fig


def comparison(recs: list[dict], locations, channel_ids, window_s, band="sem", name="") -> Figure:
    """One slice, all recordings: A electrodes per recording, B mean rates overlaid, C rate per unit.

    recs: one dict per recording, in order, with label, unit_chans, t, per_unit and
    rates ({unit id: Hz}). Units missing from a recording (left out as noise) get an ×.
    """
    words = [r["label"].split() for r in recs]
    n = next((i for i, w in enumerate(zip(*words)) if len(set(w)) > 1), 0)
    labels = [" ".join(w[n:]) or r["label"] for w, r in zip(words, recs)]
    colors = CONDITIONS[:len(recs)]
    pos = {c: i for i, c in enumerate(channel_ids)}

    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 2.2], wspace=0.22, hspace=0.35)
    fig.suptitle(name, fontsize=12, y=0.99)

    axA = fig.add_subplot(gs[:, 0])
    axA.scatter(locations[:, 0], locations[:, 1], s=5, c=GREY_L, edgecolors="none")
    sizes = np.geomspace(75, 12, len(recs))   # concentric: first recording largest, at the back
    for k, (r, c, lab) in enumerate(zip(recs, colors, labels)):
        idx = sorted({pos[ch] for chs in r["unit_chans"].values() for ch in chs if ch in pos})
        axA.scatter(locations[idx, 0], locations[idx, 1], s=sizes[k], c=c,
                    edgecolors="none", zorder=2 + k, label=lab)
    axA.set_aspect("equal"); axA.invert_yaxis()
    axA.set_xlabel("x position (µm)"); axA.set_ylabel("y position (µm)")
    axA.legend(loc="lower right", fontsize=8, title="Electrodes > 3× noise", title_fontsize=8)
    axA.set_title("A", loc="left")

    axB = fig.add_subplot(gs[0, 1])
    for r, c, lab in zip(recs, colors, labels):
        pu = r["per_unit"]
        if not len(pu):
            continue
        mean = pu.mean(axis=0)
        if band != "none":
            sem = pu.std(axis=0) / np.sqrt(len(pu))
            lo, hi = {"range": (pu.min(axis=0), pu.max(axis=0)),
                      "iqr": tuple(np.percentile(pu, [25, 75], axis=0)),
                      "sem": (mean - sem, mean + sem)}[band]
            axB.fill_between(r["t"], lo, hi, color=c, alpha=0.12, linewidth=0)
        axB.plot(r["t"], mean, color=c, lw=1.6, label=f"{lab} ({len(pu)} units)")
    band_txt = {"range": ", shaded min–max", "iqr": ", shaded IQR", "sem": ", shaded ± SEM"}.get(band, "")
    axB.text(0.5, 1.02, f"mean across units, {window_s:.0f} s sliding window{band_txt}",
             transform=axB.transAxes, ha="center", fontsize=8)
    axB.set_xlim(0, max(r["t"][-1] for r in recs)); axB.set_ylim(0, None)
    axB.set_xlabel("Time (s)"); axB.set_ylabel("Mean firing rate per unit (Hz)")
    axB.legend(loc="upper right"); axB.set_title("B", loc="left")

    axC = fig.add_subplot(gs[1, 1])
    units = sorted(set().union(*(r["rates"] for r in recs)))
    w = 0.8 / len(recs)
    for k, (r, c, lab) in enumerate(zip(recs, colors, labels)):
        x = np.arange(len(units)) + (k - (len(recs) - 1) / 2) * w
        have = np.array([u in r["rates"] for u in units], bool)
        axC.bar(x[have], [r["rates"][u] for u, h in zip(units, have) if h], width=w * 0.9,
                color=c, label=lab)
        axC.scatter(x[~have], np.zeros((~have).sum()), marker="x", s=30, color=c, zorder=3)
    axC.set_xticks(range(len(units)), [str(u) for u in units])
    axC.set_yscale("symlog", linthresh=0.1)
    axC.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axC.set_xlabel("Unit"); axC.set_ylabel("Firing rate (Hz)")
    axC.text(0.5, 1.02, "whole-recording rate per unit, × = left out of that recording as noise",
             transform=axC.transAxes, ha="center", fontsize=8)
    axC.legend(loc="upper right", fontsize=8); axC.set_title("C", loc="left")
    return fig


# private helper functions

def _scale_bar(ax, x, y, dx, dy, xlabel, ylabel, color="k") -> None:
    ax.plot([x, x], [y, y + dy], color=color, lw=1.5, solid_capstyle="butt")
    ax.plot([x, x + dx], [y, y], color=color, lw=1.5, solid_capstyle="butt")
    ax.text(x + dx / 2, y - dy * 0.12, xlabel, ha="center", va="top", fontsize=7)
    ax.text(x - dx * 0.04, y + dy / 2, ylabel, ha="right", va="center",
            fontsize=7, rotation=90)


apply_style()




