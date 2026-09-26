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
GREY_M = "#cfcfcf"
BG     = "#f0f0f0"

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
             ax=None) -> Figure:
    """Electrodes in grey; those belonging to a unit in purple, unit centres black."""
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
    ax.scatter(locations[used, 0], locations[used, 1],
               s=18, c=PURPLE, edgecolors="none", zorder=2)

    ax.scatter(units["x_um"], units["y_um"],
               s=12, c="k", edgecolors="none", zorder=3)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

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

    ax.set_xlim(0, t[-1])
    ax.axis("off")

    span = t[-1]
    dx = span * 0.2
    label = f"{dx:.0f} ms" if ms else f"{dx*1000:.0f} ms"
    _scale_bar(ax, span * 0.02, -seg.shape[1] * offset_uv + 20,
               dx, 50, label, "50 µV")
    return ax



def rates(t_mid, per_unit, bin_s, spike_times, duration, name="") -> Figure:
    """Rate over time without letting one fast unit dominate.

    A: each unit normalised to its own mean, median and IQR across units.
    B: every unit's rate on a symlog axis. C: cumulative spike count (no binning).
    """
    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"{name}  ·  {len(per_unit)} units  ·  {bin_s:.0f} s bins", fontsize=10)

    if not len(per_unit):
        axA.text(0.5, 0.5, "no units", transform=axA.transAxes, ha="center", color="#555555")
        return fig

    m = per_unit.mean(axis=1, keepdims=True)
    norm = np.divide(per_unit, m, out=np.zeros_like(per_unit, dtype=float), where=m > 0)
    q25, q50, q75 = np.percentile(norm, [25, 50, 75], axis=0)
    axA.plot(t_mid, q50, color=PURPLE, lw=1.6, label="median")
    axA.fill_between(t_mid, q25, q75, color=PURPLE, alpha=0.25, linewidth=0, label="IQR")
    axA.axhline(1, color="#555555", lw=0.6, ls=":")
    axA.set_ylabel("Rate / unit mean")
    axA.legend(loc="upper right")
    axA.set_title("A")

    for row in per_unit:
        axB.plot(t_mid, row, color=PURPLE, lw=0.8, alpha=0.6)
    axB.set_yscale("symlog", linthresh=0.1)
    axB.set_ylim(0, None)
    axB.set_ylabel("Firing rate (Hz)")
    axB.set_title("B")

    for t in spike_times:
        axC.step(np.r_[0, t, duration], np.r_[0, np.arange(1, len(t) + 1), len(t)],
                 where="post", color=PURPLE, lw=0.8, alpha=0.6)
    axC.set_yscale("symlog", linthresh=10)
    axC.set_ylabel("Cumulative spikes")
    axC.set_xlabel("Time (s)")
    axC.set_xlim(0, duration)
    axC.set_title("C")
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
        for a, xl in ((a_isi, "ISI (ms)"), (a_amp, "amplitude (µV)")):
            a.tick_params(labelsize=6)
            a.set_xlabel(xl, fontsize=7)
            a.set_yticks([])

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
            a.set_xlabel("ms", fontsize=8)
        if k % ncols == 0:
            a.set_ylabel("µV", fontsize=8)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.tight_layout()
    return fig


def summary(units, locations, channel_ids, unit_chans, seg, zseg, fs,
            params, name="", tag="", n_unit=None) -> Figure:
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.5, 0.8], wspace=0.28)

    if tag:
        fig.suptitle(tag, fontsize=12, y=1.0)

    axA = fig.add_subplot(gs[0, 0])
    unit_map(units, locations, channel_ids, unit_chans, params, ax=axA)
    axA.set_title("A", loc="left")
    axA.text(0.5, 1.02, f"{len(units)} units, purple = unit electrodes, black = centres",
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

    fig.text(
        0.5, -0.02,
        f"{name}  ·  {params.sorter}  ·  "
        f"{params.freq_min:.0f}–{params.freq_max:.0f} Hz band-pass  ·  "
        f"{params.common_reference} reference  ·  "
        f"snr ≥ {params.min_snr:g}, presence ≥ {params.min_presence:g}, "
        f"≥ {params.min_spikes} spikes",
        ha="center", fontsize=7, color="#555555",
    )
    return fig


# private helper functions

def _scale_bar(ax, x, y, dx, dy, xlabel, ylabel, color="k") -> None:
    ax.plot([x, x], [y, y + dy], color=color, lw=1.5, solid_capstyle="butt")
    ax.plot([x, x + dx], [y, y], color=color, lw=1.5, solid_capstyle="butt")
    ax.text(x + dx / 2, y - dy * 0.12, xlabel, ha="center", va="top", fontsize=7)
    ax.text(x - dx * 0.04, y + dy / 2, ylabel, ha="right", va="center",
            fontsize=7, rotation=90)


apply_style()




