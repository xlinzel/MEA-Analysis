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
                f"merge ≤ {params.merge_thresh:g}  ·  "
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



def population_rate(rates: pd.DataFrame, ax=None, source: str = "unit"):
    """Mean rate ± SEM, with population total on the right axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    if len(rates) == 0:
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(f"Mean firing rate per {source} (Hz)", fontsize=10)
        ax.text(0.5, 0.5, "no units", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#555555")
        return ax

    t = rates["t_s"].to_numpy()
    mean = rates["mean_hz"].to_numpy()
    sem = rates["sem_hz"].to_numpy()
    n = int(rates["n_units"].iloc[0])

    ax.plot(t, mean, color=PURPLE, lw=1.6)
    ax.fill_between(t, mean - sem, mean + sem,
                    color=PURPLE, alpha=0.25, linewidth=0)

    ax.plot([], [], color=PURPLE, lw=1.6, label=f"mean ({n} {source}s)")
    ax.fill_between([], [], color=PURPLE, alpha=0.25, linewidth=0, label="± SEM")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    top = (mean + sem).max() * 1.1
    ax.set_ylim(0, top)
    ax.set_xlim(0, t[-1] + (t[1] - t[0]) / 2 if len(t) > 1 else t[-1])
    ax.set_ylabel(f"Mean firing rate per {source} (Hz)", fontsize=10)
    ax.margins(x=0)

    ax2 = ax.twinx()
    ax2.set_ylim(0, top * n)
    ax2.set_ylabel("Population rate (Hz)", fontsize=10)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    return ax


def unit_rates(rates: pd.DataFrame, per_unit: np.ndarray, ax=None,
               source: str = "unit"):
    """Every unit's rate over time, one line each."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    if per_unit is None or not per_unit.size:
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Firing rate (Hz)", fontsize=10)
        ax.text(0.5, 0.5, "no units", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#555555")
        return ax

    t = rates["t_s"].to_numpy()
    for row in per_unit:
        ax.plot(t, row, color=PURPLE, lw=0.8, alpha=0.6)

    ax.set_ylim(0, per_unit.max() * 1.1)
    ax.set_xlim(0, t[-1] + (t[1] - t[0]) / 2 if len(t) > 1 else t[-1])
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Firing rate (Hz)", fontsize=10)
    ax.margins(x=0)
    return ax


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
            "peak_to_valley", "half_width"]
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
            f"p2v {u.get('peak_to_valley', float('nan'))*1000:.2f} ms\n"
            f"hw {u.get('half_width', float('nan'))*1000:.2f} ms\n"
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
            rates, params, name="", tag="", n_unit=None,
            per_unit=None) -> Figure:
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1.5, 0.8],
            height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.28)

    if tag:
        fig.suptitle(tag, fontsize=12, y=0.98)

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

    axC = fig.add_subplot(gs[1, :])
    population_rate(rates, ax=axC)
    axC.set_title("C", loc="left")
    axC.tick_params(labelbottom=False)

    axD = fig.add_subplot(gs[2, :], sharex=axC)
    unit_rates(rates, per_unit, ax=axD)
    axD.set_title("D", loc="left")
    axD.text(0.5, 1.02, f"{len(units)} units, one line each",
             transform=axD.transAxes, ha="center", fontsize=8)

    fig.text(
        0.5, 0.01,
        f"{name}  ·  {params.sorter}  ·  "
        f"{params.freq_min:.0f}–{params.freq_max:.0f} Hz band-pass  ·  "
        f"{params.common_reference} reference  ·  "
        f"merge ≤ {params.merge_thresh:g}  ·  "
        f"snr ≥ {params.min_snr:g}, isi ≤ {params.max_isi_viol:g}  ·  "
        f"one source = one sorted unit",
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




