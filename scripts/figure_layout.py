#!/usr/bin/env python
"""Shared figure layout helper.

Places a legend and a series of footnote lines strictly BELOW an axes, at
positions MEASURED from the renderer rather than hand-tuned figure fractions.

Why this exists: a hand-tuned fraction (e.g. bbox_to_anchor=(0.5, 0.275)) can
silently land on top of the multi-line x tick labels, which live OUTSIDE
``ax.get_window_extent()`` (the plot rectangle). Anchoring to
``ax.get_tightbbox()`` instead makes the placement correct for any tick-label
height, so a chart with three-line tick labels cannot break it.

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "scripts"))
    from figure_layout import stack_below_axes

    stack_below_axes(fig, ax, handles, [note1, note2], xc=0.54)
"""

from __future__ import annotations


def stack_below_axes(
    fig,
    ax,
    handles,
    notes,
    xc,
    *,
    gap_legend: float = 0.024,
    gap_text: float = 0.026,
    gap_line: float = 0.013,
    fontsize: float = 8.2,
    legend_fontsize: float = 9.5,
    color: str = "gray",
    warn_floor: float = 0.01,
):
    """Stack a legend then footnote lines below ``ax``, top-to-bottom.

    Args:
        fig, ax:     the figure and the axes to sit below.
        handles:     legend handles (may be empty to skip the legend).
        notes:       footnote strings, drawn top-to-bottom.
        xc:          figure-fraction x to centre everything on (pass the axes
                     centre, NOT 0.5, or side-by-side panels collide).
        gap_*:       vertical gaps in figure fraction between the successive
                     blocks (legend, text, then text-to-text).
        warn_floor:  warn if a note would fall below this y.

    Returns:
        (legend_or_None, [text_artists])
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    legend = None
    y = ax.get_tightbbox(r).transformed(inv).y0 - gap_legend
    if handles:
        legend = fig.legend(handles=handles, loc="upper center",
                            bbox_to_anchor=(xc, y), fontsize=legend_fontsize,
                            frameon=True, framealpha=0.95, borderpad=0.7,
                            handlelength=2.4)
        fig.canvas.draw()
        y = legend.get_window_extent(r).transformed(inv).y0 - gap_text

    texts = []
    for i, note in enumerate(notes):
        t = fig.text(xc, y, note, ha="center", va="top", fontsize=fontsize,
                     color=color)
        texts.append(t)
        fig.canvas.draw()
        y = t.get_window_extent(r).transformed(inv).y0 - gap_line
        if y < warn_floor:
            print(f"[warn] note {i} would run off the figure (y={y:.3f})")

    return legend, texts
