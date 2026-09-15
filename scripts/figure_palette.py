#!/usr/bin/env python
"""Shared colour palette for experiment figures.

Single source of truth so that a colour family stays identical across
experiment figures instead of being copy-pasted per script.

Provenance: Phase F (2026-09-08_phaseF_fp8pot-foursuite) introduced the
four-suite palette -- four hues (one per LIBERO suite) crossed with three
lightness steps (one per quantization protocol, light = milder). Phase G
reuses the same hues so the two figure families read as one set.

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N] / "scripts"))
    from figure_palette import SUITE_SHADES, GOAL_SHADES_4, C_FP, readable_on
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase F four-suite palette
# ---------------------------------------------------------------------------

SUITE_LABELS = ["Spatial", "Object", "Goal", "LIBERO-10"]

# hue -> (light, mid, dark). Same hue inside a row, light -> dark down the row.
SUITE_SHADES = {
    "Spatial":   ("#9ecae1", "#4292c6", "#08519c"),  # blue
    "Object":    ("#fdae6b", "#e6550d", "#a63603"),  # orange
    "Goal":      ("#a1d99b", "#31a354", "#006d2c"),  # green
    "LIBERO-10": ("#bcbddc", "#807dba", "#54278f"),  # purple
}

# Four-step Goal (green) ramp for figures that express a progression with a
# single hue: light -> dark as the setting gets more aggressive.
# The middle three entries are exactly SUITE_SHADES["Goal"] (ColorBrewer
# Greens 3/6/8); the lightest entry extends the ramp with Greens 2.
GOAL_SHADES_4 = ("#c7e9c0", "#a1d99b", "#31a354", "#006d2c")

# Five-step blue ramp (ColorBrewer Blues 3/4/6/7/9). Used when a figure zooms
# into one branch of the hierarchy and needs five distinguishable steps of the
# same hue -- e.g. Phase I breaking Vision+Connector down into its sub-modules,
# which mirrors the blue (first-hue) branch of the top-level breakdown.
# Shares #9ecae1 and #4292c6 with the three-step Spatial family above.
BLUE_SHADES_5 = ("#c6dbef", "#9ecae1", "#4292c6", "#2171b5", "#08306b")

# ---------------------------------------------------------------------------
# neutral / semantic colours
# ---------------------------------------------------------------------------

C_FP = "#d9d9d9"      # full-precision reference bar
C_REF = "#4d4d4d"     # reference line for an in-figure anchor
C_POS = "#1a7f37"     # improvement
C_NEG = "#b3261e"     # degradation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def luminance(hex_color: str) -> float:
    """Perceived luminance of a #rrggbb colour, in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def readable_on(hex_color: str) -> str:
    """Pick black or white text for maximum contrast on a fill colour.

    Needed because a light shade (e.g. GOAL_SHADES_4[0]) makes white text
    unreadable while a dark shade makes black text unreadable.
    """
    return "white" if luminance(hex_color) < 0.55 else "black"
