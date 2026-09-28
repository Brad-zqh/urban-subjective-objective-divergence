"""Shared, deterministic style and QA helpers for the V211 figure suite.

All public renderers use this module so that typography, the signed colour
grammar, panel lettering, and export checks cannot silently drift between
figures.  It preserves the legacy V6 editable/raster export contract and also
keeps a 600-dpi JPEG for the delivery index.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import string

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.text import Text
from PIL import Image


# Resolve the monorepo root from this shared module.  This keeps every figure
# renderer portable across Windows, Linux, and a clean GitHub checkout.
ROOT = Path(__file__).resolve().parents[1]

BLUE = "#3B4CC0"
BLUE_MID = "#8DB0D5"
OFFWHITE = "#F7F7F7"
RED_MID = "#E6A6A1"
RED = "#B40426"
INK = "#000000"
MUTED = "#69727E"
GRID = "#DDE3EA"
MISSING = "#D5D9DF"
# 4 pt labels are technically present in a vector file but are not readable
# once a multi-panel figure is placed at the manuscript's double-column width.
# Keep a single, submission-wide floor so that small-multiple panels and all
# independent colour-bar tick labels survive the final Word/PDF placement.
MIN_VISIBLE_TEXT_PT = 5.5

# Submission-wide display language. Figure builders import these labels so
# the same construct is not renamed between the main text and Supplementary
# Information. Internal dataframe column names remain unchanged.
DIMENSION_LABELS = {
    1: "Green / blue",
    2: "Density",
    3: "Access",
    4: "Safety",
    5: "Cleanliness",
    6: "Social",
    7: "Heat",
    8: "Air pollution",
    9: "Noise",
    10: "Traffic",
}

SES_LABELS = {
    "ses__age_65plus_pct": "Age ≥65",
    "ses__age_under18_pct": "Age <18",
    "ses__bachelors_or_higher_pct_25plus": "Higher education",
    "ses__black_alone_pct": "Black population",
    "ses__commute_transit_pct": "Transit commute",
    "ses__commute_walk_pct": "Walk commute",
    "ses__gross_rent_30plus_pct": "Rent burden",
    "ses__hispanic_latino_pct": "Hispanic / Latino",
    "ses__log_income_nominal": "Log income",
    "ses__male_pct": "Male population",
    "ses__population_log1p": "Population size",
    "ses__poverty_pct": "Poverty",
    "ses__unemployment_pct": "Unemployment",
}

MODEL_LABELS = {
    "OLS": "OLS/Ridge",
    "GTWR": "GTWR",
    "MLP_GPU": "MLP",
    "GraphSAGE_GPU": "GraphSAGE",
    "GTNNWR_GPU": "GTNNWR",
    "GTCNNWR_GPU": "GTCNNWR",
    "GTGNNWR_GPU": "GTGNNWR",
    "MM_GTGNNWR": "MM-GTGNNWR",
}

MODALITY_LABELS = {
    "gsv_only": "GSV",
    "google_only": "Google Maps",
    "flickr_only": "Flickr",
    "image_only": "Image channels",
    "text_only": "Text channels",
}

EVALUATION_STAGE_LABELS = {
    "train": "Train",
    "validation": "Validation",
    "outer": "Outer test",
    "ridge": "Ridge",
}

DIV = LinearSegmentedColormap.from_list(
    "gmh_exact_red_blue", [BLUE, BLUE_MID, OFFWHITE, RED_MID, RED]
)
DIV.set_bad(MISSING)


def setup_style(base_size: float = 6.2) -> None:
    """Apply the shared Helvetica-first manuscript style.

    Helvetica is preferred when available; Arial is the metrically compatible
    system fallback used on the current Windows rendering host.
    """
    # A small global lift keeps the visual hierarchy consistent across the
    # independently maintained renderers. Explicitly larger local labels are
    # left untouched; only the formerly tiny base/tick labels move upward.
    display_size = max(float(base_size) * 1.08, 6.2)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": display_size,
            "font.weight": "normal",
            "axes.titlesize": display_size + 0.6,
            "axes.titleweight": "normal",
            "axes.labelsize": display_size,
            "axes.labelweight": "normal",
            "xtick.labelsize": max(display_size - 0.8, MIN_VISIBLE_TEXT_PT),
            "ytick.labelsize": max(display_size - 0.8, MIN_VISIBLE_TEXT_PT),
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#59616B",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def blacken_figure_text(fig) -> None:
    """Enforce black, readable typography for every visible glyph."""
    for artist in fig.findobj(match=Text):
        artist.set_color("#000000")
        if artist.get_text().strip() and artist.get_fontsize() < MIN_VISIBLE_TEXT_PT:
            artist.set_fontsize(MIN_VISIBLE_TEXT_PT)


def symmetric_colourbar_ticks(vmin: float, vmax: float) -> list[float]:
    """Return five legible signed ticks with an explicit zero anchor."""
    limit = max(abs(float(vmin)), abs(float(vmax)))
    return [-limit, -limit / 2.0, 0.0, limit / 2.0, limit]


def panel(ax, index: int, title: str = "", x: float = -0.10, y: float = 1.04) -> None:
    """Add a regular-weight title and a bold lowercase panel letter."""
    if title:
        ax.set_title(title, loc="left", pad=2.4, fontweight="normal")
    ax.text(
        x,
        y,
        string.ascii_lowercase[index],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        clip_on=False,
    )


def clean_axes(ax, grid_axis: str | None = "both") -> None:
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.38, alpha=0.9)


def symmetric_limit(values, quantile: float = 0.985, floor: float = 1e-9) -> float:
    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return floor
    return max(float(np.quantile(np.abs(array), quantile)), floor)


def signed_norm(values, quantile: float = 0.985) -> TwoSlopeNorm:
    lim = symmetric_limit(values, quantile=quantile)
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def signed_magnitude_quantile_scale(values, quantiles=(.15, .45, .70, .88, .985)):
    """Return a zero-anchored discrete red-white-blue scale.

    The class widths follow quantiles of ``abs(values)`` rather than quantiles
    of the signed values themselves.  This keeps zero semantically fixed,
    prevents an imbalanced sign distribution from moving the neutral class,
    and still gives low-contrast local structure enough visual separation.
    The outer 1.5% is clipped for display only; source values are untouched.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    magnitudes = np.abs(array[array != 0])
    if magnitudes.size == 0:
        magnitudes = np.array([1.0])
    levels = np.asarray(np.quantile(magnitudes, quantiles), dtype=float)
    floor = max(float(levels[-1]) * 1e-7, 1e-12)
    levels = np.maximum(levels, floor)
    for index in range(1, len(levels)):
        if levels[index] <= levels[index - 1]:
            levels[index] = levels[index - 1] + floor
    boundaries = np.r_[-levels[::-1], levels]
    colors = [DIV(value) for value in (.02, .13, .24, .36, .50, .64, .76, .87, .98)]
    cmap = ListedColormap(colors, name="gmh_signed_magnitude_quantiles")
    cmap.set_bad(MISSING)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm, boundaries


def add_equal_height_colorbar(fig, ax, mappable, width: float = 0.050, gap: float = 0.020):
    """Attach a colour bar with exactly the mapped panel's plotting height."""
    # The colourbar is expressed in axes coordinates, so y=0 and height=1
    # make its spine align exactly with the plotted panel rather than merely
    # approximating the height after layout.
    cax = ax.inset_axes([1.0 + gap, 0.0, width, 1.0], transform=ax.transAxes)
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=5.8, length=1.5, width=0.50, pad=0.65)
    cb.outline.set_linewidth(0.35)
    return cb, cax


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_delivery(fig, out_dir: Path, stem: str, *, dpi: int = 600) -> dict:
    """Save the V6 vector/raster set plus JPEG, then verify every raster."""
    # Materialize tick labels (especially colour-bar ticks) before enforcing
    # the minimum type size; otherwise Matplotlib can create them lazily only
    # during the first save and bypass the typography gate.
    fig.canvas.draw()
    blacken_figure_text(fig)
    fig.canvas.draw()
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{stem}.svg"
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    tiff = out_dir / f"{stem}.tiff"
    jpg = out_dir / f"{stem}.jpg"
    fig.savefig(svg, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(
        tiff,
        dpi=dpi,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        jpg,
        dpi=dpi,
        facecolor="white",
        pil_kwargs={"quality": 96, "subsampling": 0, "optimize": True},
    )
    for raster in (jpg, png, tiff):
        with Image.open(raster) as image:
            image.verify()
    with Image.open(jpg) as image:
        dimensions = list(image.size)
        stored_dpi = [float(v) for v in image.info.get("dpi", (0, 0))]
    svg_text = svg.read_text(encoding="utf-8")
    return {
        "jpg": str(jpg),
        "pdf": str(pdf),
        "png": str(png),
        "svg": str(svg),
        "tiff": str(tiff),
        "jpg_dimensions_px": dimensions,
        "jpg_dpi_metadata": stored_dpi,
        "jpg_nonempty": jpg.stat().st_size > 100_000,
        "svg_nonempty": svg.stat().st_size > 10_000,
        "svg_text_editable": "<text" in svg_text,
        "delivery_formats": ["jpg", "pdf", "png", "svg", "tiff"],
    }


def write_validation(out_dir: Path, payload: dict) -> None:
    checks = payload.setdefault("checks", {})
    payload["all_checks_passed"] = bool(checks) and all(bool(v) for v in checks.values())
    def json_default(value):
        """Convert NumPy/Pandas scalar values without weakening validation."""
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    (out_dir / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
