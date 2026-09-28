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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


# Resolve the monorepo root from this shared module.  This keeps every figure
# renderer portable across Windows, Linux, and a clean GitHub checkout.
ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets/fonts/nimbus-sans"

BLUE = "#3B4CC0"
BLUE_MID = "#8DB0D5"
OFFWHITE = "#F7F7F7"
RED_MID = "#E6A6A1"
RED = "#B40426"
INK = "#252A33"
MUTED = "#69727E"
GRID = "#DDE3EA"
MISSING = "#D5D9DF"

DIV = LinearSegmentedColormap.from_list(
    "gmh_exact_red_blue", [BLUE, BLUE_MID, OFFWHITE, RED_MID, RED]
)
DIV.set_bad(MISSING)


def setup_style(base_size: float = 6.2) -> None:
    """Register Nimbus Sans and apply the shared manuscript style."""
    for path in FONT_DIR.glob("NimbusSans-*.otf"):
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": base_size,
            "font.weight": "normal",
            "axes.titlesize": base_size + 0.6,
            "axes.titleweight": "normal",
            "axes.labelsize": base_size,
            "axes.labelweight": "normal",
            "xtick.labelsize": base_size - 0.8,
            "ytick.labelsize": base_size - 0.8,
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


def add_equal_height_colorbar(fig, ax, mappable, width: float = 0.038, gap: float = 0.026):
    """Attach an individual colour bar with exactly the map Axes height."""
    cax = ax.inset_axes([1.0 + gap, 0.0, width, 1.0], transform=ax.transAxes)
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(labelsize=3.3, length=1.0, width=0.35, pad=0.4)
    cb.outline.set_linewidth(0.35)
    return cb, cax


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_delivery(fig, out_dir: Path, stem: str, *, dpi: int = 600) -> dict:
    """Save the V6 vector/raster set plus JPEG, then verify every raster."""
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
