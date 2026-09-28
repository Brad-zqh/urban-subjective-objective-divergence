"""Non-destructive JPG/SVG export and lightweight figure QA helpers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_overlap_pairs(fig, *, fraction: float = 0.12) -> list[dict]:
    """Report material displayed-text bbox collisions for human review."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidate = list(fig.texts)
    parents = {id(artist): "figure" for artist in fig.texts}
    axes = list(fig.axes)
    for ax in list(fig.axes):
        axes.extend(ax.child_axes)
    for ax in axes:
        label = ax.get_title(loc="left") or ax.get_title() or "untitled axis"
        parts = [ax.title, ax._left_title, ax._right_title, *ax.texts]
        if ax.axison:
            parts.extend([ax.xaxis.label, ax.yaxis.label])
            parts.extend(ax.get_xticklabels())
            parts.extend(ax.get_yticklabels())
        if ax.get_legend() is not None:
            parts.extend(ax.get_legend().get_texts())
        candidate.extend(parts)
        parents.update({id(artist): label for artist in parts})
    for legend in fig.legends:
        candidate.extend(legend.get_texts())
    texts = []
    seen = set()
    for artist in candidate:
        if id(artist) in seen or not isinstance(artist, Text):
            continue
        seen.add(id(artist))
        if artist.get_visible() and artist.get_text().strip():
            texts.append(artist)
    boxes = [(artist, artist.get_window_extent(renderer)) for artist in texts]
    collisions = []
    for index, (left, box_left) in enumerate(boxes):
        if not all(math.isfinite(v) for v in box_left.extents):
            continue
        if box_left.width <= 0 or box_left.height <= 0:
            continue
        for right, box_right in boxes[index + 1:]:
            if not all(math.isfinite(v) for v in box_right.extents):
                continue
            if box_right.width <= 0 or box_right.height <= 0:
                continue
            x = min(box_left.x1, box_right.x1) - max(box_left.x0, box_right.x0)
            y = min(box_left.y1, box_right.y1) - max(box_left.y0, box_right.y0)
            if x <= 0 or y <= 0:
                continue
            overlap = x * y / min(box_left.width * box_left.height,
                                  box_right.width * box_right.height)
            if overlap >= fraction:
                collisions.append({
                    "left": left.get_text()[:60], "right": right.get_text()[:60],
                    "left_axes": parents.get(id(left), "figure"),
                    "right_axes": parents.get(id(right), "figure"),
                    "left_box": [round(float(v), 1) for v in box_left.extents],
                    "right_box": [round(float(v), 1) for v in box_right.extents],
                    "overlap_fraction": round(float(overlap), 3),
                })
    return collisions


def save_jpg_svg(fig, output: Path, *, dpi: int = 600) -> dict:
    """Save exactly the two user-requested formats without changing data."""
    output.mkdir(parents=True, exist_ok=True)
    stem = output / output.name
    svg = stem.with_suffix(".svg")
    jpg = stem.with_suffix(".jpg")
    fig.canvas.draw()
    overlaps = text_overlap_pairs(fig)
    fig.savefig(svg, facecolor="white")
    fig.savefig(
        jpg,
        dpi=dpi,
        facecolor="white",
        pil_kwargs={"quality": 96, "subsampling": 0, "optimize": True},
    )
    with Image.open(jpg) as im:
        im.verify()
    with Image.open(jpg) as im:
        dimensions = list(im.size)
        stored_dpi = [float(v) for v in im.info.get("dpi", (0, 0))]
    svg_text = svg.read_text(encoding="utf-8")
    plt.close(fig)
    return {
        "jpg": str(jpg),
        "svg": str(svg),
        "jpg_dimensions_px": dimensions,
        "jpg_dpi_metadata": stored_dpi,
        "jpg_nonempty": jpg.stat().st_size > 100_000,
        "svg_nonempty": svg.stat().st_size > 10_000,
        "svg_text_editable": "<text" in svg_text,
        "delivery_formats": ["jpg", "svg"],
        "text_overlap_pairs": overlaps[:100],
        "text_overlap_count": len(overlaps),
    }
