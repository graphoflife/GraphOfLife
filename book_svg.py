#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The book's figures as SVG files.

A figure is one chart or a grid of them, drawn to a file that every Markdown
reader shows as an image — the Book tab, Obsidian, GitHub — so a chapter is
complete as plain text and pictures, with nothing to run.

A chart is a dict:

    title           drawn above it
    x, y            axes: {label, min, max, log, categories}
    series          what is drawn, in order (see below)
    guides          dashed lines across it: {axis: "x"|"y", at, label}
    legend          False for none; otherwise every series with a label
    height          of the plotting area, in pixels

and a series is one of

    a line          {x, y, colour, width, dash, alpha, label}, with optional
                    bands behind it: lo/hi (drawn darker) and outerLo/outerHi
    points          the same with points: True and a dot size
    bars            {kind: "bars", x0, x1, y}: one bar from x0 to x1 each,
                    rising from 0 — or from y0, for bars stacked on others
    an area         {kind: "area", x, lo, hi}: filled between lo and hi
    cells           {kind: "cells", x0, x1, y0, y1, value}: rectangles coloured
                    by value on the chart's colour bar — a two-dimensional
                    histogram, say

and a chart with cells (or a network coloured by value) carries a colour bar,
{map, min, max, label, log}, drawn under it.

A picture of a network is {kind: "network", title, nodes: {x, y, group or
value}, edges: [[i, j], …], colour: {labels: […]} or {by: "value", map, min,
max, log, label}}; `edgeGroup` (one number per edge) and `edgeLabels` draw
groups of edges in colours of their own over the faint rest.

The colours are the page's own (web/js/ink.js, web/css/style.css), on the
page's dark card, so a figure looks the same inside the app and outside it.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The line colours, in order — the same as Ink.LINES in web/js/ink.js.
PALETTE = ("#5ac8fa", "#ffd166", "#ff6b6b", "#7ee787", "#c792ea", "#f78c6c", "#89ddff", "#e5e5e5")
CARD = "#18202b"
TEXT = "#eef4fa"
LABEL = "#9ab0c3"
DIM = "#7e94a5"
#: The site's one typeface (--font in web/css/style.css). Shown as an image, an SVG
#: sees only the fonts installed where it is read; the Book tab inlines its
#: figures, so there it is the web font itself.
FONT = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace"

WIDTH = 780              # of a whole figure, in pixels
LEGEND_ROW = 16


#: Colour maps, as a few colours to interpolate between. "viridis" runs from
#: dark to light, so a value reads as brightness; "signed" runs from blue
#: through the card's own dark grey to red, so that zero recedes and the two
#: signs stand out against it.
MAPS = {
    "viridis": ("#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39", "#fde725"),
    "signed": ("#5ac8fa", "#3a6f8f", "#2a3442", "#8f4a4a", "#ff6b6b"),
}


def cmap(name: str, t: float) -> str:
    """The colour at t (0 to 1) of a colour map."""
    stops = MAPS[name]
    t = min(1.0, max(0.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    a, b = stops[i], stops[i + 1]
    mix = [round(int(a[k:k + 2], 16) + (t - i) * (int(b[k:k + 2], 16) - int(a[k:k + 2], 16)))
           for k in (1, 3, 5)]
    return "#" + "".join(f"{v:02x}" for v in mix)


def scaled(value: float, bar: Dict[str, Any]) -> float:
    """Where a value falls on a colour bar, from 0 to 1."""
    lo, hi = bar["min"], bar["max"]
    if bar.get("log"):
        value, lo, hi = (math.log10(max(v, 1e-12)) for v in (value, lo, hi))
    if bar.get("map") == "signed" and bar.get("symlog"):
        f = lambda v: math.copysign(math.log10(1 + abs(v)), v)
        value, lo, hi = f(value), f(lo), f(hi)
    return (value - lo) / ((hi - lo) or 1)


def esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def width_of(text: str, size: float) -> float:
    """How wide a text is drawn: every character of a monospaced face is 0.6 em."""
    return len(text) * size * 0.6


def colour(series: Dict[str, Any], i: int) -> str:
    c = series.get("colour")
    if isinstance(c, int):
        return PALETTE[c % len(PALETTE)]
    return c or PALETTE[i % len(PALETTE)]


def f1(v: float) -> str:
    """A coordinate, to a tenth of a pixel."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

def nice_ticks(lo: float, hi: float, target: float) -> List[float]:
    """Round numbers between lo and hi, about `target` of them (as stats.js does)."""
    if not hi > lo:
        return [lo]
    rough = (hi - lo) / max(1.0, target)
    magnitude = 10 ** math.floor(math.log10(rough))
    scaled = rough / magnitude
    step = magnitude * (1 if scaled <= 1 else 2 if scaled <= 2 else 2.5 if scaled <= 2.5
                        else 5 if scaled <= 5 else 10)
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + step * 1e-6:
        out.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return out


class Axis:
    """
    One axis: its range in its own units (logarithms on a log axis), where
    its ticks go and how they read. A categorical axis puts one category at
    each whole number and names it.
    """

    def __init__(self, spec: Dict[str, Any], values: Sequence[float], target: float,
                 pad: bool = False) -> None:
        self.log = bool(spec.get("log"))
        self.categories = spec.get("categories")
        if self.categories:
            k = len(self.categories)
            self.lo, self.hi = -0.5, k - 0.5
            self.ticks = list(range(k))
            return
        finite = [s for s in (self.scale(v) for v in values) if s is not None]
        lo = self.scale(spec["min"]) if spec.get("min") is not None else (min(finite) if finite else 0.0)
        hi = self.scale(spec["max"]) if spec.get("max") is not None else (max(finite) if finite else 1.0)
        if not hi > lo:
            hi = lo + 1
        if pad:
            room = 0.04 * (hi - lo)
            if spec.get("max") is None:
                hi += room
            if spec.get("min") is None and not (not self.log and lo == 0):
                lo -= room
        self.lo, self.hi = lo, hi
        self.ticks = self._log_ticks() if self.log else nice_ticks(lo, hi, target)
        # As many decimals as the ticks need and no more: 0.25 apart wants two, 2 apart none.
        self.digits = max([next((d for d in range(5) if abs(round(t, d) - t) < 1e-9), 4)
                           for t in self.ticks] + [0])

    def scale(self, v: Any) -> Optional[float]:
        if v is None or isinstance(v, str):
            return None
        v = float(v)
        if not math.isfinite(v):
            return None
        if self.log:
            return math.log10(v) if v > 0 else None
        return v

    def _log_ticks(self) -> List[float]:
        """Powers of ten when the axis spans two or more; else 1, 2 and 5 times them."""
        decades = [float(p) for p in range(math.ceil(self.lo - 1e-9), math.floor(self.hi + 1e-9) + 1)]
        if len(decades) >= 2:
            return decades
        for steps in ((1, 2, 5), (1, 2, 3, 4, 5, 6, 7, 8, 9)):
            out = [math.log10(m * 10 ** p) for p in range(math.floor(self.lo) - 1, math.ceil(self.hi) + 1)
                   for m in steps]
            out = [t for t in out if self.lo - 1e-9 <= t <= self.hi + 1e-9]
            if len(out) >= 2:
                return out
        return [self.lo, self.hi]

    def label(self, t: float) -> str:
        if self.categories:
            return str(self.categories[int(round(t))])
        if self.log:
            v = 10 ** t
            if v >= 1 and abs(v - round(v)) < 1e-6 * v:
                return f"{int(round(v)):,}"
            return f"{v:.6f}".rstrip("0").rstrip(".")
        s = f"{t:,.{self.digits}f}"
        return "0" if s.strip("-0.,") == "" else s


# ---------------------------------------------------------------------------
# A chart
# ---------------------------------------------------------------------------

def legend_entries(fig: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """(label, colour, how it is marked) for every series that names itself."""
    if fig.get("legend") is False:
        return []
    if fig.get("kind") == "network":
        nodes = [(label, PALETTE[i % len(PALETTE)], "dot")
                 for i, label in enumerate((fig.get("colour") or {}).get("labels") or [])]
        edges = [(label, PALETTE[(i + 1) % len(PALETTE)], "line")
                 for i, label in enumerate(fig.get("edgeLabels") or [])]
        return nodes + edges
    out = []
    for i, s in enumerate(fig.get("series") or []):
        if not s.get("label"):
            continue
        mark = "box" if s.get("kind") in ("bars", "area") else "dot" if s.get("points") else "line"
        out.append((s["label"], colour(s, i), mark))
    return out


def legend_rows(fig: Dict[str, Any], width: float) -> List[List[Tuple[str, str, str]]]:
    """The legend, packed into rows that fit the chart's width."""
    rows: List[List[Tuple[str, str, str]]] = []
    used = width
    for entry in legend_entries(fig):
        need = 26 + width_of(entry[0], 11.5) + 14
        if not rows or used + need > width:
            rows.append([])
            used = 0
        rows[-1].append(entry)
        used += need
    return rows


def bar_rows(fig: Dict[str, Any]) -> int:
    """The legend rows a colour bar takes: its strip and its labels."""
    bar = fig.get("colourbar") or ((fig.get("colour") or {}) if (fig.get("colour") or {}).get("by") == "value" else None)
    return 3 if bar else 0


def wrap(text: str, room: float, size: float) -> List[str]:
    """A category's name on one line, or two if it does not fit."""
    if width_of(text, size) <= room or " " not in text:
        return [text]
    words = text.split(" ")
    best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
    return [" ".join(words[:best]), " ".join(words[best:])]


class Chart:
    """One chart drawn into the box (x, y, w, h) of a figure."""

    def __init__(self, fig: Dict[str, Any], x: float, y: float, w: float, h: float, uid: str) -> None:
        self.fig, self.uid = fig, uid
        self.box = (x, y, w, h)

    @staticmethod
    def height(fig: Dict[str, Any], w: float, small: bool) -> float:
        plot = fig.get("height") or ((440 if not small else 300) if fig.get("kind") == "network"
                                     else (330 if not small else 210))
        top = 30 if fig.get("title") else 12
        bottom = 12 if fig.get("kind") == "network" else 54 if fig["x"].get("label") else 36
        if fig.get("kind") != "network" and fig["x"].get("categories"):
            bottom += 12
        return top + plot + bottom + LEGEND_ROW * (len(legend_rows(fig, w - 70)) + bar_rows(fig))

    def svg(self) -> List[str]:
        if self.fig.get("kind") == "network":
            return self.network()
        return self.chart()

    # -- the frame ---------------------------------------------------------------

    def title(self, out: List[str]) -> None:
        x, y, _, _ = self.box
        if self.fig.get("title"):
            out.append(f'<text x="{f1(x + 14)}" y="{f1(y + 20)}" fill="{TEXT}" font-size="13.5" '
                       f'font-weight="600">{esc(self.fig["title"])}</text>')

    def legend(self, out: List[str], left: float, width: float) -> None:
        x, y, w, h = self.box
        rows = legend_rows(self.fig, width)
        for r, row in enumerate(rows):
            ly = y + h - 8 - (len(rows) - 1 - r) * LEGEND_ROW
            lx = left
            for label, c, mark in row:
                if mark == "line":
                    out.append(f'<line x1="{f1(lx)}" y1="{f1(ly - 4)}" x2="{f1(lx + 18)}" y2="{f1(ly - 4)}" '
                               f'stroke="{c}" stroke-width="2.5"/>')
                elif mark == "dot":
                    out.append(f'<circle cx="{f1(lx + 9)}" cy="{f1(ly - 4)}" r="4" fill="{c}"/>')
                else:
                    out.append(f'<rect x="{f1(lx + 3)}" y="{f1(ly - 9)}" width="11" height="10" fill="{c}" '
                               f'opacity="0.85"/>')
                out.append(f'<text x="{f1(lx + 24)}" y="{f1(ly)}" fill="{LABEL}" font-size="11.5">'
                           f'{esc(label)}</text>')
                lx += 26 + width_of(label, 11.5) + 14

    def colourbar(self, out: List[str], left: float, bar: Dict[str, Any]) -> None:
        """The strip that says what a colour means, under the chart: its name, the strip, its ends."""
        x, y, w, h = self.box
        top = y + h - 8 - LEGEND_ROW * (len(legend_rows(self.fig, w - 70)) + 3) + 4
        width = min(280.0, w - 60)
        label = bar.get("label", "")
        if label:
            out.append(f'<text x="{f1(left)}" y="{f1(top + 9)}" fill="{DIM}" font-size="11">'
                       f'{esc(label)}{" (logarithmic)" if bar.get("log") else ""}</text>')
        top += 15
        steps = 48
        for k in range(steps):
            out.append(f'<rect x="{f1(left + width * k / steps)}" y="{f1(top)}" width="{f1(width / steps + 0.6)}" '
                       f'height="9" fill="{cmap(bar.get("map", "viridis"), k / (steps - 1))}"/>')
        fmt = lambda v: f"{v:,.0f}" if abs(v) >= 10 or float(v).is_integer() else f"{v:.2g}"
        out.append(f'<text x="{f1(left)}" y="{f1(top + 22)}" fill="{LABEL}" font-size="11">{esc(fmt(bar["min"]))}</text>')
        out.append(f'<text x="{f1(left + width)}" y="{f1(top + 22)}" fill="{LABEL}" font-size="11" '
                   f'text-anchor="end">{esc(fmt(bar["max"]))}</text>')
        if bar.get("map") == "signed":
            out.append(f'<text x="{f1(left + width / 2)}" y="{f1(top + 22)}" fill="{LABEL}" font-size="11" '
                       'text-anchor="middle">0</text>')

    # -- a chart of lines, bands, points and bars --------------------------------

    def chart(self) -> List[str]:
        fig = self.fig
        bx, by, bw, bh = self.box
        series = fig.get("series") or []
        xs: List[float] = []
        ys: List[float] = []
        for s in series:
            if s.get("kind") == "bars":
                xs += list(s["x0"]) + list(s["x1"])
                ys += list(s["y"]) + list(s.get("y0") or [])
                if not fig["y"].get("log"):
                    ys.append(0)
                continue
            if s.get("kind") == "cells":
                xs += list(s["x0"]) + list(s["x1"])
                ys += list(s["y0"]) + list(s["y1"])
                continue
            xs += [v for v in s.get("x") or [] if v is not None]
            for key in ("y", "lo", "hi", "outerLo", "outerHi"):
                ys += [v for v in s.get(key) or [] if v is not None]

        rows = legend_rows(fig, bw - 70)
        top = by + (30 if fig.get("title") else 12)
        bottom = by + bh - LEGEND_ROW * (len(rows) + bar_rows(fig)) - (54 if fig["x"].get("label") else 36) \
            - (12 if fig["x"].get("categories") else 0)
        yaxis = Axis(fig["y"], ys, max(3.0, (bottom - top) / 48), pad=True)
        ylabels = [yaxis.label(t) for t in yaxis.ticks]
        left = bx + 24 + max([width_of(t, 11) for t in ylabels] + [10]) + 8
        right = bx + bw - 16
        w, h = right - left, bottom - top
        xaxis = Axis(fig["x"], xs, max(3.0, w / 95))

        def px(v: Any) -> Optional[float]:
            s = xaxis.scale(v)
            return None if s is None else left + (s - xaxis.lo) / (xaxis.hi - xaxis.lo) * w

        def py(v: Any) -> Optional[float]:
            s = yaxis.scale(v)
            return None if s is None else bottom - (s - yaxis.lo) / (yaxis.hi - yaxis.lo) * h

        out: List[str] = []
        self.title(out)
        # Grid and ticks.
        for t, text in zip(yaxis.ticks, ylabels):
            at = bottom - (t - yaxis.lo) / (yaxis.hi - yaxis.lo) * h
            if top - 0.5 <= at <= bottom + 0.5:
                out.append(f'<line x1="{f1(left)}" y1="{f1(at)}" x2="{f1(right)}" y2="{f1(at)}" '
                           'stroke="#fff" stroke-opacity="0.07"/>')
                out.append(f'<text x="{f1(left - 7)}" y="{f1(at + 4)}" fill="{LABEL}" font-size="11" '
                           f'text-anchor="end">{esc(text)}</text>')
        slot = w / max(1, len(xaxis.ticks))
        for t in xaxis.ticks:
            at = left + (t - xaxis.lo) / (xaxis.hi - xaxis.lo) * w
            if not left - 0.5 <= at <= right + 0.5:
                continue
            if not xaxis.categories:
                out.append(f'<line x1="{f1(at)}" y1="{f1(top)}" x2="{f1(at)}" y2="{f1(bottom)}" '
                           'stroke="#fff" stroke-opacity="0.07"/>')
            out.append(f'<line x1="{f1(at)}" y1="{f1(bottom)}" x2="{f1(at)}" y2="{f1(bottom + 4)}" '
                       'stroke="#fff" stroke-opacity="0.35"/>')
            lines = wrap(xaxis.label(t), slot - 6, 11) if xaxis.categories else [xaxis.label(t)]
            for k, text in enumerate(lines):
                half = width_of(text, 11) / 2
                cx = min(max(at, bx + 4 + half), bx + bw - 4 - half)
                out.append(f'<text x="{f1(cx)}" y="{f1(bottom + 17 + 13 * k)}" fill="{LABEL}" '
                           f'font-size="11" text-anchor="middle">{esc(text)}</text>')
        out.append(f'<path d="M{f1(left)} {f1(top)}V{f1(bottom)}H{f1(right)}" fill="none" '
                   'stroke="#fff" stroke-opacity="0.18"/>')
        # Axis names.
        if fig["x"].get("label"):
            low = bottom + 38 + (12 if xaxis.categories else 0)
            out.append(f'<text x="{f1(left + w / 2)}" y="{f1(low)}" fill="{DIM}" font-size="11.5" '
                       f'text-anchor="middle">{esc(fig["x"]["label"])}</text>')
        if fig["y"].get("label"):
            cy = top + h / 2
            # A long name on a short axis is set smaller, so that it stays beside its axis.
            size = min(11.5, max(8.0, 11.5 * h / max(1.0, width_of(fig["y"]["label"], 11.5))))
            out.append(f'<text transform="translate({f1(bx + 14)} {f1(cy)}) rotate(-90)" fill="{DIM}" '
                       f'font-size="{size:.1f}" text-anchor="middle">{esc(fig["y"]["label"])}</text>')

        clip = f"{self.uid}-clip"
        out.append(f'<clipPath id="{clip}"><rect x="{f1(left)}" y="{f1(top - 1)}" width="{f1(w)}" '
                   f'height="{f1(h + 2)}"/></clipPath><g clip-path="url(#{clip})">')

        def band(s: Dict[str, Any], lo_key: str, hi_key: str, alpha: float, c: str) -> None:
            if not s.get(lo_key) or not s.get(hi_key):
                return
            run: List[Tuple[float, float, float]] = []

            def flush() -> None:
                if len(run) > 1:
                    pts = [f"{f1(a)} {f1(b)}" for a, b, _ in run] + [f"{f1(a)} {f1(c2)}" for a, _, c2 in reversed(run)]
                    out.append(f'<path d="M{"L".join(pts)}Z" fill="{c}" fill-opacity="{alpha}"/>')
                run.clear()

            for t, a, b in zip(s["x"], s[lo_key], s[hi_key]):
                X, A, B = px(t), py(a), py(b)
                if X is None or A is None or B is None:
                    flush()
                else:
                    run.append((X, A, B))
            flush()

        for i, s in enumerate(series):
            c = colour(s, i)
            if s.get("kind") == "cells":
                bar = fig["colourbar"]
                for a, b, lo_, hi_, v in zip(s["x0"], s["x1"], s["y0"], s["y1"], s["value"]):
                    A, B, C, D = px(a), px(b), py(lo_), py(hi_)
                    if None in (A, B, C, D) or v is None:
                        continue
                    out.append(f'<rect x="{f1(min(A, B))}" y="{f1(min(C, D))}" width="{f1(abs(B - A) + 0.4)}" '
                               f'height="{f1(abs(D - C) + 0.4)}" fill="{cmap(bar.get("map", "viridis"), scaled(v, bar))}"/>')
            elif s.get("kind") == "area":
                band(s, "lo", "hi", s.get("alpha", 0.85), c)
            elif s.get("kind") == "bars":
                floor = bottom if fig["y"].get("log") else py(0)
                bars = []
                for j, (a, b, v) in enumerate(zip(s["x0"], s["x1"], s["y"])):
                    A, B, V = px(a), px(b), py(v)
                    base = py(s["y0"][j]) if s.get("y0") else floor
                    if A is None or B is None or V is None or base is None:
                        continue
                    bars.append(f'<rect x="{f1(min(A, B) + 0.5)}" y="{f1(min(V, base))}" '
                                f'width="{f1(max(1.0, abs(B - A) - 1))}" height="{f1(abs(base - V))}"/>')
                out.append(f'<g fill="{c}" fill-opacity="{s.get("alpha", 0.8)}">{"".join(bars)}</g>')
            else:
                band(s, "outerLo", "outerHi", 0.10, c)
                band(s, "lo", "hi", 0.22, c)
        for i, s in enumerate(series):
            if s.get("kind") in ("area", "bars", "cells") or not s.get("y"):
                continue
            c = colour(s, i)
            if s.get("points"):
                r = (s.get("size") or 3) / 2
                dots = [f'<circle cx="{f1(X)}" cy="{f1(Y)}" r="{r}"/>'
                        for X, Y in ((px(t), py(v)) for t, v in zip(s["x"], s["y"]))
                        if X is not None and Y is not None]
                out.append(f'<g fill="{c}" fill-opacity="{s.get("alpha", 0.7)}">{"".join(dots)}</g>')
                continue
            parts, pen = [], False
            for t, v in zip(s["x"], s["y"]):
                X, Y = px(t), py(v)
                if X is None or Y is None:
                    pen = False
                    continue
                parts.append(f'{"L" if pen else "M"}{f1(X)} {f1(Y)}')
                pen = True
            if parts:
                dash = f' stroke-dasharray="{" ".join(str(d) for d in s["dash"])}"' if s.get("dash") else ""
                alpha = f' stroke-opacity="{s["alpha"]}"' if s.get("alpha") is not None else ""
                out.append(f'<path d="{"".join(parts)}" fill="none" stroke="{c}" '
                           f'stroke-width="{s.get("width", 1.6)}" stroke-linejoin="round"{dash}{alpha}/>')
        out.append("</g>")

        for g in fig.get("guides") or []:
            if g.get("axis") == "y":
                at = py(g["at"])
                if at is None or not top <= at <= bottom:
                    continue
                out.append(f'<line x1="{f1(left)}" y1="{f1(at)}" x2="{f1(right)}" y2="{f1(at)}" '
                           f'stroke="{LABEL}" stroke-opacity="0.7" stroke-dasharray="5 4"/>')
                if g.get("label"):
                    out.append(f'<text x="{f1(right - 4)}" y="{f1(at - 5)}" fill="{LABEL}" font-size="11" '
                               f'text-anchor="end">{esc(g["label"])}</text>')
            else:
                at = px(g["at"])
                if at is None or not left <= at <= right:
                    continue
                out.append(f'<line x1="{f1(at)}" y1="{f1(top)}" x2="{f1(at)}" y2="{f1(bottom)}" '
                           f'stroke="{LABEL}" stroke-opacity="0.7" stroke-dasharray="5 4"/>')
                if g.get("label"):
                    out.append(f'<text x="{f1(at + 5)}" y="{f1(top + 12)}" fill="{LABEL}" '
                               f'font-size="11">{esc(g["label"])}</text>')
        self.legend(out, left, bw - 70)
        if fig.get("colourbar"):
            self.colourbar(out, left, fig["colourbar"])
        return out

    # -- a picture of a network --------------------------------------------------

    def network(self) -> List[str]:
        fig = self.fig
        bx, by, bw, bh = self.box
        rows = legend_rows(fig, bw - 70)
        top = by + (30 if fig.get("title") else 12)
        bottom = by + bh - 12 - LEGEND_ROW * (len(rows) + bar_rows(fig))
        left, right = bx + 14, bx + bw - 14
        w, h = right - left, bottom - top
        xs, ys = fig["nodes"]["x"], fig["nodes"]["y"]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        scale = min(w / ((x1 - x0) or 1), h / ((y1 - y0) or 1))
        ox = left + (w - (x1 - x0) * scale) / 2
        oy = top + (h - (y1 - y0) * scale) / 2
        X = [ox + (v - x0) * scale for v in xs]
        Y = [oy + (v - y0) * scale for v in ys]
        out: List[str] = []
        self.title(out)
        edge_group = fig.get("edgeGroup") or [0] * len(fig["edges"])
        for g in sorted(set(edge_group)):
            path = "".join(f"M{f1(X[a])} {f1(Y[a])}L{f1(X[b])} {f1(Y[b])}"
                           for (a, b), k in zip(fig["edges"], edge_group) if k == g)
            if g == 0:
                out.append(f'<path d="{path}" fill="none" stroke="{LABEL}" '
                           f'stroke-opacity="{fig.get("edgeAlpha", 0.35)}" stroke-width="0.6"/>')
            else:
                out.append(f'<path d="{path}" fill="none" stroke="{PALETTE[g % len(PALETTE)]}" '
                           f'stroke-opacity="0.9" stroke-width="{0.6 + 0.5 * g}"/>')
        r = fig.get("nodeSize", 2.4)
        c = fig.get("colour") or {}
        if c.get("by") == "value":
            values = fig["nodes"]["value"]
            # The extreme values are drawn last, on top, where they can be seen.
            order = sorted(range(len(xs)), key=lambda i: abs(scaled(values[i], c) - (0.5 if c.get("map") == "signed" else 0)))
            out.append("".join(f'<circle cx="{f1(X[i])}" cy="{f1(Y[i])}" r="{r}" '
                               f'fill="{cmap(c.get("map", "viridis"), scaled(values[i], c))}"/>' for i in order))
        else:
            groups = fig["nodes"].get("group") or [0] * len(xs)
            for g in sorted(set(groups)):
                dots = "".join(f'<circle cx="{f1(X[i])}" cy="{f1(Y[i])}" r="{r}"/>'
                               for i in range(len(xs)) if groups[i] == g)
                out.append(f'<g fill="{PALETTE[g % len(PALETTE)]}">{dots}</g>')
        self.legend(out, left, bw - 70)
        if c.get("by") == "value":
            self.colourbar(out, left, c)
        return out


# ---------------------------------------------------------------------------
# A figure
# ---------------------------------------------------------------------------

def render(charts: List[Dict[str, Any]], columns: int = 1, name: str = "figure") -> str:
    """
    One figure: its charts in a grid of `columns`, row by row, on one card.
    `name` makes its ids its own, so that several figures inlined in one page
    do not clip each other's charts.
    """
    prefix = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "figure"
    columns = max(1, min(columns, len(charts)))
    w = WIDTH / columns
    small = columns > 1
    body: List[str] = []
    y = 4.0
    for r in range(0, len(charts), columns):
        row = charts[r:r + columns]
        h = max(Chart.height(fig, w, small) for fig in row)
        for c, fig in enumerate(row):
            body += Chart(fig, c * w, y, w, h, f"{prefix}-{r + c}").svg()
        y += h
    height = y + 4
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {f1(height)}" '
            f'width="{WIDTH}" height="{f1(height)}" font-family="{FONT}">'
            f'<rect width="{WIDTH}" height="{f1(height)}" rx="10" fill="{CARD}"/>'
            + "".join(body) + "</svg>\n")
