#!/usr/bin/env python3
"""
make_mark.py -- draw the GraphOfLife mark, and put it where it belongs.

Six agents, each one an eye, and only some of them joined: the middle reaches
three of the five and the other two hang off their neighbours, so it reads as a
graph that is not a complete one. The middle one looks straight out; the rest
look around.

The eyes are generated rather than drawn by hand. Each is an iris disc with
spokes running out from a round pupil, and there are six of them, which is
around fifty shapes — enough that keeping the header and the favicon identical
by editing both is a losing game. This writes both from the same numbers:

    python3 tools/make_mark.py

The header copy takes its colour from the page, so one definition is accent
blue in the top bar and white over the front page's backdrop. The favicon has
no page to inherit from, so it carries its own colours and a dark rounded
background to sit on either a light or a dark tab bar.
"""

from __future__ import annotations

import math
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(HERE, "web", "index.html")
FAVICON = os.path.join(HERE, "web", "favicon.svg")

PUPIL_INK = "#05070a"

# Where each agent sits, how big it is, and which way it is looking. The gaze
# is a direction, scaled below — the iris and pupil travel together, because
# that is what an eye does.
AGENTS = [
    # x,     y,      r,    gaze x, gaze y
    (16.00, 16.00, 5.20, 0.00, 0.00),      # the middle one looks straight out
    (16.00, 5.70, 3.40, 0.00, 1.00),
    (25.80, 12.82, 3.40, -1.00, 0.16),
    (22.05, 24.33, 3.40, -0.72, -0.69),
    (9.95, 24.33, 3.40, 0.72, -0.69),
    (6.20, 12.82, 3.40, 1.00, -0.04),
]

# Joined so the links draw a G: out from the middle to the agent at five
# o'clock, then round the ring the long way and stopping short of closing it.
# That leaves the gap on the right, between two o'clock and five, which is a
# G's opening — and the spur from the middle, arriving at the lower end of the
# arc, is its bar. Going the other way round puts the gap at the bottom, which
# reads as a Q. Five links over six agents, so it is a path: no cycle in it.
LINKS = [
    ((16.00, 16.00), (22.05, 24.33)),      # the bar, out to five o'clock
    ((22.05, 24.33), (9.95, 24.33)),       # five to seven
    ((9.95, 24.33), (6.20, 12.82)),        # seven to ten
    ((6.20, 12.82), (16.00, 5.70)),        # ten to twelve
    ((16.00, 5.70), (25.80, 12.82)),       # twelve to two, and stop
]

IRIS_SHARE = 0.60      # of the agent's radius
PUPIL_SHARE = 0.31
GAZE_SHARE = 0.15      # how far the iris slides towards what it is looking at
SPOKES = 9


def eye(x: float, y: float, r: float, gx: float, gy: float) -> list[str]:
    """One eye: an iris, spokes running out from the centre, and a pupil."""
    cx = x + gx * r * GAZE_SHARE
    cy = y + gy * r * GAZE_SHARE
    iris = r * IRIS_SHARE
    pupil = r * PUPIL_SHARE

    out = [f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{iris:.2f}" '
           f'fill="{PUPIL_INK}" opacity="0.34"/>']

    # Spokes, so the iris has some structure to it rather than being a flat
    # disc. Started clear of the pupil and stopped short of the rim.
    inner, outer = pupil * 1.12, iris * 0.94
    spokes = []
    for k in range(SPOKES):
        angle = (k / SPOKES) * math.tau + 0.2
        spokes.append(
            f'M{cx + math.cos(angle) * inner:.2f} {cy + math.sin(angle) * inner:.2f}'
            f'L{cx + math.cos(angle) * outer:.2f} {cy + math.sin(angle) * outer:.2f}')
    out.append(f'<path d="{" ".join(spokes)}" stroke="{PUPIL_INK}" '
               f'stroke-width="{r * 0.075:.2f}" opacity="0.42" fill="none"/>')

    out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{pupil:.2f}" fill="{PUPIL_INK}"/>')
    return out


def mark(colour: str, indent: str) -> str:
    """The whole mark, in whatever colour it is asked for."""
    pad = "\n" + indent
    links = " ".join(f"M{a[0]:.2f} {a[1]:.2f}L{b[0]:.2f} {b[1]:.2f}" for a, b in LINKS)

    parts = [
        f'<g stroke="{colour}" stroke-width="1.4" fill="none" '
        f'stroke-linecap="round" opacity="0.5">',
        f'  <path d="{links}"/>',
        '</g>',
        f'<g fill="{colour}">',
    ]
    for x, y, r, _, _ in AGENTS:
        parts.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}"/>')
    parts.append('</g>')
    for agent in AGENTS:
        parts.extend(eye(*agent))
    return pad.join(parts)


def main() -> None:
    symbol = (
        '  <symbol id="gol-mark" viewBox="0 0 32 32">\n'
        '    ' + mark("currentColor", "    ") + '\n'
        '  </symbol>'
    )
    html = open(INDEX).read()
    replaced, count = re.subn(
        r'  <symbol id="gol-mark" viewBox="0 0 32 32">.*?</symbol>',
        lambda _: symbol, html, count=1, flags=re.S)
    assert count == 1, "could not find the symbol in index.html"
    open(INDEX, "w").write(replaced)

    favicon = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">\n'
        '  <!-- Written by tools/make_mark.py. A favicon has no page to inherit\n'
        '       colours from, and has to hold up against a light tab bar as well as\n'
        '       a dark one, so it brings its own background. -->\n'
        '  <rect width="32" height="32" rx="7" fill="#0d1117"/>\n'
        '  ' + mark("#8fa8e8", "  ") + '\n'
        '</svg>\n'
    )
    open(FAVICON, "w").write(favicon)

    print(f"wrote the mark into {INDEX}")
    print(f"wrote {FAVICON}")
    print("  now: inkscape --export-type=png -w 32  -h 32  web/favicon.svg")
    print("       inkscape --export-type=png -w 180 -h 180 web/favicon-180.png")


if __name__ == "__main__":
    main()
