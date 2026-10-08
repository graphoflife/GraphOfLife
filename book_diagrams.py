#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The book's diagrams, drawn as SVG.

    python3 book_diagrams.py        writes book/diagrams/*.svg

The diagrams are drawn here rather than by hand so that their geometry is
exact — the starting ring is a real Watts–Strogatz ring, the worked examples
add up — and so that any of them can be drawn again. They are images in the
chapters, read alike in the Book tab, in Obsidian and on GitHub, so each is
drawn on a card of the page's own dark colours (those of book_svg.py) and
holds nothing but shapes and text.
"""
from __future__ import annotations

import math
import os
import re
from typing import List, Tuple

import networkx as nx

from book_svg import CARD

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "book", "diagrams")

STYLE = """<style>
.dg-t{fill:var(--text,#eef4fa);font:11.5px 'JetBrains Mono',ui-monospace,Menlo,Consolas,'DejaVu Sans Mono',monospace}
.dg-b{fill:var(--text,#eef4fa);font:600 11.5px 'JetBrains Mono',ui-monospace,Menlo,Consolas,'DejaVu Sans Mono',monospace}
.dg-h{fill:var(--text,#eef4fa);font:600 13.5px 'JetBrains Mono',ui-monospace,Menlo,Consolas,'DejaVu Sans Mono',monospace}
.dg-m{fill:var(--muted,#9ab0c3);font:10.5px 'JetBrains Mono',ui-monospace,Menlo,Consolas,'DejaVu Sans Mono',monospace}
.dg-s{fill:var(--muted,#9ab0c3);font:9.5px 'JetBrains Mono',ui-monospace,Menlo,Consolas,'DejaVu Sans Mono',monospace}
.dg-box{fill:var(--panel-2,#2d3d50);stroke:var(--border-strong,#5a6e86);stroke-width:1}
.dg-box2{fill:var(--panel-3,#384c64);stroke:var(--border-strong,#5a6e86);stroke-width:1}
.dg-frame{fill:none;stroke:var(--border,#435468);stroke-width:1;stroke-dasharray:4 3}
.dg-line{stroke:var(--muted,#9ab0c3);stroke-width:1.4;fill:none}
.dg-faint{stroke:var(--muted-2,#7e94a5);stroke-width:1;fill:none;opacity:.6}
.dg-acc{stroke:var(--accent,#64bdff);stroke-width:2;fill:none}
.dg-warn{stroke:#ffd166;stroke-width:2;fill:none}
.dg-bad{stroke:#ff6b6b;stroke-width:2;fill:none}
.dg-good{stroke:#7ee787;stroke-width:2;fill:none}
.dg-node{fill:var(--accent,#64bdff)}
.dg-node2{fill:#ffd166}
.dg-node3{fill:#7ee787}
.dg-node4{fill:#c792ea}
.dg-dead{fill:#ff6b6b}
.dg-grey{fill:var(--muted-2,#7e94a5)}
.dg-fill{fill:var(--accent,#64bdff);opacity:.85}
.dg-fill2{fill:#ffd166;opacity:.85}
.dg-fill3{fill:#c792ea;opacity:.85}
</style>
<defs><marker id="dg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted,#9ab0c3)"/></marker></defs>"""


def svg(width: int, height: int, body: str, title: str) -> str:
    # The colours were written as the page's CSS variables with these as
    # fallbacks; an SVG shown as an image sees no variables, so the fallbacks
    # are what it is drawn in.
    style = re.sub(r"var\(--[\w-]+,(#[0-9a-fA-F]+)\)", r"\1", STYLE)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="{title}">'
            f'<title>{title}</title>{style}<rect width="{width}" height="{height}" rx="10" '
            f'fill="{CARD}"/>{body}</svg>\n')


def text(x: float, y: float, s: str, cls: str = "dg-t", anchor: str = "start") -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lead = len(s) - len(s.lstrip(" "))
    s = "\u00a0" * lead + s[lead:]
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>'


def box(x: float, y: float, w: float, h: float, cls: str = "dg-box", r: int = 6) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" class="{cls}"/>'


def arrow(x1: float, y1: float, x2: float, y2: float, cls: str = "dg-line") -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" '
            f'marker-end="url(#dg-arrow)"/>')


def seg(x1: float, y1: float, x2: float, y2: float, cls: str = "dg-line") -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>'


def dot(x: float, y: float, r: float = 6, cls: str = "dg-node") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}"/>'


def lines(x: float, y: float, rows: List[str], cls: str = "dg-t", gap: int = 18) -> str:
    return "".join(text(x, y + i * gap, r, cls) for i, r in enumerate(rows))


# ---------------------------------------------------------------------------

def ring() -> str:
    """The founders' starting ring: a lattice, and the same lattice rewired."""
    n, k, p = 16, 4, 0.2
    lattice = nx.watts_strogatz_graph(n, k, 0.0, seed=3)
    rewired = nx.watts_strogatz_graph(n, k, p, seed=3)
    body = []
    for cx, graph, title, sub in ((190, lattice, "1. A ring lattice", "16 founders, each joined to the 2 nearest on either side"),
                                  (570, rewired, "2. The same ring, rewired", "each connection moved with probability p = 0.2")):
        cy, r = 200, 130
        pos = {i: (cx + r * math.sin(2 * math.pi * i / n), cy - r * math.cos(2 * math.pi * i / n)) for i in range(n)}
        for a, b in graph.edges():
            on_ring = min((a - b) % n, (b - a) % n) <= k // 2
            body.append(seg(*pos[a], *pos[b], "dg-line" if on_ring else "dg-warn"))
        for i in range(n):
            body.append(dot(*pos[i], 7))
        body.append(text(cx, 34, title, "dg-h", "middle"))
        body.append(text(cx, 52, sub, "dg-m", "middle"))
    moved = sum(1 for a, b in rewired.edges() if min((a - b) % n, (b - a) % n) > k // 2)
    body.append(text(380, 366, f"Yellow: the {moved} connections that were moved to a founder chosen at random —",
                     "dg-m", "middle"))
    body.append(text(380, 382, "the shortcuts that make the ring a small world.", "dg-m", "middle"))
    return svg(760, 396, "".join(body), "The founders' starting ring")


def one_iteration() -> str:
    """The two phases of one iteration, in the order the engine runs them."""
    body = []
    col = [(20, "Phase 1 · Reproduction", [
        "1. Messages: every agent looks at itself and its neighbours",
        "   and writes a message to each (the pre-pass).",
        "2. Every agent with at least one token looks again and decides:",
        "   • the share of its tokens to give a child,",
        "   • which of itself and its neighbours the child joins,",
        "   • which of its own connections it hands to the child.",
        "3. Children are born: a copy of the parent's brain,",
        "   changed with probability 0.2, and the tokens given.",
        "4. Handed-over connections move from parent to child.",
        "5. Messages written in step 2 are delivered.",
        "6. Cleanup: agents with 0 tokens die; everything not joined",
        "   to the largest connected piece dies; their tokens are",
        "   shared out at random among the survivors.",
        "→ frame 2t is recorded",
    ]), (500, "Phase 2 · The game", [
        "1. Messages: the pre-pass again.",
        "2. Every agent looks once more, writes its messages, and",
        "   stakes all of its tokens on itself and its neighbours,",
        "   marking part of each stake as revolutionary.",
        "3. Every node goes to its largest staker — or to a",
        "   coalition of smaller revolutionary stakers that",
        "   outweighs it. The node takes on the winner's brain",
        "   and holds every token staked on it.",
        "4. Connections that carried no tokens in this game are cut.",
        "5. Messages are delivered.",
        "6. Cleanup, as in phase 1.",
        "7. Every brain changes, each with probability 0.2.",
        "→ frame 2t + 1 is recorded",
    ])]
    for x, title, rows in col:
        body.append(box(x, 40, 440, 330, "dg-box"))
        body.append(text(x + 14, 64, title, "dg-h"))
        body.append(lines(x + 14, 92, rows, "dg-t", 19))
    body.append(arrow(470, 205, 498, 205, "dg-acc"))
    body.append(f'<path d="M720,372 C720,412 250,412 250,374" class="dg-acc" marker-end="url(#dg-arrow)"/>')
    body.append(text(485, 428, "iteration t + 1 begins", "dg-m", "middle"))
    body.append(text(485, 24, "One iteration t", "dg-h", "middle"))
    return svg(970, 440, "".join(body), "One iteration: reproduction, then the game")


def reproduction() -> str:
    """A worked birth, with numbers that follow the rules exactly."""
    body = []
    body.append(text(20, 26, "A birth, worked through", "dg-h"))
    # before
    P, A_, B, C = (150, 170), (60, 90), (240, 90), (150, 290)
    body.append(text(150, 58, "Before", "dg-b", "middle"))
    for q in (A_, B, C):
        body.append(seg(*P, *q))
    for q, name, tok in ((P, "P", 17), (A_, "A", 5), (B, "B", 30), (C, "C", 8)):
        body.append(dot(*q, 16, "dg-node"))
        body.append(text(q[0], q[1] + 5, name, "dg-b", "middle"))
        body.append(text(q[0] + 22, q[1] - 12, f"{tok}", "dg-m"))
    # decisions
    rows = [
        "P holds 17 tokens. Its brain looks at P, A, B and C (4 columns).",
        "Child's share: the two reproduction outputs, averaged over the",
        "4 columns, are (0.8, 1.2): share = 0.8 / (0.8 + 1.2) = 0.4,",
        "so the child gets ⌊0.4 · 17⌋ = ⌊6.8⌋ = 6 tokens; P keeps 11.",
        "Links, one yes/no per column: P yes, A no, B yes, C no.",
        "Handover, one yes/no per neighbour: A no, B no, C yes —",
        "P's connection to C moves to the child.",
        "The child's brain is a copy of P's, changed with probability 0.2.",
    ]
    body.append(lines(300, 74, rows, "dg-t", 21))
    # after
    ox = 470
    P2, A2, B2, C2, K = (ox + 90, 400), (ox + 0, 330), (ox + 180, 330), (ox + 90, 520), (ox + 210, 470)
    body.append(text(ox + 90, 300, "After", "dg-b", "middle"))
    for a, b in ((P2, A2), (P2, B2), (P2, K), (K, B2), (K, C2)):
        body.append(seg(*a, *b))
    for q, name, tok, cls in ((P2, "P", 11, "dg-node"), (A2, "A", 5, "dg-node"), (B2, "B", 30, "dg-node"),
                              (C2, "C", 8, "dg-node"), (K, "c", 6, "dg-node2")):
        body.append(dot(*q, 16, cls))
        body.append(text(q[0], q[1] + 5, name, "dg-b", "middle"))
        body.append(text(q[0] + 22, q[1] - 12, f"{tok}", "dg-m"))
    body.append(text(20, 560, "Tokens before: 17 + 5 + 30 + 8 = 60. After: 11 + 6 + 5 + 30 + 8 = 60. "
                              "A birth moves tokens; it never makes them.", "dg-m"))
    body.append(text(20, 345, "The new connections P–c and c–B must carry", "dg-m"))
    body.append(text(20, 362, "tokens in the coming game, or they are cut", "dg-m"))
    body.append(text(20, 379, "at its end (as is every other connection).", "dg-m"))
    return svg(760, 580, "".join(body), "A birth, worked through")


def game() -> str:
    """A worked game on one node, with a coalition that wins it."""
    body = [text(20, 26, "One node in the game, worked through", "dg-h")]
    V, U, W, X = (380, 190), (190, 120), (560, 120), (380, 330)
    for q in (U, W, X):
        body.append(seg(*V, *q))
    body.append(seg(*W, *X, "dg-faint"))
    stakes = [(U, "U", "stakes 7 on V", "none of it revolutionary"),
              (W, "W", "stakes 4 on V", "all 4 revolutionary"),
              (X, "X", "stakes 3 on V", "all 3 revolutionary"),
              (V, "V", "stakes 2 on itself", "1 of them revolutionary")]
    for q, name, s1, s2 in stakes:
        body.append(dot(*q, 17, "dg-node" if name != "W" else "dg-node2"))
        body.append(text(q[0], q[1] + 5, name, "dg-b", "middle"))
        dx = {"U": -170, "W": 30, "X": 30, "V": 30}[name]
        dy = 52 if name == "V" else 0       # below V, clear of its line to W
        body.append(text(q[0] + dx, q[1] - 28 + dy, s1, "dg-t"))
        body.append(text(q[0] + dx, q[1] - 12 + dy, s2, "dg-m"))
    rows = [
        "Who takes node V?",
        "The largest single stake is U's: H = 7. U is the hegemon.",
        "The revolutionaries other than U: V (1), X (3), W (4).",
        "Their revolutionary tokens add up to R = 1 + 3 + 4 = 8.",
        "R > H, so the coalition wins. Which of them? Add them up,",
        "smallest first, until the sum passes (R + H) / 2 = 7.5:",
        "   1 (V) → 4 (V, X) → 8 (V, X, W): W tips it. W wins V.",
        "",
        "V keeps its place and its connections, takes on a copy of",
        "W's brain, and holds every token staked on it:",
        "   7 + 4 + 3 + 2 = 16 tokens.",
    ]
    body.append(lines(20, 410, rows, "dg-t", 20))
    return svg(760, 640, "".join(body), "One node in the game, worked through")


def revolution() -> str:
    """The coalition rule as a picture: a weighted median."""
    body = [text(20, 26, "The coalition rule: who wins a node", "dg-h")]
    scale = 50
    x0, y = 40, 70
    body.append(text(x0, y - 8, "Revolutionary stakes, smallest first (R = 8):", "dg-m"))
    for amount, name, cls in ((1, "V", "dg-fill"), (3, "X", "dg-fill"), (4, "W", "dg-fill2")):
        w = amount * scale
        body.append(f'<rect x="{x0}" y="{y}" width="{w}" height="34" class="{cls}"/>')
        body.append(text(x0 + w / 2, y + 22, f"{name}: {amount}", "dg-b", "middle"))
        x0 += w
    y2 = 150
    body.append(text(40, y2 - 8, "The hegemon's whole stake (H = 7):", "dg-m"))
    body.append(f'<rect x="40" y="{y2}" width="{7 * scale}" height="34" class="dg-fill3"/>')
    body.append(text(40 + 3.5 * scale, y2 + 22, "U: 7", "dg-b", "middle"))
    mid = 40 + 7.5 * scale
    body.append(seg(mid, 50, mid, 205, "dg-bad"))
    body.append(text(mid + 6, 220, "(R + H) / 2 = 7.5", "dg-m"))
    rows = [
        "A coalition wins exactly when R > H. The winner is the revolutionary in whose",
        "stretch the line (R + H) / 2 falls — here W. Stakers with equal amounts form",
        "one step, and the winner is drawn among them at random.",
    ]
    body.append(lines(40, 255, rows, "dg-t", 19))
    return svg(760, 320, "".join(body), "The coalition rule")


def cleanup() -> str:
    """What the cleanup removes, and where its tokens go."""
    body = [text(20, 26, "Cleanup, worked through", "dg-h")]
    pos = {"a": (90, 130), "b": (190, 90), "c": (190, 180), "d": (290, 130), "e": (390, 90),
           "f": (490, 140), "g": (590, 110), "h": (590, 200)}
    edges = [("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("c", "d"), ("d", "e"), ("e", "f"),
             ("g", "h"), ("f", "g")]
    tokens = {"a": 4, "b": 9, "c": 2, "d": 6, "e": 0, "f": 3, "g": 5, "h": 1}
    for a, b in edges:
        body.append(seg(*pos[a], *pos[b]))
    for n, (x, y) in pos.items():
        cls = "dg-dead" if n in "efgh" else "dg-node"
        body.append(dot(x, y, 15, cls))
        body.append(text(x, y + 5, n, "dg-b", "middle"))
        body.append(text(x + 18, y - 14, str(tokens[n]), "dg-m"))
    body.append(f'<rect x="465" y="70" width="160" height="160" rx="10" class="dg-frame"/>')
    rows = [
        "e holds 0 tokens: it starves.",
        "Without e, the world is in two pieces: {a, b, c, d} and {f, g, h}.",
        "Only the largest piece lives on, so f, g and h are cut off.",
        "Their 3 + 5 + 1 = 9 tokens are shared out among a, b, c and d:",
        "each of the 9 goes to one of the 4 survivors, each equally likely",
        "(a multinomial draw). The 4 survivors held 21; now they hold 30 — all",
        "the tokens there are.",
    ]
    body.append(lines(20, 260, rows, "dg-t", 20))
    return svg(760, 410, "".join(body), "Cleanup, worked through")


def brain() -> str:
    """The brain: what it reads, its layers, and what each output means."""
    body = [text(20, 26, "The brain of B1, looking at one candidate", "dg-h")]
    inputs = [("1", "is the candidate the agent itself? (1 or 0)"),
              ("4", "log(1 + tokens) and log(1 + degree),"),
              ("", "   of the agent and of the candidate"),
              ("24", "6 quantiles of log tokens and of log degree"),
              ("", "   over the two neighbourhoods"),
              ("120", "4 messages of 30 numbers: agent→agent,"),
              ("", "   agent→candidate, candidate→agent,"),
              ("", "   candidate→candidate"),
              ("5", "random numbers, uniform between −2 and 2")]
    body.append(box(20, 44, 330, 236, "dg-box"))
    body.append(text(32, 64, "154 inputs", "dg-b"))
    for i, (n, what) in enumerate(inputs):
        y = 90 + i * 21
        body.append(text(34, y, n, "dg-b"))
        body.append(text(70, y, what, "dg-m"))
    layers = [50, 45, 40, 35, 30]
    x = 384
    for i, width in enumerate(layers):
        h = width * 3
        body.append(box(x, 162 - h / 2, 30, h, "dg-box2", 4))
        body.append(text(x + 15, 162 + h / 2 + 16, str(width), "dg-m", "middle"))
        if i:
            body.append(arrow(x - 16, 162, x - 2, 162))
        x += 46
    body.append(arrow(352, 162, 382, 162))
    body.append(text(475, 270, "5 hidden layers, sigmoid", "dg-m", "middle"))
    body.append(arrow(x - 14, 162, x + 6, 162))
    outputs = [("2", "the child's share of the agent's tokens"),
               ("2", "join the child to this candidate? (yes, no)"),
               ("2", "… read as a probability, or take the larger?"),
               ("1", "stake score for this candidate"),
               ("2", "spread the stake by score, or all on the best?"),
               ("2", "revolutionary share of this stake"),
               ("2", "hand this connection to the child? (yes, no)"),
               ("2", "… read as a probability, or take the larger?"),
               ("30", "the message to this candidate (through tanh)")]
    bx = x + 10
    body.append(box(bx, 44, 980 - bx - 10, 236, "dg-box"))
    body.append(text(bx + 12, 64, "45 outputs (linear)", "dg-b"))
    for i, (n, what) in enumerate(outputs):
        yy = 90 + i * 21
        body.append(text(bx + 14, yy, n, "dg-b"))
        body.append(text(bx + 44, yy, what, "dg-m"))
    rows = ["The same network reads every candidate — the agent itself and each neighbour — as one column of",
            "154 numbers, and gives one column of 45 outputs per candidate. 15,795 weights and biases in all."]
    body.append(lines(20, 308, rows, "dg-m", 18))
    return svg(980, 340, "".join(body), "The brain of B1")


def gini() -> str:
    """The Lorenz curve and the Gini coefficient, on four agents."""
    body = [text(20, 26, "Four agents, their Lorenz curve, and their Gini coefficient", "dg-h")]
    x0, y0, size = 60, 330, 260
    body.append(seg(x0, y0, x0 + size, y0))
    body.append(seg(x0, y0, x0, y0 - size))
    body.append(seg(x0, y0, x0 + size, y0 - size, "dg-faint"))
    tokens = [1, 2, 3, 10]
    total = sum(tokens)
    cum = [0]
    for t in tokens:
        cum.append(cum[-1] + t)
    pts = [(x0 + size * i / 4, y0 - size * c / total) for i, c in enumerate(cum)]
    area = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + f" {x0 + size},{y0}"
    body.append(f'<polygon points="{x0},{y0} {area}" style="fill:var(--accent,#64bdff);opacity:.18"/>')
    body.append('<polyline points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + '" class="dg-acc"/>')
    for px, py in pts:
        body.append(dot(px, py, 3.5, "dg-node"))
    for i in range(5):
        body.append(text(x0 + size * i / 4, y0 + 18, f"{i}/4", "dg-s", "middle"))
    for c in cum:
        body.append(text(x0 - 8, y0 - size * c / total + 4, f"{c}/16", "dg-s", "end"))
    body.append(text(x0 + size / 2, y0 + 38, "share of agents, poorest first", "dg-m", "middle"))
    body.append(text(x0 + 0.72 * size - 12, y0 - 0.72 * size, "perfect equality", "dg-s", "end"))
    rows = ["Tokens: 1, 2, 3 and 10 (16 in all).",
            "Lorenz curve: the share of all tokens held",
            "by the poorest share of agents.",
            "",
            "Gini = (Σ (2i − n − 1) · xᵢ) / (n · Σ xᵢ),",
            "with the xᵢ sorted from smallest (i = 1):",
            "(−3·1 − 1·2 + 1·3 + 3·10) / (4 · 16)",
            "= 28 / 64 = 0.4375.",
            "",
            "The same number is 1 − 2 × (the area under",
            "the Lorenz curve) = 1 − 2 × 0.28125.",
            "0 means everyone holds the same; (n − 1)/n,",
            "here 0.75, that one agent holds everything."]
    body.append(lines(380, 70, rows, "dg-t", 21))
    return svg(760, 380, "".join(body), "The Lorenz curve and the Gini coefficient")


def reproducibility() -> str:
    """The four ways Experiment 1 runs each seed."""
    body = [text(20, 26, "Experiment 1: each seed run four ways, to iteration 30", "dg-h")]
    x0, x1 = 230, 720
    sx = lambda it: x0 + (x1 - x0) * it / 30
    rows = [("straight", [(0, 30, "dg-acc")], [], "one process, start to finish (the reference)"),
            ("stopped at 10", [(0, 10, "dg-acc"), (10, 30, "dg-good")], [(10, "checkpoint · new process")], ""),
            ("cut off at 15", [(0, 15, "dg-acc"), (10, 30, "dg-good")], [(10, "last checkpoint"), (15, "killed")], ""),
            ("all BLAS threads", [(0, 30, "dg-warn")], [], "the matrix library on every core")]
    for i, (name, spans, marks, note) in enumerate(rows):
        y = 70 + i * 70
        body.append(text(20, y + 5, name, "dg-b"))
        for a, b, cls in spans:
            yy = y + (12 if cls == "dg-good" and a < 15 and name == "cut off at 15" else 0)
            body.append(seg(sx(a), yy, sx(b), yy, cls))
        for at, label in marks:
            cls = "dg-dead" if label == "killed" else "dg-node"
            body.append(dot(sx(at), y, 5, cls))
            body.append(text(sx(at), y - 12, label, "dg-s", "middle"))
        if note:
            body.append(text(x0, y + 26, note, "dg-s"))
    for it in (0, 10, 15, 20, 30):
        body.append(text(sx(it), 350, str(it), "dg-s", "middle"))
    body.append(text((x0 + x1) / 2, 368, "iteration", "dg-m", "middle"))
    body.append(text(20, 396, "Every variant is compared with the straight run of its seed: every frame, every row of",
                     "dg-m"))
    body.append(text(20, 413, "statistics, and every array of the last checkpoint.", "dg-m"))
    return svg(760, 425, "".join(body), "Experiment 1: four ways to run each seed")


DIAGRAMS = {"ring": ring, "one-iteration": one_iteration, "reproduction": reproduction,
            "game": game, "revolution": revolution, "cleanup": cleanup, "brain": brain,
            "gini": gini, "reproducibility": reproducibility}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, draw in DIAGRAMS.items():
        with open(os.path.join(OUT, f"{name}.svg"), "w") as f:
            f.write(draw())
        print(f"book/diagrams/{name}.svg")


if __name__ == "__main__":
    main()
