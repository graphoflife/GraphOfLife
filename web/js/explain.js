/*
 * The algorithm, one panel at a time.
 *
 * Each panel is a short looping animation of a single step, drawn procedurally
 * rather than recorded, so it stays small, sharp at any size, and can be
 * corrected when the algorithm changes — a folder of GIFs would be stale the
 * first time a rule moved.
 *
 * Every step is shown twice. On the left it happens to one agent, large enough
 * to follow; on the right the same rule runs across a whole small world at
 * once, which is what the mechanic actually looks like and is usually the
 * surprising part. Both are the same draw function over a different graph.
 *
 * Positions come from a force layout — repulsion between every pair, springs
 * along the edges — rather than from coordinates written by hand, so the
 * pictures space themselves however the graph happens to be shaped. The layout
 * covers every node and edge the animation will ever show, including ones that
 * appear halfway through, so nothing jumps as it arrives and an edge is never
 * left hanging off a node that is not there: drawGraph clamps an edge's opacity
 * to the fainter of its two endpoints, which makes that class of glitch
 * impossible rather than merely unlikely.
 *
 * The order here is the order the engine does things in, including the parts
 * that are easy to leave out of a description: messages ride along with
 * observation rather than happening separately, cleanup runs after both phases
 * rather than once, and every brain in the world mutates each iteration, not
 * only the newborns.
 */

const Explain = {
  ink: {
    bg: '#0d1117',
    edge: 'rgba(190, 200, 215, 0.42)',
    node: '#8fa8e8',
    rich: '#f0a878',
    pale: '#e8eef8',
    dim: 'rgba(190, 200, 215, 0.18)',
    good: '#7fd4a0',
    warn: '#e8896b',
    eye: '#05070a',
    text: '#8b9bab'
  },

  // ---- deterministic randomness ----------------------------------------
  //
  // Panels are laid out and cast from a seed, so a picture is the same every
  // time it is drawn and can be checked by drawing it.

  rng(seed) {
    let s = (seed >>> 0) || 1;
    return () => {
      s ^= s << 13; s >>>= 0;
      s ^= s >>> 17;
      s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  },

  /** A stable pick of `count` distinct items. */
  choose(items, count, seed) {
    const rand = this.rng(seed);
    const pool = items.slice();
    const out = [];
    while (out.length < count && pool.length) {
      out.push(pool.splice(Math.floor(rand() * pool.length), 1)[0]);
    }
    return out;
  },

  // ---- graphs ----------------------------------------------------------

  /**
   * A ring of `n` agents joined to their `k` nearest, with some links moved
   * elsewhere — the same small-world shape a run actually starts from.
   */
  ringLattice(n, k, rewireP, seed) {
    const rand = this.rng(seed);
    const key = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
    const have = new Set();
    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= Math.floor(k / 2); j++) have.add(key(i, (i + j) % n));
    }

    const edges = [];
    for (const pair of [...have]) {
      const [a, b] = pair.split(',').map(Number);
      if (rand() < rewireP) {
        const c = Math.floor(rand() * n);
        if (c !== a && !have.has(key(a, c))) {
          have.delete(pair);
          have.add(key(a, c));
          edges.push([a, c]);
          continue;
        }
      }
      edges.push([a, b]);
    }
    return { nodes: [...Array(n).keys()], edges };
  },

  /** One agent and the neighbours it can see, for the single-agent column. */
  neighbourhood() {
    return {
      nodes: [0, 1, 2, 3, 4],
      edges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [3, 4]]
    };
  },

  adjacency(graph) {
    const adj = new Map(graph.nodes.map(id => [id, []]));
    for (const [a, b] of graph.edges) {
      if (adj.has(a) && !adj.get(a).includes(b)) adj.get(a).push(b);
      if (adj.has(b) && !adj.get(b).includes(a)) adj.get(b).push(a);
    }
    return adj;
  },

  /** The connected pieces of a graph, each as a graph of its own. */
  componentsOf(graph) {
    const adj = this.adjacency(graph);
    const seen = new Set();
    const pieces = [];
    for (const start of graph.nodes) {
      if (seen.has(start)) continue;
      const nodes = [];
      const stack = [start];
      seen.add(start);
      while (stack.length) {
        const id = stack.pop();
        nodes.push(id);
        for (const near of adj.get(id) || []) {
          if (seen.has(near)) continue;
          seen.add(near);
          stack.push(near);
        }
      }
      const held = new Set(nodes);
      pieces.push({ nodes, edges: graph.edges.filter(([a, b]) => held.has(a) && held.has(b)) });
    }
    return pieces;
  },

  /**
   * Positions for a whole graph, one piece at a time.
   *
   * A graph in two pieces cannot be laid out in one pass: nothing joins them,
   * so the repulsion drives them apart until the bounding box is mostly the
   * gap between them and fitting that box squashes both to a smudge. Each
   * piece is laid out in a column of its own, sized by how many agents it
   * holds, which also reads correctly — a splinter group should look like one.
   */
  layout(graph, w, h, seed, pad = 30) {
    const pieces = this.componentsOf(graph);
    if (pieces.length === 1) return this.layoutPiece(graph, w, h, seed, pad);

    pieces.sort((a, b) => b.nodes.length - a.nodes.length);
    const weights = pieces.map(p => Math.sqrt(p.nodes.length));
    const total = weights.reduce((a, b) => a + b, 0);

    const pos = new Map();
    let left = 0;
    pieces.forEach((piece, i) => {
      const width = w * (weights[i] / total);
      const sub = this.layoutPiece(piece, width, h, seed + i * 17,
                                   Math.min(pad, width * 0.2));
      for (const [id, p] of sub) pos.set(id, { x: left + p.x, y: p.y });
      left += width;
    });
    return pos;
  },

  /**
   * Fruchterman-Reingold: every pair pushes apart, every edge pulls together,
   * a little gravity keeps the whole from drifting off, and the step size
   * cools so it settles instead of oscillating. Run once per panel and kept,
   * since the graph does not change while it animates.
   */
  layoutPiece(graph, w, h, seed, pad = 30) {
    const ids = graph.nodes;
    const n = ids.length;
    const rand = this.rng(seed);
    const at = ids.map(() => ({ x: rand() * 2 - 1, y: rand() * 2 - 1 }));
    const index = new Map(ids.map((id, i) => [id, i]));
    const links = graph.edges
      .map(([a, b]) => [index.get(a), index.get(b)])
      .filter(([a, b]) => a !== undefined && b !== undefined && a !== b);

    const ideal = Math.sqrt(1 / Math.max(1, n)) * 1.4;
    let temp = 0.32;
    for (let pass = 0; pass < 300; pass++) {
      const push = at.map(() => ({ x: 0, y: 0 }));

      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dx = at[i].x - at[j].x, dy = at[i].y - at[j].y;
          let d = Math.hypot(dx, dy);
          // Two nodes exactly on top of each other have no direction to
          // separate along, so give them one.
          if (d < 1e-6) { dx = (rand() - 0.5) * 1e-3; dy = (rand() - 0.5) * 1e-3; d = Math.hypot(dx, dy); }
          const f = (ideal * ideal) / d;
          push[i].x += (dx / d) * f; push[i].y += (dy / d) * f;
          push[j].x -= (dx / d) * f; push[j].y -= (dy / d) * f;
        }
      }
      for (const [a, b] of links) {
        const dx = at[a].x - at[b].x, dy = at[a].y - at[b].y;
        const d = Math.hypot(dx, dy) || 1e-6;
        const f = (d * d) / ideal;
        push[a].x -= (dx / d) * f; push[a].y -= (dy / d) * f;
        push[b].x += (dx / d) * f; push[b].y += (dy / d) * f;
      }
      for (let i = 0; i < n; i++) {
        push[i].x -= at[i].x * 0.08;
        push[i].y -= at[i].y * 0.08;
        const d = Math.hypot(push[i].x, push[i].y) || 1e-6;
        const step = Math.min(d, temp);
        at[i].x += (push[i].x / d) * step;
        at[i].y += (push[i].y / d) * step;
      }
      temp *= 0.985;
    }

    // Fit the result into the canvas, one scale for both axes so the shape the
    // forces settled on is not stretched.
    let lo = { x: Infinity, y: Infinity }, hi = { x: -Infinity, y: -Infinity };
    for (const p of at) {
      lo.x = Math.min(lo.x, p.x); hi.x = Math.max(hi.x, p.x);
      lo.y = Math.min(lo.y, p.y); hi.y = Math.max(hi.y, p.y);
    }
    const scale = Math.min((w - pad * 2) / Math.max(hi.x - lo.x, 1e-6),
                           (h - pad * 2) / Math.max(hi.y - lo.y, 1e-6));
    const midX = (lo.x + hi.x) / 2, midY = (lo.y + hi.y) / 2;

    const pos = new Map();
    ids.forEach((id, i) => pos.set(id, {
      x: w / 2 + (at[i].x - midX) * scale,
      y: h / 2 + (at[i].y - midY) * scale
    }));
    return pos;
  },

  // ---- drawing ---------------------------------------------------------

  /**
   * The whole graph in one call, and the only place edges are drawn.
   *
   * An edge never outlives its endpoints: its opacity is capped at the fainter
   * of the two, so a node that fades out takes its edges with it and a node
   * that has not arrived yet cannot be reached by one. Every panel goes
   * through here for that reason.
   */
  drawGraph(ctx, scene, opts = {}) {
    const nodeAlpha = opts.nodeAlpha || (() => 1);
    const edgeAlpha = opts.edgeAlpha || (() => 1);
    const edgeColour = opts.edgeColour || (() => this.ink.edge);
    const edgeWidth = opts.edgeWidth || (() => (scene.crowd ? 1 : 1.3));
    const nodeColour = opts.nodeColour || (() => this.ink.node);
    const nodeRadius = opts.nodeRadius || (() => scene.r);
    const glow = opts.glow || (() => 0);

    scene.edges.forEach(([a, b], i) => {
      const alpha = Math.min(nodeAlpha(a), nodeAlpha(b)) * edgeAlpha(a, b, i);
      if (alpha <= 0.01) return;
      const p = scene.pos.get(a), q = scene.pos.get(b);
      if (!p || !q) return;
      ctx.globalAlpha = alpha;
      this.edge(ctx, p, q, edgeColour(a, b, i), edgeWidth(a, b, i));
    });

    for (const id of scene.ids) {
      const alpha = nodeAlpha(id);
      if (alpha <= 0.01) continue;
      const r = nodeRadius(id);
      if (r <= 0.2) continue;
      ctx.globalAlpha = alpha;
      this.node(ctx, scene.pos.get(id), r, nodeColour(id), glow(id));
    }
    ctx.globalAlpha = 1;
  },

  edge(ctx, a, b, colour, width = 1) {
    ctx.strokeStyle = colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  },

  node(ctx, p, r, colour, glow = 0) {
    if (!p) return;
    if (glow > 0) {
      ctx.save();
      ctx.globalAlpha *= 0.22 * glow;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r * 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  },

  /**
   * A few layers of dots inside an agent. Below about nine pixels the layers
   * stop being separable, so it drops to a single cluster rather than drawing
   * a smudge and calling it a network.
   */
  brain(ctx, p, r, t, seed = 0, changed = -1) {
    ctx.save();
    ctx.translate(p.x, p.y);

    if (r < 9) {
      for (let i = 0; i < 3; i++) {
        const lit = 0.4 + 0.6 * Math.max(0, Math.sin(t * 3 + seed - i * 0.7));
        ctx.fillStyle = (changed === i) ? this.ink.rich : `rgba(255,255,255,${lit})`;
        ctx.beginPath();
        ctx.arc((i - 1) * r * 0.42, 0, r * 0.19, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      return;
    }

    const cols = [[-1, 2], [0, 3], [1, 2]];
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 0.5;
    const layers = [];
    cols.forEach(([cx, count], ci) => {
      const layer = [];
      for (let i = 0; i < count; i++) {
        layer.push({ x: cx * r * 0.5, y: (i - (count - 1) / 2) * r * 0.44 });
      }
      layers.push(layer);
      if (ci > 0) {
        for (const a of layers[ci - 1]) for (const b of layer) {
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
    });
    layers.flat().forEach((q, i) => {
      const lit = 0.45 + 0.55 * Math.max(0, Math.sin(t * 3 + seed - i * 0.5));
      ctx.fillStyle = (changed === i) ? this.ink.rich : `rgba(255,255,255,${lit})`;
      ctx.beginPath();
      ctx.arc(q.x, q.y, r * 0.13, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  },

  /**
   * How open an agent's eyes are: one, except for the moment of a blink.
   * Every agent gets its own rhythm from its seed, so a crowd does not blink
   * in unison.
   */
  blink(t, seed) {
    const period = 3.1 + (seed % 11) * 0.31;
    const phase = (t + seed * 1.37) % period;
    const span = 0.15;
    if (phase > span) return 1;
    return Math.abs(Math.cos((Math.PI * phase) / span));
  },

  /**
   * Two eyes, looking towards a point. Flat black shapes on the body of the
   * agent — an icon rather than a face — that shut and open again.
   */
  eyes(ctx, p, r, towards, t, seed = 0) {
    const open = this.blink(t, seed);
    const dx = towards.x - p.x, dy = towards.y - p.y;
    const len = Math.hypot(dx, dy) || 1;
    const gaze = { x: dx / len, y: dy / len };
    const spread = r * 0.4;
    const px = -gaze.y * spread, py = gaze.x * spread;

    ctx.fillStyle = this.ink.eye;
    for (const side of [-1, 1]) {
      const ex = p.x + px * side + gaze.x * r * 0.12;
      const ey = p.y + py * side + gaze.y * r * 0.12;
      if (open < 0.18) {
        // Shut: a lid, drawn across the direction the pair sits in.
        ctx.save();
        ctx.translate(ex, ey);
        ctx.rotate(Math.atan2(py, px));
        ctx.fillRect(-r * 0.2, -r * 0.06, r * 0.4, r * 0.12);
        ctx.restore();
        continue;
      }
      ctx.beginPath();
      ctx.ellipse(ex, ey, r * 0.19, r * 0.27 * open, Math.atan2(gaze.y, gaze.x), 0, Math.PI * 2);
      ctx.fill();
    }
  },

  envelope(ctx, x, y, s, alpha = 1) {
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = '#eef2f8';
    ctx.strokeStyle = 'rgba(20,26,34,0.85)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.rect(x - s, y - s * 0.66, s * 2, s * 1.32);
    ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - s, y - s * 0.66);
    ctx.lineTo(x, y + s * 0.14);
    ctx.lineTo(x + s, y - s * 0.66);
    ctx.stroke();
    ctx.restore();
  },

  /** Dots orbiting an agent, one per token, capped so it stays readable. */
  tokens(ctx, p, count, r, t, colour, alpha = 1) {
    const shown = Math.min(count, 7);
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = colour;
    for (let i = 0; i < shown; i++) {
      const a = t * 1.05 + (i / shown) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(p.x + Math.cos(a) * r * 1.75, p.y + Math.sin(a) * r * 1.75,
              Math.max(1.2, r * 0.14), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  },

  /** A dot travelling from a to b, at 0..1 of the way. */
  mote(ctx, a, b, at, size, colour, arc = 0) {
    const x = a.x + (b.x - a.x) * at;
    const y = a.y + (b.y - a.y) * at - Math.sin(at * Math.PI) * arc;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
    return { x, y };
  },

  label(ctx, text, w, h, colour) {
    ctx.fillStyle = colour || this.ink.text;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, w / 2, h - 9);
    ctx.textAlign = 'left';
  },

  /** A loop that runs 0 to 1 and holds briefly at each end. */
  cycle(t, period) {
    const x = (t % period) / period;
    return Math.max(0, Math.min(1, (x - 0.12) / 0.72));
  },

  /** Eased 0 at `from`, 1 at `to`. */
  ramp(x, from, to) {
    const s = Math.max(0, Math.min(1, (x - from) / (to - from)));
    return s * s * (3 - 2 * s);
  }
};

// ---------------------------------------------------------------------------
// The panels
//
// Each one builds the graph it needs — the whole graph, including anything
// that appears partway through — picks who does what, and then draws. The same
// draw runs over the single-agent graph and the crowd; what changes is how
// many agents have a part.
// ---------------------------------------------------------------------------

Explain.PANELS = [
  {
    title: 'A world to start with',
    text: 'A ring of agents wired to a few neighbours each, and a fixed pile of '
        + 'tokens shared between them. Tokens are the only currency and the '
        + 'total never changes — everything that follows moves them around.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(20, 4, 0.2, 7)
                                        : Explain.ringLattice(7, 2, 0, 3) };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 7);
      const arrival = id => Explain.ramp(k, (scene.ids.indexOf(id) / scene.ids.length) * 0.4, 0.55);
      Explain.drawGraph(ctx, scene, { nodeAlpha: arrival });
      if (k > 0.6) {
        scene.ids.forEach((id, i) => {
          Explain.tokens(ctx, scene.pos.get(id), 1 + (i * 3) % 5, scene.r, t,
                         Explain.ink.good, Explain.ramp(k, 0.6, 0.8));
        });
      }
      Explain.label(ctx, k > 0.6 ? 'and a fixed pile of tokens' : 'agents, and who they can reach',
                    scene.w, scene.h);
    }
  },

  {
    title: 'Every agent carries a brain',
    text: 'A small neural network, never trained. It is copied from a parent '
        + 'with mutation, and that is the only way behaviour ever changes. '
        + 'Every choice an agent makes is read off its outputs.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(20, 4, 0.2, 7)
                                        : Explain.ringLattice(5, 2, 0, 11) };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      Explain.drawGraph(ctx, scene, { nodeRadius: () => scene.r * 1.12 });
      ctx.globalAlpha = Explain.ramp(k, 0.05, 0.4);
      scene.ids.forEach((id, i) => {
        Explain.brain(ctx, scene.pos.get(id), scene.r * 1.12, t, i * 0.6);
      });
      ctx.globalAlpha = 1;
      Explain.label(ctx, 'no two quite the same', scene.w, scene.h);
    }
  },

  {
    title: 'It looks, and it speaks',
    text: 'Each agent observes itself and every neighbour at once: their '
        + 'tokens, their degree, and the messages it was sent. In the same '
        + 'breath it writes a short message back to each of them, and one to '
        + 'itself — which is the only memory it has. Nothing forces a '
        + 'message to mean anything.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(15, 4, 0.2, 5)
                                        : Explain.neighbourhood() };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      Explain.drawGraph(ctx, scene, { nodeRadius: () => scene.r * 1.1 });

      // Each agent's gaze moves from one neighbour to the next, so over a loop
      // it has looked at all of them — which is what observing every neighbour
      // at once looks like when it has to be drawn one glance at a time.
      scene.ids.forEach((id, i) => {
        const near = scene.adj.get(id) || [];
        if (!near.length) return;
        const at = scene.pos.get(id);
        const look = scene.pos.get(near[Math.floor(t * 0.7 + i) % near.length]);
        Explain.eyes(ctx, at, scene.r * 1.1, look, t, i + 1);
      });

      // Envelopes cross in both directions at once: everyone writes to
      // everyone they can reach, in the same step.
      const send = Explain.ramp(k, 0.15, 0.9);
      scene.edges.forEach(([a, b], i) => {
        const p = scene.pos.get(a), q = scene.pos.get(b);
        const at = (send + i * 0.13) % 1;
        const size = scene.crowd ? 3 : 4.5;
        Explain.envelope(ctx, p.x + (q.x - p.x) * at, p.y + (q.y - p.y) * at, size, 0.9);
        Explain.envelope(ctx, q.x + (p.x - q.x) * at, q.y + (p.y - q.y) * at, size, 0.55);
      });
      Explain.label(ctx, 'a message to every neighbour, and one to itself', scene.w, scene.h);
    }
  },

  {
    title: 'Reproduction',
    text: 'An agent decides what share of its tokens to spend on a child and '
        + 'pays the full price. The child inherits a mutated copy of the brain '
        + 'and is wired to whichever of the parent’s neighbours the parent '
        + 'picks. The parent can also hand over one of its own edges — '
        + 'giving away position rather than copying it.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(16, 4, 0.2, 9)
                                      : Explain.neighbourhood();
      const adj = Explain.adjacency(graph);
      const parents = scale === 'crowd'
        ? Explain.choose(graph.nodes, 5, 31)
        : [0];

      // The child, the edge to its parent, and the edge it is handed are all
      // part of the graph that gets laid out, so the newborn already has
      // somewhere sensible to be before it appears.
      const births = [];
      parents.forEach((parent, i) => {
        const near = adj.get(parent) || [];
        if (!near.length) return;
        const child = `c${parent}`;
        const handed = near[i % near.length];
        graph.nodes.push(child);
        graph.edges.push([parent, child], [child, handed]);
        births.push({ parent, child, handed });
      });
      return { graph, roles: { births } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      const { births } = scene.roles;
      const isChild = new Set(births.map(b => b.child));
      const grown = Explain.ramp(k, 0.12, 0.8);
      const handedOver = new Map(births.map(b => [`${b.parent}|${b.handed}`, b]));

      Explain.drawGraph(ctx, scene, {
        nodeAlpha: id => (isChild.has(id) ? grown : 1),
        nodeRadius: id => (isChild.has(id) ? scene.r * (0.35 + 0.65 * grown) : scene.r),
        nodeColour: id => (isChild.has(id) ? Explain.ink.good : Explain.ink.node),
        // The handed edge leaves the parent as the child takes it up. Both are
        // in the layout, so neither has to be invented mid-animation.
        edgeAlpha: (a, b) => {
          if (handedOver.has(`${a}|${b}`) || handedOver.has(`${b}|${a}`)) return 1 - grown * 0.85;
          return 1;
        },
        edgeColour: (a, b) => {
          if (isChild.has(a) || isChild.has(b)) return `rgba(127,212,160,${0.3 + 0.7 * grown})`;
          if (handedOver.has(`${a}|${b}`) || handedOver.has(`${b}|${a}`)) return Explain.ink.rich;
          return Explain.ink.edge;
        }
      });

      // Tokens leaving the parent, which is what the child is made of.
      for (const b of births) {
        const p = scene.pos.get(b.parent), c = scene.pos.get(b.child);
        for (let i = 0; i < 3; i++) {
          const at = (k * 1.6 - i * 0.16);
          if (at <= 0 || at >= 1) continue;
          Explain.mote(ctx, p, c, at, Math.max(1.4, scene.r * 0.17), Explain.ink.good);
        }
        if (grown > 0.45) {
          ctx.globalAlpha = Explain.ramp(grown, 0.45, 0.9);
          Explain.brain(ctx, c, scene.r * grown, t, 2);
          ctx.globalAlpha = 1;
        }
      }
      Explain.label(ctx, k > 0.55 ? 'one edge copied, one given away'
                                  : 'the parent pays the full price', scene.w, scene.h);
    }
  },

  {
    title: 'Rewire',
    text: 'An agent can hand one of its edges to another of its neighbours. '
        + 'The link (agent, other) becomes (recipient, other): it drops out of '
        + 'the middle and leaves the two it stood between joined directly. A '
        + 'rewire never creates an edge.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(18, 4, 0.15, 4)
                                      : Explain.neighbourhood();
      const adj = Explain.adjacency(graph);
      const actors = scale === 'crowd' ? Explain.choose(graph.nodes, 5, 17) : [0];

      const rewires = [];
      const used = new Set();
      for (const a of actors) {
        const near = (adj.get(a) || []).filter(x => !used.has(`${a}|${x}`));
        if (near.length < 2) continue;
        const other = near[0], recipient = near[1];
        used.add(`${a}|${other}`);
        // The link the rewire creates is part of the laid-out graph, so it
        // grows into a place that was already left for it.
        graph.edges.push([recipient, other]);
        rewires.push({ a, other, recipient });
      }
      return { graph, roles: { rewires } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      const { rewires } = scene.roles;
      const moved = Explain.ramp(k, 0.2, 0.85);
      const key = (a, b) => (String(a) < String(b) ? `${a}|${b}` : `${b}|${a}`);
      const leaving = new Map(rewires.map(r => [key(r.a, r.other), r]));
      const arriving = new Map(rewires.map(r => [key(r.recipient, r.other), r]));

      Explain.drawGraph(ctx, scene, {
        edgeAlpha: (a, b) => {
          if (leaving.has(key(a, b))) return 1 - moved;
          if (arriving.has(key(a, b))) return moved;
          return 1;
        },
        edgeColour: (a, b) => {
          if (leaving.has(key(a, b))) return Explain.ink.warn;
          if (arriving.has(key(a, b))) return Explain.ink.good;
          return Explain.ink.edge;
        },
        edgeWidth: (a, b) => (leaving.has(key(a, b)) || arriving.has(key(a, b))
          ? (scene.crowd ? 1.6 : 2) : (scene.crowd ? 1 : 1.3)),
        glow: id => (rewires.some(r => r.a === id) ? 0.5 * (1 - moved) : 0)
      });

      // The end of the link slides from the agent to the recipient, which is
      // the whole of what a rewire is.
      for (const r of rewires) {
        const from = scene.pos.get(r.a), to = scene.pos.get(r.recipient);
        Explain.mote(ctx, from, to, moved, Math.max(1.8, scene.r * 0.24), Explain.ink.pale);
      }
      Explain.label(ctx, k > 0.55 ? 'the agent has dropped out of the middle'
                                  : 'handing the link on', scene.w, scene.h);
    }
  },

  {
    title: 'Starve, cull, share out',
    text: 'Agents at zero tokens are removed. Of what is left, only the largest '
        + 'connected piece survives — a splinter group is culled however '
        + 'healthy it looks. Everything the dead held is pooled and scattered '
        + 'at random over the survivors, so the total still balances. This '
        + 'happens after both phases, not once.',
    build(scale) {
      const size = scale === 'crowd' ? 18 : 6;
      const graph = scale === 'crowd' ? Explain.ringLattice(size, 4, 0.18, 6)
                                      : Explain.ringLattice(size, 2, 0, 2);
      // A splinter group, joined to nothing: the cull is about connection, not
      // about health, so it has to be genuinely detached to make the point.
      const splinter = scale === 'crowd' ? ['s0', 's1', 's2', 's3'] : ['s0', 's1'];
      graph.nodes.push(...splinter);
      for (let i = 0; i + 1 < splinter.length; i++) graph.edges.push([splinter[i], splinter[i + 1]]);
      if (splinter.length > 2) graph.edges.push([splinter[0], splinter[splinter.length - 1]]);

      const starved = Explain.choose(graph.nodes.filter(n => typeof n === 'number'),
                                     scale === 'crowd' ? 4 : 1, 23);
      return { graph, roles: { splinter: new Set(splinter), starved: new Set(starved) } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 7);
      const { splinter, starved } = scene.roles;
      const gone = Explain.ramp(k, 0.25, 0.6);

      Explain.drawGraph(ctx, scene, {
        // Both the starved and the splinter fade; drawGraph takes their edges
        // with them, so nothing is left hanging in the air.
        nodeAlpha: id => (starved.has(id) || splinter.has(id) ? 1 - gone : 1),
        nodeColour: id => (starved.has(id) ? Explain.ink.warn
                          : splinter.has(id) ? Explain.ink.dim : Explain.ink.node),
        glow: id => (starved.has(id) ? 0.6 * (1 - gone) : 0)
      });

      const survivors = scene.ids.filter(id => !starved.has(id) && !splinter.has(id));
      if (k > 0.6) {
        const share = Explain.ramp(k, 0.6, 0.85);
        survivors.forEach((id, i) => {
          Explain.tokens(ctx, scene.pos.get(id), 2 + (i % 3), scene.r, t, Explain.ink.good, share);
        });
      }
      Explain.label(ctx, k > 0.6 ? 'what the dead held goes to the living'
                                 : 'broke, or cut adrift', scene.w, scene.h);
    }
  },

  {
    title: 'The game',
    text: 'Every agent spreads its tokens across itself and its neighbours '
        + '— a Colonel Blotto game played on the graph. Whoever commits '
        + 'most to a node takes it. Nothing is destroyed: the tokens move to '
        + 'wherever they were sent.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(18, 4, 0.2, 8)
                                        : Explain.neighbourhood() };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      const sent = Explain.ramp(k, 0.1, 0.75);
      Explain.drawGraph(ctx, scene, {
        glow: id => (k > 0.8 ? 0.5 * ((scene.ids.indexOf(id) % 3 === 0) ? 1 : 0) : 0),
        nodeColour: id => (k > 0.8 && scene.ids.indexOf(id) % 3 === 0
          ? Explain.ink.rich : Explain.ink.node)
      });

      // Every agent stakes every neighbour, and itself. Drawn as dots leaving
      // in all directions at once, because that is how it happens.
      const size = Math.max(1.3, scene.r * 0.19);
      for (const id of scene.ids) {
        const from = scene.pos.get(id);
        for (const near of scene.adj.get(id) || []) {
          const to = scene.pos.get(near);
          if (sent > 0 && sent < 1) Explain.mote(ctx, from, to, sent, size, Explain.ink.pale);
        }
        // Its own claim, held close rather than sent anywhere.
        if (sent > 0.2) {
          ctx.fillStyle = Explain.ink.pale;
          ctx.globalAlpha = Math.min(1, sent * 1.5);
          ctx.beginPath();
          ctx.arc(from.x, from.y - scene.r * 1.6, size, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
      Explain.label(ctx, k > 0.8 ? 'whoever committed most takes the node'
                                 : 'everyone stakes everyone', scene.w, scene.h);
    }
  },

  {
    title: 'Revolutions',
    text: 'The largest bid does not always win. Part of every allocation is '
        + 'flagged as a revolt token, and the small allocators are counted up '
        + 'from the weakest. The moment that lower group outweighs everyone '
        + 'above it, the revolution carries and the winner is drawn from the '
        + 'group that tipped it.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(16, 4, 0.15, 12)
                                      : Explain.neighbourhood();
      const adj = Explain.adjacency(graph);
      const prizes = scale === 'crowd' ? Explain.choose(graph.nodes, 3, 41) : [0];
      const fights = prizes.map(prize => {
        const near = adj.get(prize) || [];
        return { prize, leader: near[0], rebels: near.slice(1, 4) };
      }).filter(f => f.leader !== undefined && f.rebels.length >= 2);
      return { graph, roles: { fights } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 7);
      const { fights } = scene.roles;
      const carried = k > 0.62;
      const rise = Explain.ramp(k, 0.3, 0.62);

      const leaders = new Set(fights.map(f => f.leader));
      const rebels = new Set(fights.flatMap(f => f.rebels));
      const prizes = new Set(fights.map(f => f.prize));

      Explain.drawGraph(ctx, scene, {
        nodeColour: id => {
          // The node being fought over is pale until it falls, so it never
          // wears the same colour as the leader bidding for it.
          if (prizes.has(id)) return carried ? Explain.ink.good : Explain.ink.pale;
          if (leaders.has(id)) return Explain.ink.rich;
          if (rebels.has(id)) return Explain.ink.good;
          return Explain.ink.node;
        },
        nodeRadius: id => (leaders.has(id) ? scene.r * 1.35 : scene.r),
        glow: id => (prizes.has(id) ? 0.7 : leaders.has(id) ? 0.4 * (1 - rise) : 0)
      });

      // The coalition: the weakest counted upward until together they outweigh
      // the leader. Drawn as the rebels binding to one another.
      ctx.lineWidth = scene.crowd ? 1.2 : 1.6;
      for (const f of fights) {
        for (let i = 0; i + 1 < f.rebels.length; i++) {
          const p = scene.pos.get(f.rebels[i]), q = scene.pos.get(f.rebels[i + 1]);
          ctx.strokeStyle = `rgba(127,212,160,${rise * 0.9})`;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      Explain.label(ctx, carried ? 'the coalition tipped it'
                                 : 'the small allocators counted up', scene.w, scene.h);
    }
  },

  {
    title: 'The winner moves in',
    text: 'This is the selection step. The winner’s brain is copied into '
        + 'the node it took, and the node’s new balance is everything that '
        + 'was sent there. Phase one spreads brains around; phase two decides '
        + 'which of them win.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(16, 4, 0.15, 15)
                                      : Explain.neighbourhood();
      const adj = Explain.adjacency(graph);
      const winners = scale === 'crowd' ? Explain.choose(graph.nodes, 4, 53) : [1];
      const taken = new Set();
      const conquests = [];
      for (const from of winners) {
        const target = (adj.get(from) || []).find(x => !taken.has(x) && !winners.includes(x));
        if (target === undefined) continue;
        taken.add(target);
        conquests.push({ from, to: target });
      }
      return { graph, roles: { conquests } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      const { conquests } = scene.roles;
      const arrived = Explain.ramp(k, 0.12, 0.78);
      const done = arrived > 0.92;
      const losers = new Map(conquests.map(c => [c.to, c]));
      const winners = new Set(conquests.map(c => c.from));

      Explain.drawGraph(ctx, scene, {
        nodeColour: id => {
          if (winners.has(id)) return Explain.ink.node;
          if (losers.has(id)) return done ? Explain.ink.node : 'rgba(150,110,110,0.75)';
          return Explain.ink.node;
        },
        nodeRadius: id => (winners.has(id) || losers.has(id) ? scene.r * 1.2 : scene.r),
        edgeColour: (a, b) => ((winners.has(a) && losers.has(b)) || (winners.has(b) && losers.has(a))
          ? Explain.ink.pale : Explain.ink.edge)
      });

      for (const c of conquests) {
        const from = scene.pos.get(c.from), to = scene.pos.get(c.to);
        Explain.brain(ctx, from, scene.r * 1.2, t, 0);
        // The loser is overwritten rather than removed, which is what being
        // conquered means here: the node stays, the brain in it does not.
        if (done) {
          Explain.brain(ctx, to, scene.r * 1.2, t, 0);
        } else {
          ctx.globalAlpha = 0.5;
          Explain.brain(ctx, to, scene.r * 1.2, t * 0.3, 4);
          ctx.globalAlpha = 1;
          const at = { x: from.x + (to.x - from.x) * arrived,
                       y: from.y + (to.y - from.y) * arrived - Math.sin(arrived * Math.PI) * scene.r };
          Explain.node(ctx, at, scene.r * 0.75, Explain.ink.good, 0.8);
          Explain.brain(ctx, at, scene.r * 0.75, t, 0);
        }
      }
      Explain.label(ctx, done ? 'the same brain now runs both'
                              : 'the winner’s brain is copied across', scene.w, scene.h);
    }
  },

  {
    title: 'Everyone mutates',
    text: 'Not only the newborns. After the game every brain in the world is '
        + 'mutated — a sparse jitter of its weights, with an occasional '
        + 'larger redraw. This is the whole engine of variation, and it runs '
        + 'every iteration whether an agent reproduced or not.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(20, 4, 0.2, 7)
                                        : Explain.ringLattice(5, 2, 0, 11) };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 5);
      Explain.drawGraph(ctx, scene, { nodeRadius: () => scene.r * 1.12 });
      // A sweep across the world: sparse, and every agent caught by it.
      scene.ids.forEach((id, i) => {
        const p = scene.pos.get(id);
        const reached = k > (i / scene.ids.length) * 0.6;
        const changed = reached ? Math.floor((i * 5 + Math.floor(t)) % 7) : -1;
        Explain.brain(ctx, p, scene.r * 1.12, t, i * 0.6, changed);
        if (reached && k < 0.9) {
          ctx.globalAlpha = 0.5 * (1 - k);
          Explain.node(ctx, p, scene.r * 1.5, Explain.ink.rich, 0.5);
          ctx.globalAlpha = 1;
        }
      });
      Explain.label(ctx, 'every brain, every iteration', scene.w, scene.h);
    }
  },

  {
    title: 'Quiet edges are cut',
    text: 'An edge that carried no tokens at all this phase is removed, so the '
        + 'graph keeps only the connections anyone actually used. Then the '
        + 'world is cleaned up again — starve, cull, share out — and '
        + 'the next iteration begins.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(18, 4, 0.2, 21)
                                      : Explain.neighbourhood();
      const cut = new Set(Explain.choose([...graph.edges.keys()],
                                         Math.max(1, Math.round(graph.edges.length * 0.3)), 61));
      return { graph, roles: { cut } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 6);
      const { cut } = scene.roles;
      const removed = Explain.ramp(k, 0.45, 0.85);

      Explain.drawGraph(ctx, scene, {
        edgeAlpha: (a, b, i) => (cut.has(i) ? 1 - removed : 1),
        edgeColour: (a, b, i) => (cut.has(i) ? Explain.ink.warn : Explain.ink.edge),
        edgeWidth: (a, b, i) => (cut.has(i) ? (scene.crowd ? 1 : 1.3) : (scene.crowd ? 1.3 : 1.7))
      });

      // The edges that did carry something show it; the quiet ones show
      // nothing, which is exactly why they go.
      const flow = Explain.ramp(k, 0.05, 0.45);
      scene.edges.forEach(([a, b], i) => {
        if (cut.has(i) || flow <= 0 || flow >= 1) return;
        const p = scene.pos.get(a), q = scene.pos.get(b);
        Explain.mote(ctx, p, q, (flow + i * 0.17) % 1, Math.max(1.2, scene.r * 0.17), Explain.ink.pale);
      });
      Explain.label(ctx, k > 0.6 ? 'unused links are gone' : 'which edges carried anything?',
                    scene.w, scene.h);
    }
  }
];

// ---------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------

Object.assign(Explain, {
  canvases: [],
  started: false,
  scenes: new Map(),

  init() {
    const host = document.getElementById('explainPanels');
    if (!host) return;

    host.innerHTML = this.PANELS.map((panel, i) => `
      <section class="explain-panel">
        <div class="explain-art">
          <canvas data-panel="${i}" data-scale="solo"></canvas>
          <span class="explain-step">${i + 1}</span>
          <span class="explain-cap">one agent</span>
        </div>
        <div class="explain-art">
          <canvas data-panel="${i}" data-scale="crowd"></canvas>
          <span class="explain-cap">the whole world</span>
        </div>
        <div class="explain-words">
          <h3>${panel.title}</h3>
          <p>${panel.text}</p>
        </div>
      </section>`).join('');

    this.canvases = [...host.querySelectorAll('canvas')];
    this.scenes.clear();
    if (!this.started) {
      this.started = true;
      requestAnimationFrame(now => this.frame(now));
    }
  },

  /**
   * The graph for one panel at one size, laid out and ready to draw on.
   *
   * Built once and kept: the forces are the expensive part and the graph does
   * not change while it animates. A resize throws the old one away, since the
   * layout is fitted to the box it was made for.
   */
  scene(index, scale, w, h) {
    const cacheKey = `${index}:${scale}:${w}x${h}`;
    const held = this.scenes.get(cacheKey);
    if (held) return held;

    const panel = this.PANELS[index];
    const built = panel.build(scale);
    const graph = built.graph;
    const crowd = scale === 'crowd';
    const r = crowd ? Math.max(4.5, Math.min(w, h) * 0.035) : Math.max(9, Math.min(w, h) * 0.075);

    const scene = {
      ids: graph.nodes,
      edges: graph.edges,
      adj: this.adjacency(graph),
      pos: this.layout(graph, w, h - 16, index * 31 + (crowd ? 7 : 1), r * 2.2),
      roles: built.roles || {},
      crowd, r, w, h
    };
    this.scenes.set(cacheKey, scene);
    return scene;
  },

  /**
   * Paint one frame of every panel that is actually on screen.
   *
   * Twenty-two canvases animating at once is more work than the page needs,
   * and most of them are scrolled out of sight at any moment. Off-screen
   * panels are skipped entirely, which keeps this to whatever the reader is
   * looking at.
   */
  frame(now) {
    const t = now / 1000;
    if (typeof App !== 'undefined' && App.view === 'explain') {
      for (const canvas of this.canvases) {
        const box = canvas.getBoundingClientRect();
        if (box.bottom < -80 || box.top > window.innerHeight + 80) continue;
        this.paint(canvas, t);
      }
    }
    requestAnimationFrame(next => this.frame(next));
  },

  paint(canvas, t) {
    const index = Number(canvas.dataset.panel);
    const panel = this.PANELS[index];
    if (!panel) return;

    const dpr = window.devicePixelRatio || 1;
    const box = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width)), h = Math.max(1, Math.round(box.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.ink.bg;
    ctx.fillRect(0, 0, w, h);
    ctx.lineCap = 'round';
    panel.draw(ctx, t, this.scene(index, canvas.dataset.scale || 'solo', w, h));
  }
});
