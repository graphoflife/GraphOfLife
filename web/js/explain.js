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
        const lit = 0.45 + 0.55 * Math.max(0, Math.sin(t * 3 + seed - i * 0.7));
        ctx.fillStyle = (changed === i) ? this.ink.rich : `rgba(5,7,10,${lit})`;
        ctx.beginPath();
        ctx.arc((i - 1) * r * 0.42, 0, r * 0.19, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
      return;
    }

    const cols = [[-1, 2], [0, 3], [1, 2]];
    ctx.strokeStyle = 'rgba(5,7,10,0.4)';
    ctx.lineWidth = 0.6;
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
      ctx.fillStyle = (changed === i) ? this.ink.rich : `rgba(5,7,10,${lit})`;
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
   * One eye, on the body of the agent, looking towards a point.
   *
   * A lens of two arcs, a dark iris that slides towards whatever is being
   * looked at, and a highlight — the shape of an eye icon, built from
   * primitives so it scales and blinks rather than being an image. The iris is
   * clipped to the lens, so at the edge of a look it is cut off by the lid
   * exactly as a real one would be.
   *
   * A gaze aimed at the agent itself has nowhere to point, so the iris settles
   * in the middle. That is what an agent observing itself looks like.
   */
  eye(ctx, p, r, towards, t, seed = 0) {
    const open = this.blink(t, seed);
    const halfW = r * 0.8, halfH = r * 0.44;
    const line = Math.max(0.9, r * 0.11);

    const lens = () => {
      const lid = halfH * 2 * Math.max(open, 0.001);
      ctx.beginPath();
      ctx.moveTo(p.x - halfW, p.y);
      ctx.quadraticCurveTo(p.x, p.y - lid, p.x + halfW, p.y);
      ctx.quadraticCurveTo(p.x, p.y + lid, p.x - halfW, p.y);
      ctx.closePath();
    };

    if (open < 0.12) {
      ctx.strokeStyle = this.ink.eye;
      ctx.lineWidth = line;
      ctx.beginPath();
      ctx.moveTo(p.x - halfW, p.y);
      ctx.lineTo(p.x + halfW, p.y);
      ctx.stroke();
      return;
    }

    lens();
    ctx.fillStyle = '#f4f7fb';
    ctx.fill();

    ctx.save();
    lens();
    ctx.clip();
    const dx = towards.x - p.x, dy = towards.y - p.y;
    const len = Math.hypot(dx, dy);
    // Vertical travel is smaller than horizontal because the lens is wider
    // than it is tall; an iris that moved equally would leave the white.
    const gx = len > 0.5 ? (dx / len) * halfW * 0.4 : 0;
    const gy = len > 0.5 ? (dy / len) * halfH * 0.5 : 0;
    const iris = Math.min(halfH * 1.15, r * 0.4);
    ctx.fillStyle = this.ink.eye;
    ctx.beginPath();
    ctx.arc(p.x + gx, p.y + gy, iris, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.beginPath();
    ctx.arc(p.x + gx + iris * 0.36, p.y + gy - iris * 0.36, iris * 0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    lens();
    ctx.strokeStyle = this.ink.eye;
    ctx.lineWidth = line;
    ctx.stroke();
  },

  /**
   * What an agent is looking at right now: one of its neighbours, or itself.
   *
   * The gaze jumps from one to the next and holds, rather than sweeping
   * between them. An agent does not swing its attention around the
   * neighbourhood — it reads all of them in one step, and this is that step
   * drawn one glance at a time.
   */
  gazeTarget(scene, id, t, seed = 0) {
    const targets = [...(scene.adj.get(id) || []), id];   // itself, last
    const step = Math.floor(t / 1.15 + seed * 0.6);
    return scene.pos.get(targets[((step % targets.length) + targets.length) % targets.length]);
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

  /**
   * Tokens as dots held close around an agent, one per token, capped so it
   * stays readable. Big enough to count at a glance and tight enough to the
   * body that whose they are is never in question.
   */
  tokens(ctx, p, count, r, t, colour, alpha = 1) {
    const shown = Math.min(count, 7);
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = colour;
    for (let i = 0; i < shown; i++) {
      const a = t * 1.05 + (i / shown) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(p.x + Math.cos(a) * r * 1.3, p.y + Math.sin(a) * r * 1.3,
              Math.max(2.4, r * 0.28), 0, Math.PI * 2);
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

  /**
   * One observation step, shared by both phases.
   *
   * Every agent opens its eye, reads its neighbours and itself, and writes a
   * message to each of them. The two phases observe identically — the same
   * code draws both, because the same code runs both.
   */
  observation(ctx, t, scene, glowing = new Set()) {
    const r = scene.r * 1.15;
    this.drawGraph(ctx, scene, {
      nodeRadius: () => r,
      nodeColour: id => (glowing.has(id) ? this.ink.good : this.ink.node),
      glow: id => (glowing.has(id) ? 0.8 : 0)
    });
    scene.ids.forEach((id, i) => {
      // Inset, so a rim of the agent still shows around the white.
      this.eye(ctx, scene.pos.get(id), r * 0.82, this.gazeTarget(scene, id, t, i), t, i + 1);
    });

    // Messages cross in both directions at once: everyone writes to everyone
    // they can reach, in the same step.
    const send = this.ramp(this.cycle(t, 6), 0.15, 0.9);
    const size = scene.crowd ? 3 : 4.5;
    scene.edges.forEach(([a, b], i) => {
      const p = scene.pos.get(a), q = scene.pos.get(b);
      const at = (send + i * 0.13) % 1;
      this.envelope(ctx, p.x + (q.x - p.x) * at, p.y + (q.y - p.y) * at, size, 0.9);
      this.envelope(ctx, q.x + (p.x - q.x) * at, q.y + (p.y - q.y) * at, size, 0.55);
    });
  },

  /**
   * A cleanup panel, used for both of them — the one after reproduction and
   * the one after the game, since the engine runs the identical step twice.
   *
   * The graph is built so the two rules are visibly connected rather than two
   * unrelated removals. A side group hangs off the rest through a single
   * agent, and that agent is one of the ones holding nothing. It starves
   * first; the group it was carrying is then attached to nothing, and the
   * largest-piece rule takes it however healthy it looks.
   */
  elimination({ title, text, seed }) {
    return {
      title,
      text,
      build(scale) {
        const size = scale === 'crowd' ? 13 : 6;
        const graph = scale === 'crowd' ? Explain.ringLattice(size, 4, 0.15, seed)
                                        : Explain.ringLattice(size, 2, 0, seed);
        const bridge = 'bridge';
        const side = scale === 'crowd' ? ['s0', 's1', 's2', 's3'] : ['s0', 's1'];
        graph.nodes.push(bridge, ...side);
        graph.edges.push([0, bridge], [bridge, side[0]]);
        for (let i = 0; i + 1 < side.length; i++) graph.edges.push([side[i], side[i + 1]]);
        if (side.length > 2) graph.edges.push([side[0], side[side.length - 1]]);

        // Node 0 is what the bridge hangs from, so it must not starve too —
        // the group has to come adrift for one reason, not two.
        const others = graph.nodes.filter(id => typeof id === 'number' && id !== 0);
        const starved = new Set([bridge,
          ...Explain.choose(others, scale === 'crowd' ? 3 : 1, seed + 5)]);
        return { graph, roles: { starved, side: new Set(side) } };
      },
      draw(ctx, t, scene) {
        const k = Explain.cycle(t, 9);
        const { starved, side } = scene.roles;
        const starvedGone = Explain.ramp(k, 0.3, 0.5);
        const adrift = k > 0.5;
        const sideGone = Explain.ramp(k, 0.56, 0.76);

        Explain.drawGraph(ctx, scene, {
          nodeAlpha: id => (starved.has(id) ? 1 - starvedGone
                           : side.has(id) ? 1 - sideGone : 1),
          nodeColour: id => (starved.has(id) ? Explain.ink.warn
                            : side.has(id) && adrift ? Explain.ink.dim : Explain.ink.node),
          glow: id => (starved.has(id) ? 0.6 * (1 - starvedGone) : 0)
        });

        if (k > 0.78) {
          const share = Explain.ramp(k, 0.78, 0.95);
          scene.ids.filter(id => !starved.has(id) && !side.has(id))
            .forEach((id, i) => Explain.tokens(ctx, scene.pos.get(id), 2 + (i % 3),
                                               scene.r, t, Explain.ink.good, share));
        }
        Explain.label(ctx,
          k > 0.78 ? 'what the dead held goes to the living'
          : k > 0.56 ? 'and the piece it carried is cut adrift'
          : k > 0.3 ? 'the one holding them on starved too'
          : 'these are holding nothing', scene.w, scene.h);
      }
    };
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
    title: 'Initial Graph Generation',
    text: 'A ring of agents wired to a few neighbours each, with some links '
        + 'moved elsewhere — a small-world graph. A fixed pile of tokens is '
        + 'shared between them, shown as the dots each one holds. Tokens are the '
        + 'only currency and the total never changes; everything that follows '
        + 'only moves them around. Every agent also carries a brain: a small '
        + 'neural network, drawn here inside the body. It is never trained. It '
        + 'is copied from a parent with mutation, and that is the only way '
        + 'behaviour ever changes.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(18, 4, 0.2, 7)
                                        : Explain.ringLattice(6, 2, 0, 3) };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 7);
      const r = scene.r * 1.15;
      const arrival = id => Explain.ramp(k, (scene.ids.indexOf(id) / scene.ids.length) * 0.35, 0.5);
      Explain.drawGraph(ctx, scene, { nodeAlpha: arrival, nodeRadius: () => r });

      scene.ids.forEach((id, i) => {
        const a = arrival(id);
        if (a <= 0.01) return;
        ctx.globalAlpha = a;
        Explain.brain(ctx, scene.pos.get(id), r, t, i * 0.6);
        ctx.globalAlpha = 1;
        if (k > 0.55) {
          Explain.tokens(ctx, scene.pos.get(id), 1 + (i * 3) % 4, r, t,
                         Explain.ink.good, Explain.ramp(k, 0.55, 0.75));
        }
      });
      Explain.label(ctx, k > 0.55 ? 'a brain each, and a fixed pile of tokens'
                                  : 'agents, and who they can reach', scene.w, scene.h);
    }
  },

  {
    title: 'Observation — Reproduction Phase',
    text: 'Every agent opens its eye and reads its whole neighbourhood in one '
        + 'step: each neighbour’s tokens and degree, and the messages it was '
        + 'sent last time. It observes itself too — that is the glance where '
        + 'the pupil settles in the middle — and its own message to itself is '
        + 'the only memory it has. In the same breath it writes a fresh message '
        + 'to every neighbour. Everything decided in this phase is read off '
        + 'this one observation: what to say, whether to reproduce and for how '
        + 'much, and whether to hand over an edge. The '
        + 'glowing agent is the one that decided to reproduce.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(15, 4, 0.2, 5)
                                      : Explain.neighbourhood();
      const breeders = new Set(scale === 'crowd'
        ? Explain.choose(graph.nodes, 4, 77) : [0]);
      return { graph, roles: { breeders } };
    },
    draw(ctx, t, scene) {
      Explain.observation(ctx, t, scene, scene.roles.breeders);
      Explain.label(ctx, 'reading everyone, and writing back', scene.w, scene.h);
    }
  },

  {
    title: 'Reproduction + Handover',
    text: 'An agent decides what share of its tokens to spend on a child and '
        + 'pays the full price; the child starts with exactly what was spent. '
        + 'It inherits a mutated copy of the parent’s brain and is wired to '
        + 'whichever of the parent’s neighbours the parent picked. The parent '
        + 'can also hand over one of its own edges instead of copying it — '
        + 'dropping that connection itself and giving the child its place, so a '
        + 'lineage passes on position and not only tokens. Handover is '
        + 'optional: it can be switched off when a run is created, and then a '
        + 'child only ever gains links.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(15, 4, 0.2, 9)
                                      : Explain.neighbourhood();
      const adj = Explain.adjacency(graph);
      const parents = scale === 'crowd' ? Explain.choose(graph.nodes, 4, 31) : [0];

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
      const parents = new Set(births.map(b => b.parent));
      const grown = Explain.ramp(k, 0.12, 0.8);
      const handedOver = new Map(births.map(b => [`${b.parent}|${b.handed}`, b]));

      Explain.drawGraph(ctx, scene, {
        nodeAlpha: id => (isChild.has(id) ? grown : 1),
        nodeRadius: id => (isChild.has(id) ? scene.r * (0.35 + 0.65 * grown) : scene.r),
        nodeColour: id => (isChild.has(id) ? Explain.ink.good : Explain.ink.node),
        glow: id => (parents.has(id) ? 0.55 : 0),
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

      for (const b of births) {
        const p = scene.pos.get(b.parent), c = scene.pos.get(b.child);
        for (let i = 0; i < 3; i++) {
          const at = k * 1.6 - i * 0.16;
          if (at <= 0 || at >= 1) continue;
          Explain.mote(ctx, p, c, at, Math.max(1.8, scene.r * 0.22), Explain.ink.good);
        }
        if (grown > 0.45) {
          ctx.globalAlpha = Explain.ramp(grown, 0.45, 0.9);
          Explain.brain(ctx, c, scene.r * grown, t, 2);
          ctx.globalAlpha = 1;
        }
      }
      Explain.label(ctx, k > 0.55 ? 'one link copied, one handed over'
                                  : 'the parent pays the full price', scene.w, scene.h);
    }
  },

  Explain.elimination({
    title: 'Elimination',
    text: 'Two removals, in this order. First every agent holding no tokens is '
        + 'taken out. Then, of whatever is still standing, only the largest '
        + 'connected piece survives — and that second rule bites because of '
        + 'the first: the group on the side was joined to the rest through a '
        + 'single agent, and when that one starved the group was left hanging '
        + 'off nothing. It is culled however healthy it looks. Everything the '
        + 'dead held is pooled and scattered at random over the survivors, so '
        + 'the total still balances.',
    seed: 6
  }),

  {
    title: 'Observation — Game Phase',
    text: 'The same observation again, before the second half of the '
        + 'iteration. Every agent reads its neighbours and itself, and writes a '
        + 'fresh message to each of them — messages are exchanged in both '
        + 'phases, not once per iteration. The graph it is reading is not the '
        + 'one it read last time: children have been born, edges have moved, '
        + 'and the starved are gone. What it sees now decides how it plays the '
        + 'game.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(15, 4, 0.2, 23)
                                        : Explain.neighbourhood(),
               roles: { breeders: new Set() } };
    },
    draw(ctx, t, scene) {
      Explain.observation(ctx, t, scene, scene.roles.breeders);
      Explain.label(ctx, 'the same reading, on a changed graph', scene.w, scene.h);
    }
  },

  {
    title: 'Colonel Blotto Game',
    text: 'Every agent spreads its tokens across itself and its neighbours at '
        + 'once, and whoever commits most to a node takes it. Nothing is '
        + 'destroyed — the tokens move to wherever they were sent, so a node '
        + '’s new balance is everything staked on it.\n\n'
        + 'Revolutions, which are optional, decide it differently. Part of each '
        + 'allocation is flagged as revolt. The biggest single allocator is the '
        + 'hegemon; every other revolting allocator forms the mob, sorted '
        + 'weakest first. Walking up that order, a lower class accumulates, and '
        + 'at each rung the question is whether it now outweighs everyone still '
        + 'above it plus the hegemon. At the first rung where it does, the '
        + 'revolution carries — and the node goes to the strongest allocator '
        + 'in that rung, not to a random member of the crowd. Ties at that '
        + 'exact amount are split at random, and nothing else is. So a crowd of '
        + 'small allocators can take a node from someone who outspent every one '
        + 'of them, and the best-placed of them collects it.\n\n'
        + 'An edge that carried no tokens in either direction is then cut, so '
        + 'the graph keeps only the connections anyone actually used.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(15, 4, 0.18, 8)
                                      : Explain.neighbourhood();
      const adj = Explain.adjacency(graph);
      const prizes = scale === 'crowd' ? Explain.choose(graph.nodes, 2, 41) : [0];
      const fights = prizes.map(prize => {
        const near = adj.get(prize) || [];
        return { prize, hegemon: near[0], mob: near.slice(1, 4) };
      }).filter(f => f.hegemon !== undefined && f.mob.length >= 2);

      // A third of the links carry nothing and are cut at the end.
      const quiet = new Set(Explain.choose([...graph.edges.keys()],
                                           Math.max(1, Math.round(graph.edges.length * 0.28)), 61));
      return { graph, roles: { fights, quiet } };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 9);
      const { fights, quiet } = scene.roles;
      const staked = Explain.ramp(k, 0.05, 0.4);
      const bound = Explain.ramp(k, 0.42, 0.62);
      const carried = k > 0.64;
      const cut = Explain.ramp(k, 0.76, 0.95);

      const hegemons = new Set(fights.map(f => f.hegemon));
      const mob = new Set(fights.flatMap(f => f.mob));
      const prizes = new Set(fights.map(f => f.prize));
      // The strongest of the rung that tipped it is the one that collects.
      const takers = new Set(fights.map(f => f.mob[f.mob.length - 1]));

      Explain.drawGraph(ctx, scene, {
        nodeColour: id => {
          if (prizes.has(id)) return carried ? Explain.ink.good : Explain.ink.pale;
          if (hegemons.has(id)) return Explain.ink.rich;
          if (mob.has(id)) return Explain.ink.good;
          return Explain.ink.node;
        },
        nodeRadius: id => (hegemons.has(id) ? scene.r * 1.35
                          : prizes.has(id) ? scene.r * 1.15
                          : takers.has(id) && carried ? scene.r * 1.3 : scene.r),
        // Only the contested node glows. Giving the taker one as well put two
        // glowing green nodes side by side and it stopped being obvious which
        // was the prize; the taker is marked by size instead.
        glow: id => (prizes.has(id) ? 0.75
                    : hegemons.has(id) ? 0.45 * (1 - bound) : 0),
        edgeAlpha: (a, b, i) => (quiet.has(i) ? 1 - cut : 1),
        edgeColour: (a, b, i) => (quiet.has(i) && k > 0.68 ? Explain.ink.warn : Explain.ink.edge)
      });

      // Everyone stakes every neighbour, and itself, in the same step.
      const size = Math.max(1.6, scene.r * 0.22);
      if (staked > 0 && staked < 1) {
        scene.edges.forEach(([a, b], i) => {
          if (quiet.has(i)) return;             // this one carried nothing, which is why it goes
          const p = scene.pos.get(a), q = scene.pos.get(b);
          Explain.mote(ctx, p, q, staked, size, Explain.ink.pale);
          Explain.mote(ctx, q, p, staked, size, Explain.ink.pale);
        });
      }

      // The mob binding together, weakest first.
      ctx.lineWidth = scene.crowd ? 1.2 : 1.6;
      for (const f of fights) {
        for (let i = 0; i + 1 < f.mob.length; i++) {
          const p = scene.pos.get(f.mob[i]), q = scene.pos.get(f.mob[i + 1]);
          ctx.strokeStyle = `rgba(127,212,160,${bound * 0.9})`;
          ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      Explain.label(ctx,
        k > 0.76 ? 'links that carried nothing are cut'
        : carried ? 'the strongest of the rung takes it'
        : bound > 0.2 ? 'the mob counts up from the weakest'
        : 'everyone stakes everyone', scene.w, scene.h);
    }
  },

  {
    title: 'Winner conquers Nodes',
    text: 'This is the selection step, and the only place a brain is ever '
        + 'chosen. The winner’s brain is copied into the node it took, '
        + 'overwriting whatever was thinking there — the node stays, its '
        + 'occupant does not. The node’s new balance is everything that was '
        + 'staked on it. The first phase spreads brains around by reproduction; '
        + 'this one decides which of them get to keep playing.',
    build(scale) {
      const graph = scale === 'crowd' ? Explain.ringLattice(15, 4, 0.15, 15)
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
      const r = scene.r * 1.2;

      Explain.drawGraph(ctx, scene, {
        nodeColour: id => (losers.has(id) && !done ? 'rgba(150,110,110,0.8)' : Explain.ink.node),
        nodeRadius: id => (winners.has(id) || losers.has(id) ? r : scene.r),
        glow: id => (winners.has(id) ? 0.45 : 0),
        edgeColour: (a, b) => ((winners.has(a) && losers.has(b)) || (winners.has(b) && losers.has(a))
          ? Explain.ink.pale : Explain.ink.edge)
      });

      for (const c of conquests) {
        const from = scene.pos.get(c.from), to = scene.pos.get(c.to);
        Explain.brain(ctx, from, r, t, 0);
        if (done) {
          Explain.brain(ctx, to, r, t, 0);
        } else {
          ctx.globalAlpha = 0.45;
          Explain.brain(ctx, to, r, t * 0.3, 4);
          ctx.globalAlpha = 1;
          const at = { x: from.x + (to.x - from.x) * arrived,
                       y: from.y + (to.y - from.y) * arrived - Math.sin(arrived * Math.PI) * scene.r };
          Explain.node(ctx, at, scene.r * 0.8, Explain.ink.good, 0.8);
          Explain.brain(ctx, at, scene.r * 0.8, t, 0);
        }
      }
      Explain.label(ctx, done ? 'the same brain now runs both'
                              : 'the winner’s brain is copied across', scene.w, scene.h);
    }
  },

  {
    title: 'Mutation',
    text: 'Every brain in the world is mutated — not only the newborns, and '
        + 'not only the winners. A sparse jitter of the weights, with an '
        + 'occasional larger redraw, applied to everyone, every iteration, '
        + 'whether they reproduced or fought or did nothing at all. This is the '
        + 'whole engine of variation. Nothing else ever changes a brain.',
    build(scale) {
      return { graph: scale === 'crowd' ? Explain.ringLattice(18, 4, 0.2, 7)
                                        : Explain.ringLattice(5, 2, 0, 11) };
    },
    draw(ctx, t, scene) {
      const k = Explain.cycle(t, 5);
      const r = scene.r * 1.15;
      Explain.drawGraph(ctx, scene, { nodeRadius: () => r });
      scene.ids.forEach((id, i) => {
        const p = scene.pos.get(id);
        const reached = k > (i / scene.ids.length) * 0.55;
        const changed = reached ? Math.floor((i * 5 + Math.floor(t)) % 7) : -1;
        Explain.brain(ctx, p, r, t, i * 0.6, changed);
        if (reached && k < 0.9) {
          ctx.globalAlpha = 0.5 * (1 - k);
          Explain.node(ctx, p, r * 1.35, Explain.ink.rich, 0.5);
          ctx.globalAlpha = 1;
        }
      });
      Explain.label(ctx, 'every brain, every iteration', scene.w, scene.h);
    }
  },

  Explain.elimination({
    title: 'Elimination',
    text: 'The same clearing-up as before, and it happens after this phase too '
        + '— not once per iteration. Agents left holding nothing by the game '
        + 'are removed, then anything no longer attached to the largest piece '
        + 'goes with them, and what they held is shared out over the survivors. '
        + 'That closes the iteration. The next one begins back at the '
        + 'reproduction observation, on whatever graph is left standing.',
    seed: 34
  })
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
          ${panel.text.split('\n\n').map(para => `<p>${para}</p>`).join('')}
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

    // How big an agent is drawn falls with how many there are, so a panel
    // needing nine nodes does not draw them at the size a panel needing five
    // uses and leave them overlapping. One rule for both columns: the crowd is
    // smaller because it is more numerous, not because it was told to be.
    const r = Math.max(4, Math.min(20, Math.min(w, h) * 0.165 / Math.sqrt(graph.nodes.length)));

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
