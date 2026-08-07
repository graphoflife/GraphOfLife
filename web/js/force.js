/*
 * Force-directed layout.
 *
 * Written by hand rather than pulled from a CDN so the viewer keeps working
 * offline on any machine you clone the repo onto.
 *
 * Four forces per tick:
 *   repulsion  — every node pushes its neighbours in space apart. Evaluated
 *                against a uniform spatial grid rather than all pairs, so cost
 *                is roughly O(n) instead of O(n^2). This is what lets it handle
 *                a few thousand nodes.
 *   springs    — graph edges pull their endpoints toward a rest length.
 *   angular    — a node's incident edges spread evenly around it, so a degree-2
 *                node straightens toward 180 degrees, a degree-3 node toward
 *                120, and so on. Repulsion alone only separates nodes by
 *                distance and happily leaves two edges bunched on one side.
 *   centering  — a weak pull toward the origin so the drawing cannot drift off.
 *
 * Positions persist across frames: a node keeps its coordinates as long as it
 * is alive, and a newborn is seeded next to its parent. That is what stops the
 * graph from scrambling every time you step forward.
 */
class ForceLayout {
  constructor() {
    this.pos = new Map();      // id -> {x, y, vx, vy}
    this.ids = [];
    this.edges = [];
    this.adjacency = new Map(); // id -> [neighbour ids], for the angular force
    this.alpha = 1;

    this.charge = 120;
    this.linkStrength = 0.12;
    this.linkDistance = 24;
    this.centerStrength = 0.012;
    this.angularStrength = 0.15;
    this.damping = 0.86;
  }

  /**
   * Adopt a new frame, keeping the positions of nodes that still exist.
   * `parents` maps a node id to the id that spawned it.
   */
  setFrame(ids, edges, parents, carryPositions) {
    if (!carryPositions) this.pos.clear();

    const next = new Map();
    const spawnRadius = this.linkDistance * 0.6;

    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const existing = this.pos.get(id);
      if (existing) {
        next.set(id, existing);
        continue;
      }

      // A newborn appears just beside its parent, so lineages stay together.
      const parent = parents ? parents[i] : -1;
      const anchor = (parent >= 0) ? this.pos.get(parent) : null;
      const angle = Math.random() * Math.PI * 2;
      const radius = anchor ? spawnRadius : 200 * Math.sqrt(Math.random());
      next.set(id, {
        x: (anchor ? anchor.x : 0) + Math.cos(angle) * radius,
        y: (anchor ? anchor.y : 0) + Math.sin(angle) * radius,
        vx: 0,
        vy: 0
      });
    }

    this.pos = next;
    this.ids = ids;
    this.edges = edges;
    this._buildAdjacency();
  }

  /** Neighbour lists, rebuilt once per frame for the angular force. */
  _buildAdjacency() {
    const adjacency = new Map();
    for (const id of this.ids) adjacency.set(id, []);

    for (const [a, b] of this.edges) {
      const listA = adjacency.get(a);
      const listB = adjacency.get(b);
      if (listA) listA.push(b);
      if (listB) listB.push(a);
    }
    this.adjacency = adjacency;
  }

  reheat(alpha = 1) {
    this.alpha = alpha;
  }

  scatter() {
    for (const p of this.pos.values()) {
      const angle = Math.random() * Math.PI * 2;
      const radius = 250 * Math.sqrt(Math.random());
      p.x = Math.cos(angle) * radius;
      p.y = Math.sin(angle) * radius;
      p.vx = p.vy = 0;
    }
    this.alpha = 1;
  }

  /** Advance the simulation one tick. Returns false once it has settled. */
  tick() {
    if (this.alpha < 0.005) return false;

    const nodes = [];
    for (const id of this.ids) {
      const p = this.pos.get(id);
      if (p) nodes.push(p);
    }
    if (!nodes.length) return false;

    this._repel(nodes);
    this._springs();
    this._angularSpread();

    // Centering + integration.
    const k = this.centerStrength * this.alpha;
    for (const p of nodes) {
      p.vx -= p.x * k;
      p.vy -= p.y * k;
      p.vx *= this.damping;
      p.vy *= this.damping;
      p.x += p.vx;
      p.y += p.vy;
    }

    this.alpha *= 0.985;
    return true;
  }

  /**
   * Short-range repulsion using a uniform grid.
   *
   * Only nodes sharing a cell or an adjacent cell interact. Beyond the cell
   * size the force is negligible anyway, so the approximation costs little and
   * turns the quadratic blow-up into something linear.
   */
  _repel(nodes) {
    const cell = Math.max(20, this.linkDistance * 2.2);
    const grid = new Map();

    for (const p of nodes) {
      const key = `${Math.floor(p.x / cell)},${Math.floor(p.y / cell)}`;
      let bucket = grid.get(key);
      if (!bucket) grid.set(key, bucket = []);
      bucket.push(p);
    }

    const strength = this.charge * this.alpha;
    const maxDistSq = cell * cell;

    for (const p of nodes) {
      const cx = Math.floor(p.x / cell);
      const cy = Math.floor(p.y / cell);

      for (let ox = -1; ox <= 1; ox++) {
        for (let oy = -1; oy <= 1; oy++) {
          const bucket = grid.get(`${cx + ox},${cy + oy}`);
          if (!bucket) continue;

          for (const q of bucket) {
            if (q === p) continue;
            let dx = p.x - q.x;
            let dy = p.y - q.y;
            let dSq = dx * dx + dy * dy;
            if (dSq > maxDistSq) continue;

            // Two nodes exactly on top of each other have no direction to
            // separate along, so nudge them apart randomly.
            if (dSq < 0.01) {
              dx = (Math.random() - 0.5) * 0.1;
              dy = (Math.random() - 0.5) * 0.1;
              dSq = dx * dx + dy * dy;
            }

            const force = strength / dSq;
            p.vx += dx * force;
            p.vy += dy * force;
          }
        }
      }
    }
  }

  _springs() {
    const strength = this.linkStrength * this.alpha;
    for (const [a, b] of this.edges) {
      const pa = this.pos.get(a);
      const pb = this.pos.get(b);
      if (!pa || !pb) continue;

      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
      const push = (dist - this.linkDistance) / dist * strength;

      const fx = dx * push * 0.5;
      const fy = dy * push * 0.5;
      pa.vx += fx; pa.vy += fy;
      pb.vx -= fx; pb.vy -= fy;
    }
  }

  /**
   * Angular resolution: spread each node's incident edges evenly around it.
   *
   * For a node of degree d the ideal gap between neighbouring edges is
   * 2*pi/d — 180 degrees at degree 2, 120 at degree 3. Sorting the incident
   * edges by angle and comparing each consecutive gap against that ideal gives
   * exactly the pairs that are bunched too tightly; those get rotated apart
   * about the shared node.
   *
   * The two neighbours are rotated in opposite directions rather than pushed
   * outward, so this changes the shape of the drawing without fighting the
   * springs over edge length. Rotating a point by a small angle d0 about the
   * centre displaces it by d0 * (-dy, dx), which is the perpendicular of the
   * spoke — so the correction naturally scales with how far out the neighbour
   * sits, and no normalisation is needed.
   *
   * The equal and opposite reaction is applied to the centre node too, which
   * matters more than it sounds. In a dense graph a neighbour is usually a hub
   * pinned by its own edges and barely rotates at all; the light node in the
   * middle is the one free to move. Because a hub collects many conflicting
   * reactions that largely cancel while a degree-2 node collects a single
   * coherent one, the correction lands on whichever end is actually free —
   * sliding that node onto the line between its neighbours, which is precisely
   * the 180-degree arrangement being asked for.
   */
  _angularSpread() {
    const strength = this.angularStrength * this.alpha;
    if (strength <= 0) return;

    const TWO_PI = Math.PI * 2;

    for (const [id, neighbours] of this.adjacency) {
      // A single edge has nothing to spread against.
      if (neighbours.length < 2) continue;
      const centre = this.pos.get(id);
      if (!centre) continue;

      const spokes = [];
      for (const nid of neighbours) {
        const q = this.pos.get(nid);
        if (!q) continue;
        const dx = q.x - centre.x;
        const dy = q.y - centre.y;
        // Two nodes on top of each other have no angle to speak of.
        if (dx * dx + dy * dy < 1e-6) continue;
        spokes.push({ q, dx, dy, angle: Math.atan2(dy, dx) });
      }
      if (spokes.length < 2) continue;

      spokes.sort((a, b) => a.angle - b.angle);
      const ideal = TWO_PI / spokes.length;

      let reactionX = 0, reactionY = 0;

      for (let i = 0; i < spokes.length; i++) {
        const a = spokes[i];
        const b = spokes[(i + 1) % spokes.length];

        let gap = b.angle - a.angle;
        if (gap < 0) gap += TWO_PI;      // the pair that wraps past -pi
        if (gap >= ideal) continue;      // already roomy enough

        // Capped so a badly bunched node cannot fling its neighbours in one
        // tick; the layout should ease into shape, not snap.
        const step = Math.min(ideal - gap, 0.3) * strength * 0.5;

        // `b` is the one further round, so it rotates counter-clockwise and
        // `a` clockwise, opening the gap between them.
        const bx = -b.dy * step, by = b.dx * step;
        const ax = a.dy * step, ay = -a.dx * step;

        b.q.vx += bx; b.q.vy += by;
        a.q.vx += ax; a.q.vy += ay;

        reactionX -= bx + ax;
        reactionY -= by + ay;
      }

      centre.vx += reactionX;
      centre.vy += reactionY;
    }
  }

  /** Bounding box of the current layout, padded slightly. */
  bounds() {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const id of this.ids) {
      const p = this.pos.get(id);
      if (!p) continue;
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    if (!Number.isFinite(minX)) return { minX: -100, minY: -100, maxX: 100, maxY: 100 };
    const padX = (maxX - minX) * 0.06 + 20;
    const padY = (maxY - minY) * 0.06 + 20;
    return { minX: minX - padX, minY: minY - padY, maxX: maxX + padX, maxY: maxY + padY };
  }
}
