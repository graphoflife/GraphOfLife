/*
 * Force-directed layout, in two or three dimensions.
 *
 * Written by hand rather than pulled from a CDN so the viewer keeps working
 * offline on any machine you clone the repo onto.
 *
 * Four forces per tick:
 *   repulsion  — every node pushes every other apart, approximated with a
 *                Barnes-Hut tree so the cost is O(n log n) rather than
 *                quadratic. Being all-pairs rather than merely local is what
 *                lets the drawing spread outward instead of staying balled up.
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
 *
 * Every node always carries a z coordinate. In 2D mode it is simply held at
 * zero, which keeps one code path for both modes and lets a switch to 3D lift
 * the existing drawing off the plane instead of starting over.
 */
class ForceLayout {
  constructor() {
    this.pos = new Map();       // id -> {x, y, z, vx, vy, vz}
    this.ids = [];
    this.edges = [];
    this.adjacency = new Map(); // id -> [neighbour ids], for the angular force
    this.alpha = 1;

    this.dimensions = 2;
    this.charge = 20;
    this.linkStrength = 0.12;
    this.linkDistance = 24;
    this.centerStrength = 0.012;
    this.angularStrength = 0.15;
    this.damping = 0.86;

    // Barnes-Hut opening angle: how distant a clump must be before it is
    // treated as one body, and by far the strongest lever on cost — the number
    // of cells each node visits falls roughly with the cube of it.
    //
    // Measured on eighteen thousand nodes: 0.9 visits 174 cells per node and
    // takes 96ms a tick for 3.8% mean force error against exact all-pairs;
    // 1.2 visits 90 for 37ms and 6.7%; 2.0 visits 26 for 14ms and 13.8%. A few
    // percent of force error is invisible in an arrangement that is settling
    // over hundreds of ticks anyway, so the default buys the speed.
    this.theta = 1.2;

    // Tree storage, reused between ticks so a steady graph allocates nothing.
    this._tCapacity = 0;

    // Spreading spokes pairwise costs O(d^2) at a node of degree d. Hubs are
    // pinned by their own edges and barely move anyway, so past this degree the
    // angular force is skipped rather than paid for.
    this.maxAngularDegree = 24;
  }

  get is3D() { return this.dimensions === 3; }

  /**
   * Switch between 2D and 3D.
   *
   * Going up gives every node a small random z so the repulsion has something
   * to work with — starting perfectly coplanar leaves no reason to separate.
   * Going down flattens z back to zero.
   */
  setDimensions(dims) {
    if (dims === this.dimensions) return;
    this.dimensions = dims;

    for (const p of this.pos.values()) {
      if (dims === 3) {
        p.z = (Math.random() - 0.5) * this.linkDistance * 2;
      } else {
        p.z = 0;
      }
      p.vz = 0;
    }
    this.reheat(0.8);
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
      const radius = anchor ? spawnRadius : 200 * Math.sqrt(Math.random());
      const dir = this._randomDirection();

      next.set(id, {
        x: (anchor ? anchor.x : 0) + dir.x * radius,
        y: (anchor ? anchor.y : 0) + dir.y * radius,
        z: (anchor ? anchor.z : 0) + dir.z * radius,
        vx: 0, vy: 0, vz: 0
      });
    }

    this.pos = next;
    this.ids = ids;
    this.edges = edges;
    this._buildAdjacency();
  }

  _randomDirection() {
    const angle = Math.random() * Math.PI * 2;
    if (!this.is3D) {
      return { x: Math.cos(angle), y: Math.sin(angle), z: 0 };
    }
    // Uniform on the sphere: cosine of the polar angle must be uniform, or
    // points bunch at the poles.
    const cosPolar = Math.random() * 2 - 1;
    const sinPolar = Math.sqrt(1 - cosPolar * cosPolar);
    return { x: sinPolar * Math.cos(angle), y: sinPolar * Math.sin(angle), z: cosPolar };
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
      const radius = 250 * Math.sqrt(Math.random());
      const dir = this._randomDirection();
      p.x = dir.x * radius;
      p.y = dir.y * radius;
      p.z = dir.z * radius;
      p.vx = p.vy = p.vz = 0;
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

    const k = this.centerStrength * this.alpha;
    const use3D = this.is3D;

    // No node may cross more than a couple of rest lengths in one tick. This is
    // the backstop: whatever the forces conspire to produce, positions can only
    // grow linearly, so the drawing cannot blow past the point where repulsion
    // stops working and never recovers.
    const maxSpeed = Math.max(1, this.linkDistance * 2);
    const maxSpeedSq = maxSpeed * maxSpeed;

    for (const p of nodes) {
      p.vx -= p.x * k;
      p.vy -= p.y * k;
      p.vx *= this.damping;
      p.vy *= this.damping;

      if (use3D) {
        p.vz -= p.z * k;
        p.vz *= this.damping;
      } else {
        p.z = 0;
        p.vz = 0;
      }

      const speedSq = p.vx * p.vx + p.vy * p.vy + p.vz * p.vz;
      if (speedSq > maxSpeedSq) {
        const brake = maxSpeed / Math.sqrt(speedSq);
        p.vx *= brake;
        p.vy *= brake;
        p.vz *= brake;
      }

      p.x += p.vx;
      p.y += p.vy;
      if (use3D) p.z += p.vz;
    }

    this.alpha *= 0.985;
    return true;
  }

  /**
   * All-pairs repulsion, approximated with a Barnes-Hut tree.
   *
   * Every node pushes on every other, which is what a spring layout of the
   * NetworkX kind does and what makes a drawing spread outward: distant parts
   * still feel each other and push apart, so branches fan out instead of
   * folding back over the middle.
   *
   * Doing it honestly would cost O(n^2). Instead the nodes are bucketed into a
   * tree, and a clump far enough away is treated as a single body sitting at
   * its centre of mass — the standard Barnes-Hut trade, accurate where it
   * matters and O(n log n) overall.
   *
   * The tree lives in flat typed arrays rather than linked objects, and the
   * traversal is a loop over an explicit stack rather than recursion. At
   * eighteen thousand nodes the walk makes a few million visits per tick, and
   * at that volume the per-visit overhead — a function call, an allocated
   * iterator over a children array, a division to recover the centre of mass —
   * costs more than the arithmetic it wraps. The arrays are kept between ticks
   * and grown only when a frame needs more room, so a steady graph allocates
   * nothing at all.
   */
  _repel(nodes) {
    const strength = this.charge * this.alpha;
    if (strength <= 0) return;

    const cells = this._buildTree(nodes);
    if (cells <= 0) return;

    // Repulsion goes as 1/distance^2, so a pair that is almost coincident gets
    // an unbounded kick. Flooring the distance at a fraction of the rest length
    // caps that at a force the springs can still answer.
    const minDistSq = Math.max(1, (this.linkDistance * 0.2) ** 2);
    const thetaSq = this.theta * this.theta;
    const use3D = this.is3D;

    // Hoisted out of the loop: property lookups on `this` are not free when
    // they happen millions of times.
    const size = this._tSize, mass = this._tMass, body = this._tBody;
    const cx = this._tCx, cy = this._tCy, cz = this._tCz;
    const kids = this._tKids;

    let stack = this._tStack;
    if (!stack || stack.length < 64 * 8) stack = this._tStack = new Int32Array(64 * 8);

    for (let i = 0; i < nodes.length; i++) {
      const p = nodes[i];
      const px = p.x, py = p.y, pz = p.z;
      let fx = 0, fy = 0, fz = 0;

      let top = 0;
      stack[top++] = 0;

      while (top > 0) {
        const c = stack[--top];
        const m = mass[c];
        if (m === 0) continue;

        const b = body[c];
        let dx, dy, dz, distSq, weight;

        if (b >= 0) {
          if (b === i) continue;
          const q = nodes[b];
          dx = px - q.x; dy = py - q.y; dz = use3D ? pz - q.z : 0;
          distSq = dx * dx + dy * dy + dz * dz;
          weight = 1;
        } else {
          dx = px - cx[c]; dy = py - cy[c]; dz = use3D ? pz - cz[c] : 0;
          distSq = dx * dx + dy * dy + dz * dz;

          // Too close to summarise: open the cell and look at its children.
          if (!(distSq > 0 && size[c] * size[c] < thetaSq * distSq)) {
            const base = c << 3;
            for (let k = 0; k < 8; k++) {
              const kid = kids[base + k];
              if (kid > 0) {
                if (top >= stack.length) {
                  const bigger = new Int32Array(stack.length * 2);
                  bigger.set(stack);
                  stack = this._tStack = bigger;
                }
                stack[top++] = kid;
              }
            }
            continue;
          }
          weight = m;
        }

        // Two bodies exactly on top of each other have no direction to
        // separate along, so nudge them apart randomly.
        if (distSq < 1e-9) {
          dx = (Math.random() - 0.5) * 0.1;
          dy = (Math.random() - 0.5) * 0.1;
          dz = use3D ? (Math.random() - 0.5) * 0.1 : 0;
          distSq = dx * dx + dy * dy + dz * dz;
        }

        const force = (strength * weight) / (distSq < minDistSq ? minDistSq : distSq);
        fx += dx * force;
        fy += dy * force;
        if (use3D) fz += dz * force;
      }

      p.vx += fx;
      p.vy += fy;
      if (use3D) p.vz += fz;
    }
  }

  /**
   * Bucket the nodes into a quad- or octree held in flat arrays.
   *
   * Cell 0 is the root. `_tBody` holds the index of the single node a leaf
   * carries, or -1 once the cell has been split. Children are eight slots per
   * cell, unused ones left at 0 — cell 0 being the root means 0 doubles as
   * "no child". Returns how many cells were used.
   */
  _buildTree(nodes) {
    const n = nodes.length;
    if (!n) return 0;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const p = nodes[i];
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
    }
    if (!Number.isFinite(minX)) return 0;

    const use3D = this.is3D;
    const extent = Math.max(maxX - minX, maxY - minY, use3D ? maxZ - minZ : 0, 1e-6);

    // A split can add a cell per level per node; this is ample and reused.
    this._ensureTreeCapacity(Math.max(64, n * 4));

    const size = this._tSize, mass = this._tMass, body = this._tBody;
    const cx = this._tCx, cy = this._tCy, cz = this._tCz;
    const mx = this._tMx, my = this._tMy, mz = this._tMz;
    const kids = this._tKids;

    // Root
    let used = 1;
    cx[0] = (minX + maxX) / 2;
    cy[0] = (minY + maxY) / 2;
    cz[0] = use3D ? (minZ + maxZ) / 2 : 0;
    size[0] = extent;
    mass[0] = 0; mx[0] = 0; my[0] = 0; mz[0] = 0; body[0] = -1;
    kids.fill(0, 0, 8);

    // Allocate a child cell in `slot` of `parent`, or return the one already
    // there. Written out rather than returning a coordinate tuple, so the hot
    // path allocates nothing.
    const childOf = (parent, slot) => {
      const at = (parent << 3) + slot;
      const existing = this._tKids[at];
      if (existing !== 0) return existing;

      if (used >= this._tCapacity) this._ensureTreeCapacity(this._tCapacity * 2);
      const c = used++;
      const quarter = this._tSize[parent] / 4;

      this._tCx[c] = this._tCx[parent] + ((slot & 1) ? quarter : -quarter);
      this._tCy[c] = this._tCy[parent] + ((slot & 2) ? quarter : -quarter);
      this._tCz[c] = use3D ? this._tCz[parent] + ((slot & 4) ? quarter : -quarter) : 0;
      this._tSize[c] = this._tSize[parent] / 2;
      this._tMass[c] = 0; this._tMx[c] = 0; this._tMy[c] = 0; this._tMz[c] = 0;
      this._tBody[c] = -1;
      this._tKids.fill(0, c << 3, (c << 3) + 8);

      this._tKids[at] = c;
      return c;
    };

    for (let i = 0; i < n; i++) {
      const p = nodes[i];
      let c = 0;
      let depth = 0;

      // Walk down, splitting as needed, until the node lands somewhere.
      for (;;) {
        this._tMass[c] += 1;
        this._tMx[c] += p.x; this._tMy[c] += p.y; this._tMz[c] += p.z;

        // Coincident points would subdivide forever.
        if (depth > 20) break;

        // An empty leaf simply takes the node.
        if (this._tBody[c] === -1 && this._tMass[c] === 1) { this._tBody[c] = i; break; }

        const occupant = this._tBody[c];
        if (occupant >= 0) {
          // Split: push the sitting tenant one level down first.
          this._tBody[c] = -1;
          const q = nodes[occupant];
          const kid = childOf(c, this._slotFor(c, q, use3D));
          this._tMass[kid] += 1;
          this._tMx[kid] += q.x; this._tMy[kid] += q.y; this._tMz[kid] += q.z;
          this._tBody[kid] = occupant;
        }

        c = childOf(c, this._slotFor(c, p, use3D));
        depth++;
      }
    }

    // Turn the running sums into actual centres of mass, once, so the walk
    // never has to divide.
    for (let c = 0; c < used; c++) {
      const m = this._tMass[c];
      if (m > 0) { this._tMx[c] /= m; this._tMy[c] /= m; this._tMz[c] /= m; }
    }
    // The traversal reads centres from cx/cy/cz for split cells.
    for (let c = 0; c < used; c++) {
      if (this._tBody[c] < 0) {
        this._tCx[c] = this._tMx[c]; this._tCy[c] = this._tMy[c]; this._tCz[c] = this._tMz[c];
      }
    }
    return used;
  }

  _slotFor(cell, p, use3D) {
    let slot = (p.x > this._tCx[cell] ? 1 : 0) | (p.y > this._tCy[cell] ? 2 : 0);
    if (use3D && p.z > this._tCz[cell]) slot |= 4;
    return slot;
  }

  _ensureTreeCapacity(capacity) {
    if (this._tCapacity >= capacity) return;
    const grow = (old, size) => { const a = new Float64Array(size); if (old) a.set(old); return a; };

    this._tCx = grow(this._tCx, capacity);
    this._tCy = grow(this._tCy, capacity);
    this._tCz = grow(this._tCz, capacity);
    this._tSize = grow(this._tSize, capacity);
    this._tMass = grow(this._tMass, capacity);
    this._tMx = grow(this._tMx, capacity);
    this._tMy = grow(this._tMy, capacity);
    this._tMz = grow(this._tMz, capacity);

    const body = new Int32Array(capacity);
    if (this._tBody) body.set(this._tBody);
    this._tBody = body;

    const kids = new Int32Array(capacity * 8);
    if (this._tKids) kids.set(this._tKids);
    this._tKids = kids;

    this._tCapacity = capacity;
  }

  _springs() {
    const strength = this.linkStrength * this.alpha;
    const use3D = this.is3D;

    for (const [a, b] of this.edges) {
      const pa = this.pos.get(a);
      const pb = this.pos.get(b);
      if (!pa || !pb) continue;

      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const dz = use3D ? pb.z - pa.z : 0;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001;
      const push = (dist - this.linkDistance) / dist * strength;

      const fx = dx * push * 0.5;
      const fy = dy * push * 0.5;
      const fz = dz * push * 0.5;
      pa.vx += fx; pa.vy += fy;
      pb.vx -= fx; pb.vy -= fy;
      if (use3D) { pa.vz += fz; pb.vz -= fz; }
    }
  }

  _angularSpread() {
    if (this.angularStrength * this.alpha <= 0) return;
    if (this.is3D) this._angularSpread3D();
    else this._angularSpread2D();
  }

  /**
   * Angular resolution in the plane: spread each node's incident edges evenly.
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
  _angularSpread2D() {
    const strength = this.angularStrength * this.alpha;
    const TWO_PI = Math.PI * 2;

    for (const [id, neighbours] of this.adjacency) {
      if (neighbours.length < 2) continue;
      const centre = this.pos.get(id);
      if (!centre) continue;

      const spokes = [];
      for (const nid of neighbours) {
        const q = this.pos.get(nid);
        if (!q) continue;
        const dx = q.x - centre.x;
        const dy = q.y - centre.y;
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

  /**
   * The same idea on a sphere.
   *
   * Sorting by angle has no meaning in three dimensions, so instead each pair
   * of spokes repels along the sphere: any two that sit closer than the ideal
   * 2*pi/d separation get pushed apart tangentially, leaving edge lengths to
   * the springs. Two spokes settle antipodally at 180 degrees and three settle
   * into a plane at 120, matching the flat case exactly.
   *
   * Past three the target is a floor rather than a unique arrangement, so the
   * result is whichever valid configuration is nearest to hand: four spokes
   * typically settle square and planar at 90 degrees rather than into a
   * tetrahedron, since coming from a flat drawing that is the closer
   * equilibrium and both satisfy "no pair closer than the ideal".
   */
  _angularSpread3D() {
    const strength = this.angularStrength * this.alpha;

    for (const [id, neighbours] of this.adjacency) {
      const degree = neighbours.length;
      if (degree < 2 || degree > this.maxAngularDegree) continue;
      const centre = this.pos.get(id);
      if (!centre) continue;

      const spokes = [];
      for (const nid of neighbours) {
        const q = this.pos.get(nid);
        if (!q) continue;
        const dx = q.x - centre.x, dy = q.y - centre.y, dz = q.z - centre.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < 1e-3) continue;
        spokes.push({ q, dist, ux: dx / dist, uy: dy / dist, uz: dz / dist });
      }
      if (spokes.length < 2) continue;

      // Even spacing of d directions on a sphere; exact for 2 and 3, and a
      // reasonable target beyond that.
      const idealCos = Math.cos(Math.min(Math.PI, (Math.PI * 2) / spokes.length));
      let rx = 0, ry = 0, rz = 0;

      for (let i = 0; i < spokes.length; i++) {
        for (let j = i + 1; j < spokes.length; j++) {
          const a = spokes[i], b = spokes[j];
          const dot = a.ux * b.ux + a.uy * b.uy + a.uz * b.uz;
          if (dot <= idealCos) continue;   // already far enough apart

          // Direction that separates the two spokes.
          let sx = b.ux - a.ux, sy = b.uy - a.uy, sz = b.uz - a.uz;
          const sLen = Math.sqrt(sx * sx + sy * sy + sz * sz);
          if (sLen < 1e-6) {
            // Perfectly coincident spokes: pick any perpendicular to break the tie.
            sx = -a.uy; sy = a.ux; sz = 0;
            const fallback = Math.sqrt(sx * sx + sy * sy) || 1;
            sx /= fallback; sy /= fallback;
          } else {
            sx /= sLen; sy /= sLen; sz /= sLen;
          }

          const step = Math.min(dot - idealCos, 0.5) * strength * 0.5;

          // Each end moves along its OWN tangent plane. Without projecting out
          // the radial component the separation direction also lengthens the
          // spoke, and because the step scales with distance that feeds back on
          // itself — the drawing inflates without bound instead of settling.
          const bT = ForceLayout._tangent(sx, sy, sz, b);
          const aT = ForceLayout._tangent(-sx, -sy, -sz, a);
          if (!bT || !aT) continue;

          const bStep = step * b.dist;
          const aStep = step * a.dist;

          const bvx = bT.x * bStep, bvy = bT.y * bStep, bvz = bT.z * bStep;
          const avx = aT.x * aStep, avy = aT.y * aStep, avz = aT.z * aStep;

          b.q.vx += bvx; b.q.vy += bvy; b.q.vz += bvz;
          a.q.vx += avx; a.q.vy += avy; a.q.vz += avz;

          rx -= bvx + avx;
          ry -= bvy + avy;
          rz -= bvz + avz;
        }
      }

      centre.vx += rx; centre.vy += ry; centre.vz += rz;
    }
  }

  /**
   * Unit vector along (x, y, z) with the part parallel to `spoke` removed.
   *
   * Moving a neighbour along this direction turns the spoke without changing
   * its length, which is what keeps the angular force from inflating the
   * drawing. Returns null when the input is purely radial and there is no
   * tangent to speak of.
   */
  static _tangent(x, y, z, spoke) {
    const radial = x * spoke.ux + y * spoke.uy + z * spoke.uz;
    const tx = x - radial * spoke.ux;
    const ty = y - radial * spoke.uy;
    const tz = z - radial * spoke.uz;

    const len = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (len < 1e-9) return null;
    return { x: tx / len, y: ty / len, z: tz / len };
  }

  /** Bounding box of the current layout, padded slightly. */
  bounds() {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const id of this.ids) {
      const p = this.pos.get(id);
      if (!p) continue;
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.z < minZ) minZ = p.z;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
      if (p.z > maxZ) maxZ = p.z;
    }
    if (!Number.isFinite(minX)) {
      return { minX: -100, minY: -100, minZ: 0, maxX: 100, maxY: 100, maxZ: 0 };
    }

    const padX = (maxX - minX) * 0.06 + 20;
    const padY = (maxY - minY) * 0.06 + 20;
    return {
      minX: minX - padX, minY: minY - padY, minZ,
      maxX: maxX + padX, maxY: maxY + padY, maxZ
    };
  }
}
