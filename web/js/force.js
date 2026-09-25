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
 * And one thing that is deliberately *not* a force: after the forces have run,
 * the whole drawing is translated so its centre of mass sits at the origin. See
 * _recentre. A force pulling every node toward the origin is proportional to
 * how far out the node is, so it squeezes the rim harder than the middle and
 * changes the shape; a translation moves every node by the same vector and
 * changes nothing but where the drawing sits.
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
    this.pos = new Map();       // id -> {x, y, z, vx, vy, vz}, kept across frames
    this.ids = [];
    // This frame by position in `ids`: each node's entry in `pos`, each edge
    // as its two ends, and each node's neighbours as the run of `_adjList`
    // from `_adjStart[i]` to `_adjStart[i + 1]`. An end outside the frame is
    // -1. Built once by setFrame, so a tick never looks anything up by id.
    this.nodes = [];
    this.pairs = new Int32Array(0);
    this._adjStart = new Int32Array(1);
    this._adjList = new Int32Array(0);
    this.alpha = 1;

    this.dimensions = 2;
    this.charge = 20;
    this.linkStrength = 0.12;
    this.linkDistance = 24;
    this.centerStrength = 0.012;
    this.angularStrength = 0.15;
    this.damping = 0.86;

    // Where the drawing's centre of mass was last carried toward the origin.
    // See _recentre: this is a rigid translation rather than a force, and it is
    // what keeps the graph on screen once the centring force is turned off.
    this._recentredAt = 0;

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

  /**
   * How long the drawing takes to slide back to the origin, as a time
   * constant in seconds. Three of these is effectively all the way, so the
   * glide reads as about three quarters of a second — brisk enough not to
   * feel like lag, slow enough to read as movement rather than a jump.
   */
  static get RECENTRE_TAU() { return 0.25; }

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
   * `ends` lists the edges' endpoint ids, two to an edge, and `parents[i]` is
   * the id that spawned `ids[i]`.
   *
   * Seeding a newborn beside its parent is what keeps lineages together. The
   * parent is not always there to be found, though: consecutive shown frames
   * can be several generations apart — a phase filter halves them, a run that
   * records every Nth iteration thins them further, and fast playback outruns
   * the layout. Whole generations then pass unseen, and a newborn's parent may
   * itself have been born and died between two frames that were drawn.
   *
   * Giving up after one hop dropped those nodes at the origin, which is why a
   * burst of them appeared in the middle of the view whenever playback ran
   * ahead. So the search climbs: parent, grandparent, and up the line until it
   * finds an ancestor that still holds a position. Failing that it borrows
   * from whichever neighbours are already placed, since a newborn is wired to
   * its parent's neighbourhood and that is roughly where it belongs. Only a
   * node with no placed ancestor and no placed neighbour falls back to the
   * middle, which in practice means the very first frame.
   */
  setFrame(ids, ends, parents, carryPositions) {
    if (!carryPositions) this.pos.clear();

    // Where each id sits in this frame. The edges need it to become pairs of
    // positions, and the climb below needs it to follow an ancestor that is
    // itself newly born and has no coordinates of its own yet.
    const slot = new Map();
    for (let i = 0; i < ids.length; i++) slot.set(ids[i], i);

    // Adopted before seeding rather than after, so the neighbour lists the
    // layout needs anyway are already built when the fallback below wants
    // them. Otherwise finding a newborn's neighbours meant a second pass over
    // every edge in the graph.
    this.ids = ids;
    this._index(slot, ends);
    const start = this._adjStart, list = this._adjList;

    const next = new Map();
    const spawnRadius = this.linkDistance * 0.6;

    // Long chains are walked once and the answer shared by everything along
    // them, so a deep lineage costs one climb rather than one per descendant.
    const MAX_CLIMB = 200;
    const climbed = new Map();

    const ancestorAnchor = (startId) => {
      if (startId === undefined || startId < 0) return null;
      // The overwhelmingly common case, and the only one worth keeping free of
      // allocation: frames in order, parent still alive and already placed.
      const direct = this.pos.get(startId);
      if (direct) return direct;
      if (climbed.has(startId)) return climbed.get(startId);

      const chain = [];
      let current = startId, found = null, hops = 0;

      while (current !== undefined && current >= 0 && hops < MAX_CLIMB) {
        const known = this.pos.get(current);
        if (known) { found = known; break; }
        if (climbed.has(current)) { found = climbed.get(current); break; }

        chain.push(current);
        const i = slot.get(current);
        if (i === undefined) break;   // this ancestor is gone from the frame
        current = parents ? parents[i] : -1;
        hops++;
      }

      for (const id of chain) climbed.set(id, found);
      return found;
    };

    const place = (anchor, spread) => {
      const dir = this._randomDirection();
      const radius = spread !== undefined ? spread
        : anchor ? spawnRadius : 200 * Math.sqrt(Math.random());
      return {
        x: (anchor ? anchor.x : 0) + dir.x * radius,
        y: (anchor ? anchor.y : 0) + dir.y * radius,
        z: (anchor ? anchor.z : 0) + dir.z * radius,
        vx: 0, vy: 0, vz: 0
      };
    };

    const unplaced = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      const existing = this.pos.get(id);
      if (existing) { next.set(id, existing); continue; }

      const anchor = ancestorAnchor(parents ? parents[i] : -1);
      if (anchor) next.set(id, place(anchor));
      else unplaced.push(i);
    }

    // Whatever the lineage could not account for, the topology usually can.
    // Averaging the placed neighbours puts the node inside the neighbourhood
    // it is wired to rather than at the centre of the whole graph.
    //
    // Repeated while it keeps making progress, because one pass is not enough.
    // Lineage runs one way through time: stepping backwards — scrubbing left,
    // or the front page playing its recording in reverse — an agent that
    // reappears has a parent who died even earlier, so the climb finds nobody
    // and the whole group arrives unplaced together. Each one that gets placed
    // is an anchor for the next, and a single pass left the rest at the
    // origin: measured at 70 to 168 of about 400 arrivals per backward step,
    // dropped into a ball of radius 200 while the graph reached out to 4,000,
    // which the repulsion then blew apart. That is the explosion.
    if (unplaced.length) {
      // Bounded so a long chain of arrivals cannot turn this into a walk over
      // the frame once per node; whatever is left over after this is scattered
      // through the graph instead, which is cheap and looks like nothing.
      const MAX_PASSES = 12;
      let remaining = unplaced;

      for (let pass = 0; pass < MAX_PASSES && remaining.length; pass++) {
        const still = [];
        let placed = 0;
        for (const i of remaining) {
          let x = 0, y = 0, z = 0, count = 0;
          for (let a = start[i]; a < start[i + 1]; a++) {
            const p = list[a] < 0 ? undefined : next.get(ids[list[a]]);
            if (!p) continue;
            x += p.x; y += p.y; z += p.z; count++;
          }
          if (count) {
            next.set(ids[i], place({ x: x / count, y: y / count, z: z / count }));
            placed++;
          } else {
            still.push(i);
          }
        }
        remaining = still;
        if (!placed) break;      // nothing this pass, so no later pass can help
      }

      // Anything left knows nobody that has been placed — a genuinely new
      // component, or the first frame of a run. Spread through the graph
      // rather than heaped at its middle: a hundred nodes in one small ball
      // repel each other hard enough to look like an explosion, and the middle
      // of the screen is the one place that draws the eye.
      if (remaining.length) {
        let cx = 0, cy = 0, cz = 0, n = 0;
        for (const p of next.values()) { cx += p.x; cy += p.y; cz += p.z; n++; }
        if (n) {
          cx /= n; cy /= n; cz /= n;
          let reach = 0;
          for (const p of next.values()) reach += Math.hypot(p.x - cx, p.y - cy, p.z - cz);
          reach /= n;
          const centre = { x: cx, y: cy, z: cz };
          for (const i of remaining) {
            next.set(ids[i], place(centre, reach * Math.cbrt(Math.random())));
          }
        } else {
          // Nothing placed at all: a fresh layout, which is meant to start as
          // a ball and expand.
          for (const i of remaining) next.set(ids[i], place(null));
        }
      }
    }

    this.pos = next;
    this.nodes = ids.map(id => next.get(id));
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

  /**
   * The frame's edges as pairs of positions in `ids`, and every node's
   * neighbours in edge order, rebuilt once per frame.
   *
   * By position rather than by id because a tick visits every edge for the
   * springs and every spoke for the angular force. Looked up by id, that was
   * two map lookups per edge per tick, and an object for every spoke: at
   * seventy thousand nodes, 9.6ms of the springs and a fifth of the angular
   * force, every tick.
   */
  _index(slot, ends) {
    const n = this.ids.length;
    const pairs = new Int32Array(ends.length & ~1);
    for (let e = 0; e < pairs.length; e++) {
      const at = slot.get(ends[e]);
      pairs[e] = at === undefined ? -1 : at;
    }

    // Counted, then filled, so every neighbour list is one run of one array.
    // A node keeps an edge whose other end is outside the frame, as -1: it
    // still counts toward the degree the angular force is capped by.
    const start = new Int32Array(n + 1);
    for (let e = 0; e < pairs.length; e++) if (pairs[e] >= 0) start[pairs[e] + 1]++;
    let most = 0;
    for (let i = 0; i < n; i++) {
      most = Math.max(most, start[i + 1]);
      start[i + 1] += start[i];
    }
    const list = new Int32Array(start[n]);
    const fill = start.slice(0, n);
    for (let e = 0; e < pairs.length; e += 2) {
      const a = pairs[e], b = pairs[e + 1];
      if (a >= 0) list[fill[a]++] = b;
      if (b >= 0) list[fill[b]++] = a;
    }

    this.pairs = pairs;
    this._adjStart = start;
    this._adjList = list;
    // One node's spokes at a time, for the angular force: nothing is
    // allocated per spoke.
    if (!this._spokes || this._spokes.to.length < most) {
      const f64 = () => new Float64Array(most);
      this._spokes = {
        to: new Int32Array(most), order: [],
        dx: f64(), dy: f64(), angle: f64(),
        dist: f64(), ux: f64(), uy: f64(), uz: f64()
      };
    }
  }

  reheat(alpha = 1) {
    this.alpha = alpha;
  }

  /**
   * Carry out one command from the page, and say whether it moved anything.
   *
   * The worker drives its layout through this, and so does the page when it
   * has to run the layout itself. Each used to spell out a switch of its own,
   * and the two had drifted: the page's copied a force parameter even when a
   * look left it undefined, which then turned every coordinate to NaN.
   */
  apply(command) {
    switch (command.type) {
      case 'frame':
        this.setFrame(command.ids, command.ends, command.parents, command.carry);
        return true;
      case 'dimensions':
        this.setDimensions(command.dimensions);
        return true;
      case 'params':
        for (const [key, value] of Object.entries(command.params)) {
          if (typeof value === 'number') this[key] = value;
        }
        return false;
      case 'reheat':
        this.reheat(command.alpha);
        return false;
      case 'scatter':
        this.scatter();
        return true;
      default:
        return false;
    }
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

  /**
   * Advance the simulation one tick. Returns false once nothing is changing.
   *
   * The forces stop once alpha has decayed, but the recentring does not: a
   * drawing can be settled and still off centre, and a caller that stops
   * publishing the moment the forces rest would leave it there. So the two are
   * asked separately, and the tick reports movement if either of them moved
   * anything. Recentring reaches its own threshold and stops returning true,
   * which is what keeps an idle layout idle rather than spinning.
   */
  tick() {
    const nodes = this.nodes;
    if (!nodes.length) return false;

    if (this.alpha < 0.005) return this._recentre(nodes, this.is3D);

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

    this._recentre(nodes, use3D);

    this.alpha *= 0.985;
    return true;
  }

  /**
   * Slide the whole drawing so its centre of mass sits at the origin.
   *
   * Not a force. Every node moves by the same vector, so no distance between
   * any two nodes changes and no velocity changes — the arrangement is
   * identical, it is merely somewhere else. That is the difference that
   * matters: the centring *force* pulls each node toward the origin in
   * proportion to how far out it already is, which squeezes the rim harder
   * than the middle and flattens the shape. Turning that force off is what
   * lets the graph take the shape the repulsion and the springs agree on, and
   * this is what stops it wandering out of view once nothing is holding it.
   *
   * Eased against the clock rather than per tick, because the number of ticks
   * per drawn frame is a property of the machine: the worker runs as many as
   * fit in its budget, so a per-tick fraction would glide on a slow computer
   * and snap on a fast one. Against elapsed time the drawing takes the same
   * three quarters of a second to settle into place wherever it runs, which is
   * the same easing the camera uses when it follows a graph under Fit view.
   *
   * In the steady state the correction each tick is the drift each tick, which
   * is far too small to see. The easing is for the one moment it is not: a
   * drawing that is already well off-centre when this starts glides in rather
   * than jumping.
   */
  _recentre(nodes, use3D) {
    let cx = 0, cy = 0, cz = 0;
    for (const p of nodes) {
      cx += p.x;
      cy += p.y;
      if (use3D) cz += p.z;
    }
    const n = nodes.length;
    cx /= n; cy /= n; cz /= n;

    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    // Clamped, so a tab that was in the background does not come back and
    // apply a second's worth of correction in one step.
    const dt = Math.min(0.05, Math.max(0, (now - this._recentredAt) / 1000));
    this._recentredAt = now;

    const ease = 1 - Math.exp(-dt / ForceLayout.RECENTRE_TAU);
    const dx = -cx * ease, dy = -cy * ease, dz = use3D ? -cz * ease : 0;

    // Below a thousandth of a link length there is nothing worth correcting,
    // and moving every node by it is a pass over the whole graph for nothing.
    // Reporting no movement here is also what lets an idle layout stay idle:
    // the caller stops publishing once nothing moves, and without a threshold
    // the last vanishing fraction of an offset would keep it awake forever.
    const tiny = this.linkDistance * 1e-3;
    if (Math.abs(dx) < tiny && Math.abs(dy) < tiny && Math.abs(dz) < tiny) return false;

    for (const p of nodes) {
      p.x += dx;
      p.y += dy;
      if (use3D) p.z += dz;
    }
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
    const nodes = this.nodes, pairs = this.pairs;

    for (let e = 0; e < pairs.length; e += 2) {
      if (pairs[e] < 0 || pairs[e + 1] < 0) continue;
      const pa = nodes[pairs[e]];
      const pb = nodes[pairs[e + 1]];

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
    const nodes = this.nodes, start = this._adjStart, list = this._adjList;
    const { to, dx, dy, angle, order } = this._spokes;
    const byAngle = (a, b) => angle[a] - angle[b];

    for (let c = 0; c < nodes.length; c++) {
      if (start[c + 1] - start[c] < 2) continue;
      const centre = nodes[c];

      let spokes = 0;
      for (let at = start[c]; at < start[c + 1]; at++) {
        if (list[at] < 0) continue;
        const q = nodes[list[at]];
        const x = q.x - centre.x;
        const y = q.y - centre.y;
        if (x * x + y * y < 1e-6) continue;
        to[spokes] = list[at]; dx[spokes] = x; dy[spokes] = y;
        angle[spokes] = Math.atan2(y, x);
        spokes++;
      }
      if (spokes < 2) continue;

      order.length = spokes;
      for (let s = 0; s < spokes; s++) order[s] = s;
      order.sort(byAngle);
      const ideal = TWO_PI / spokes;

      let reactionX = 0, reactionY = 0;

      for (let i = 0; i < spokes; i++) {
        const a = order[i];
        const b = order[(i + 1) % spokes];

        let gap = angle[b] - angle[a];
        if (gap < 0) gap += TWO_PI;      // the pair that wraps past -pi
        if (gap >= ideal) continue;      // already roomy enough

        // Capped so a badly bunched node cannot fling its neighbours in one
        // tick; the layout should ease into shape, not snap.
        const step = Math.min(ideal - gap, 0.3) * strength * 0.5;

        const bx = -dy[b] * step, by = dx[b] * step;
        const ax = dy[a] * step, ay = -dx[a] * step;

        const qb = nodes[to[b]], qa = nodes[to[a]];
        qb.vx += bx; qb.vy += by;
        qa.vx += ax; qa.vy += ay;

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
    const nodes = this.nodes, start = this._adjStart, list = this._adjList;
    const { to, dist, ux, uy, uz } = this._spokes;
    const aT = this._aT || (this._aT = new Float64Array(3));
    const bT = this._bT || (this._bT = new Float64Array(3));

    for (let c = 0; c < nodes.length; c++) {
      const degree = start[c + 1] - start[c];
      if (degree < 2 || degree > this.maxAngularDegree) continue;
      const centre = nodes[c];

      let spokes = 0;
      for (let at = start[c]; at < start[c + 1]; at++) {
        if (list[at] < 0) continue;
        const q = nodes[list[at]];
        const dx = q.x - centre.x, dy = q.y - centre.y, dz = q.z - centre.z;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (d < 1e-3) continue;
        to[spokes] = list[at]; dist[spokes] = d;
        ux[spokes] = dx / d; uy[spokes] = dy / d; uz[spokes] = dz / d;
        spokes++;
      }
      if (spokes < 2) continue;

      // Even spacing of d directions on a sphere; exact for 2 and 3, and a
      // reasonable target beyond that.
      const idealCos = Math.cos(Math.min(Math.PI, (Math.PI * 2) / spokes));
      let rx = 0, ry = 0, rz = 0;

      for (let a = 0; a < spokes; a++) {
        for (let b = a + 1; b < spokes; b++) {
          const dot = ux[a] * ux[b] + uy[a] * uy[b] + uz[a] * uz[b];
          if (dot <= idealCos) continue;   // already far enough apart

          // Direction that separates the two spokes.
          let sx = ux[b] - ux[a], sy = uy[b] - uy[a], sz = uz[b] - uz[a];
          const sLen = Math.sqrt(sx * sx + sy * sy + sz * sz);
          if (sLen < 1e-6) {
            // Perfectly coincident spokes: pick any perpendicular to break the tie.
            sx = -uy[a]; sy = ux[a]; sz = 0;
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
          if (!ForceLayout._tangent(sx, sy, sz, ux[b], uy[b], uz[b], bT)) continue;
          if (!ForceLayout._tangent(-sx, -sy, -sz, ux[a], uy[a], uz[a], aT)) continue;

          const bStep = step * dist[b];
          const aStep = step * dist[a];

          const bvx = bT[0] * bStep, bvy = bT[1] * bStep, bvz = bT[2] * bStep;
          const avx = aT[0] * aStep, avy = aT[1] * aStep, avz = aT[2] * aStep;

          const qb = nodes[to[b]], qa = nodes[to[a]];
          qb.vx += bvx; qb.vy += bvy; qb.vz += bvz;
          qa.vx += avx; qa.vy += avy; qa.vz += avz;

          rx -= bvx + avx;
          ry -= bvy + avy;
          rz -= bvz + avz;
        }
      }

      centre.vx += rx; centre.vy += ry; centre.vz += rz;
    }
  }

  /**
   * Unit vector along (x, y, z) with the part parallel to the spoke's
   * direction (ux, uy, uz) removed, written into `out`.
   *
   * Moving a neighbour along this direction turns the spoke without changing
   * its length, which is what keeps the angular force from inflating the
   * drawing. Returns false when the input is purely radial and there is no
   * tangent to speak of.
   */
  static _tangent(x, y, z, ux, uy, uz, out) {
    const radial = x * ux + y * uy + z * uz;
    const tx = x - radial * ux;
    const ty = y - radial * uy;
    const tz = z - radial * uz;

    const len = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (len < 1e-9) return false;
    out[0] = tx / len; out[1] = ty / len; out[2] = tz / len;
    return true;
  }

  /**
   * Copy the positions into a flat array, in `ids` order.
   *
   * The renderer reads this rather than the Map, so it does not care whether
   * the layout ran on this thread or arrived from a worker — both present the
   * same three floats per node in the same order.
   */
  syncPositions(target = null) {
    const n = this.ids.length;
    // Into `target` when given: the worker writes straight into the memory it
    // shares with the page, and used to spell this loop out again to do it.
    let out = target;
    if (!out) {
      if (!this.positions || this.positions.length < n * 3) {
        this.positions = new Float32Array(n * 3);
      }
      out = this.positions;
    }
    for (let i = 0; i < n; i++) {
      const p = this.nodes[i];
      const o = i * 3;
      if (p) { out[o] = p.x; out[o + 1] = p.y; out[o + 2] = p.z; }
    }
    return out;
  }

}
