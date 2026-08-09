/*
 * The page's handle on the layout, wherever it happens to be running.
 *
 * Presents the same small surface either way — `ids`, `positions`, and the
 * handful of commands the viewer issues — so the renderer and the viewer do
 * not need to know whether a worker is involved.
 *
 * If workers are unavailable, or one fails to start, it falls back to running
 * the layout on this thread exactly as before. That path is slower on a large
 * graph but always correct, and it keeps the viewer working from a file:// URL
 * or anywhere else a worker cannot be loaded.
 */
class LayoutClient {
  constructor(workerUrl = 'js/layout-worker.js') {
    this.ids = [];
    this.positions = new Float32Array(0);
    this.count = 0;
    this.alpha = 1;
    this.dimensions = 3;
    this.usingWorker = false;
    this.shared = null;

    // Mirrors of the force parameters, so a value set before the worker is
    // ready is not lost, and so the fallback layout can be configured the
    // same way.
    this.params = {
      charge: 20, linkStrength: 0.12, linkDistance: 24,
      centerStrength: 0.012, angularStrength: 0.15,
      damping: 0.86, theta: 1.2
    };

    this._local = null;
    this._pending = null;

    // A newly allocated shared buffer starts empty. Switching to it before the
    // worker has filled it would show every node at the origin for a frame, so
    // the new buffer is held aside and adopted only once positions tagged with
    // its generation come back.
    this._pendingShared = null;
    this._bufferGen = 0;

    // Positions are an array of coordinates in the order of a particular
    // frame's ids. Adopting a new frame while the worker is still mid-batch on
    // the previous one would draw the new ids against the old ordering — slot
    // 3 holding some other node's coordinates — which paints one frame of
    // nonsense. These counters say whether what we hold matches what we are
    // about to draw.
    this._frameGen = 0;
    this._positionsGen = 0;

    try {
      this.worker = new Worker(workerUrl);
      this.worker.onmessage = (e) => this._onMessage(e.data);
      this.worker.onerror = () => this._fallBack('the layout worker failed to start');
      this.usingWorker = true;
      this.worker.postMessage({ type: 'init', buffer: null });
    } catch (err) {
      this._fallBack('workers are unavailable here');
    }
  }

  /** Run on this thread instead, keeping whatever state we already have. */
  _fallBack(reason) {
    if (this._local) return;
    console.warn(`GraphOfLife: ${reason}; running the layout on the main thread.`);
    if (this.worker) { try { this.worker.terminate(); } catch (e) { /* already gone */ } }
    this.worker = null;
    this.usingWorker = false;

    this._local = new ForceLayout();
    Object.assign(this._local, this.params);
    this._local.dimensions = this.dimensions;

    if (this._pending) {
      const f = this._pending;
      this._local.setFrame(f.ids, f.edges, f.parents, f.carry);
      this._syncLocal();
    }
  }

  /** True when the positions we hold belong to the frame we are drawing. */
  get positionsMatchFrame() {
    return this._positionsGen === this._frameGen;
  }

  _onMessage(msg) {
    if (msg.type === 'ready') {
      this.shared = msg.shared ? this.shared : null;
      return;
    }
    if (msg.type !== 'positions') return;

    // The worker has now written into the buffer it reports, so it is safe to
    // read. Until then we keep drawing from the previous one.
    if (this._pendingShared && msg.gen === this._bufferGen) {
      this.shared = this._pendingShared;
      this._pendingShared = null;
    }

    if (msg.frameGen !== undefined) this._positionsGen = msg.frameGen;

    this.alpha = msg.alpha;
    this.count = msg.count;
    if (msg.positions) this.positions = msg.positions;
    else if (this.shared) this.positions = this.shared;
  }

  // ------------------------------------------------------------------
  // Commands
  // ------------------------------------------------------------------

  setFrame(ids, edges, parents, carry) {
    this.ids = ids;
    this.count = ids.length;
    this._pending = { ids, edges, parents, carry };

    if (this._local) {
      this._frameGen++;
      this._local.setFrame(ids, edges, parents, carry);
      this._syncLocal();
      return;
    }

    // Grow the shared buffer to fit, if we are using one. Any growth is
    // pending until the worker fills it, so nothing here changes what is
    // currently being drawn.
    this._ensureShared(ids.length);
    this._frameGen++;
    this.worker.postMessage({ type: 'frame', ids, edges, parents, carry, gen: this._frameGen });
  }

  setDimensions(dims) {
    this.dimensions = dims;
    if (this._local) { this._local.setDimensions(dims); this._syncLocal(); return; }
    this.worker.postMessage({ type: 'dimensions', dimensions: dims });
  }

  setParams(params) {
    Object.assign(this.params, params);
    if (this._local) { Object.assign(this._local, params); return; }
    this.worker.postMessage({ type: 'params', params });
  }

  reheat(alpha = 1) {
    this.alpha = alpha;
    if (this._local) { this._local.reheat(alpha); return; }
    this.worker.postMessage({ type: 'reheat', alpha });
  }

  scatter() {
    if (this._local) { this._local.scatter(); this._syncLocal(); return; }
    this.worker.postMessage({ type: 'scatter' });
  }

  /**
   * Advance the layout, when it is running on this thread.
   *
   * With a worker this does nothing: the layout is already advancing on its
   * own, and the page's only job is to draw whatever has arrived.
   */
  tick() {
    if (!this._local) return false;
    const moved = this._local.tick();
    this._syncLocal();
    return moved;
  }

  _syncLocal() {
    this._positionsGen = this._frameGen;
    this.positions = this._local.syncPositions();
    this.ids = this._local.ids;
    this.count = this._local.ids.length;
    this.alpha = this._local.alpha;
  }

  /**
   * Shared memory needs the page to be cross-origin isolated, which the server
   * arranges with two headers. Without it, positions come back as copies —
   * a couple of hundred kilobytes a frame, which is affordable.
   */
  _ensureShared(nodeCount) {
    if (typeof SharedArrayBuffer === 'undefined' || !self.crossOriginIsolated) return;

    const needed = Math.max(1, nodeCount) * 3;
    if (this.shared && this.shared.length >= needed) return;

    // Held aside rather than adopted: `positions` keeps pointing at the old
    // buffer, which still has valid coordinates in it, until the worker
    // confirms it has filled the new one.
    const buffer = new SharedArrayBuffer(Math.ceil(needed * 1.5) * 4);
    this._pendingShared = new Float32Array(buffer);
    this._bufferGen++;
    this.worker.postMessage({ type: 'buffer', buffer, gen: this._bufferGen });
  }

  /** Bounding box of the current positions, padded slightly. */
  bounds() {
    const n = Math.min(this.count, Math.floor(this.positions.length / 3));
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < n; i++) {
      const o = i * 3;
      const x = this.positions[o], y = this.positions[o + 1], z = this.positions[o + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    if (!Number.isFinite(minX)) {
      return { minX: -100, minY: -100, minZ: 0, maxX: 100, maxY: 100, maxZ: 0 };
    }

    const padX = (maxX - minX) * 0.06 + 20;
    const padY = (maxY - minY) * 0.06 + 20;
    return { minX: minX - padX, minY: minY - padY, minZ,
             maxX: maxX + padX, maxY: maxY + padY, maxZ };
  }

  dispose() {
    if (this.worker) this.worker.postMessage({ type: 'stop' });
  }
}
