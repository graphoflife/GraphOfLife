/*
 * The landing page: a title over a world that is actually running.
 *
 * The background is a real run of the engine, played back rather than computed
 * here. Computing it in the page was the obvious thing to try and is the wrong
 * thing to ship: a hundred iterations of a world this size takes minutes even
 * in native Python, and in the browser it would first have to fetch a Python
 * runtime to do it with. A visitor would meet an empty page. Worse, on a
 * machine that has gol_server.py running, asking the backend for a world would
 * leave a stray run on disk every time anyone opened the front page.
 *
 * So the run is recorded once by tools/record_home_run.py and shipped with the
 * site, and this file only plays it: hand each frame to the same force layout
 * and the same renderer the Viewer uses, turn the camera at a constant rate,
 * and walk back and forth through the recording.
 *
 * Nothing here is interactive. The canvas takes no pointer events at all, so a
 * drag scrolls the page rather than half-orbiting a camera that will not
 * respond, and the whole loop stops the moment another tab is opened.
 */
const Home = {
  SOURCE: 'data/home-run.bin',

  /** Frames per second — the slowest the Viewer's own playback offers. */
  SPEED: 1,

  /** Radians per second of yaw. Slow enough to read as drift, not spin. */
  ROTATION: 0.06,

  /**
   * Framing, as the padding handed to fitToContent. Negative means the content
   * is fitted to a box larger than the canvas, which is to say: zoomed in, with
   * the graph running off the edges the way a backdrop should.
   */
  ZOOM: -0.13,

  frames: [],
  index: 0,
  direction: 1,
  accumulator: 0,
  lastTime: 0,

  renderer: null,
  layout: null,
  metrics: null,
  settings: null,

  started: false,
  active: false,
  loaded: false,
  failed: false,

  init() {
    this.canvas = document.getElementById('homeCanvas');
    this.stage = document.querySelector('.home-stage');
    if (!this.canvas) return;

    // The Viewer's default look, minus the furniture. A legend on a backdrop
    // is a label for something nobody is reading.
    this.settings = Metrics.migrateSettings({ ...Presets.builtIn('default') });
    Object.assign(this.settings, {
      showLegend: false,
      showEdgeLegend: false,
      autoFit: false,        // framed here instead, so it can be held closer
      layoutCarry: true,
      // Quieter than the Viewer's. At five thousand agents the default edge
      // opacity makes a bright mesh, and white text on a bright mesh is not
      // text. The agents stay as they are, so the structure still reads.
      edgeAlpha: 0.3,
      nodeAlpha: 0.9
    });

    this.renderer = new GraphRenderer(this.canvas);
    this.renderer.setMode3D(this.settings.dimensions === 3);

    this.layout = new LayoutClient();
    this.layout.setDimensions(this.settings.dimensions);
    this.layout.setParams({
      charge: this.settings.forceCharge,
      linkStrength: this.settings.forceLink,
      centerStrength: this.settings.forceCenter,
      angularStrength: this.settings.forceAngular,
      damping: this.settings.forceDamping,
      theta: this.settings.forceTheta
    });

    if (window.ResizeObserver) {
      this._observer = new ResizeObserver(() => this.resize());
      this._observer.observe(this.stage || this.canvas.parentElement);
    }
    window.addEventListener('resize', () => this.resize());

    this.setActive(App.view === 'home');
    if (!this.started) {
      this.started = true;
      requestAnimationFrame(t => this.animate(t));
    }
  },

  /**
   * Opened or left.
   *
   * The recording is only fetched the first time the page is actually looked
   * at, and the loop does no work at all while another tab is open — the point
   * of a backdrop is that it costs nothing when nobody can see it.
   */
  setActive(active) {
    this.active = active;
    if (!active) return;

    // Replay the titles by taking the class off and putting it back, which is
    // the only way to restart a CSS animation.
    if (this.stage) {
      this.stage.classList.remove('play');
      void this.stage.offsetWidth;
      this.stage.classList.add('play');
    }

    this.resize();
    // Retried on each visit rather than given up on for good: a backdrop that
    // missed its fetch once should come back when the page is opened again.
    if (!this.loaded && !this._loading) this.load();
  },

  async load() {
    this._loading = true;
    try {
      // Not force-cache. That will happily reuse a cached failure, and a 404
      // from before the recording existed then sticks for the life of the
      // browser cache — which is exactly what happened the first time.
      const response = await fetch(this.SOURCE);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.frames = this.decode(await response.arrayBuffer());
      if (!this.frames.length) throw new Error('the recording has no frames');

      this.loaded = true;
      this.index = 0;
      this.adopt(this.frames[0], false);
      // Framed once the layout has had a moment to spread out; before that the
      // graph is still a ball and fitting to it zooms absurdly far in.
      this._needsFraming = true;
    } catch (err) {
      // A missing backdrop is not worth an error message on a landing page.
      // The title and the note stand on their own.
      this.failed = true;
      console.warn('home backdrop unavailable:', err.message);
    } finally {
      this._loading = false;
    }
  },

  /**
   * Unpack what tools/record_home_run.py wrote.
   *
   * Four things per frame and nothing else: who exists, what they hold, who
   * their parent was, and who they are joined to. Everything is a varint, ids
   * are stored as the gaps between them once sorted, and edges come as an
   * adjacency list of gaps rather than as pairs of ids — which is the
   * difference between 0.74 MB a frame and 0.17.
   */
  decode(buffer) {
    const bytes = new Uint8Array(buffer);
    let at = 0;

    const take = () => {
      let value = 0, shift = 0;
      for (;;) {
        const byte = bytes[at++];
        value += (byte & 0x7F) * Math.pow(2, shift);
        if (!(byte & 0x80)) return value;
        shift += 7;
      }
    };

    if (bytes[0] !== 0x47 || bytes[1] !== 0x4F || bytes[2] !== 0x4C || bytes[3] !== 0x48) {
      throw new Error('not a home recording');
    }
    if (bytes[4] !== 1) throw new Error(`unknown recording version ${bytes[4]}`);
    at = 5;

    const frames = [];
    const count = take();
    for (let f = 0; f < count; f++) {
      const n = take();

      const ids = new Array(n);
      let previous = 0;
      for (let i = 0; i < n; i++) { previous += take(); ids[i] = previous; }

      const tokens = new Array(n);
      for (let i = 0; i < n; i++) tokens[i] = take();

      // Zero means no parent; anything else is a zigzagged step from the
      // child's own position, shifted up by one to leave zero spare.
      const parents = new Array(n);
      for (let i = 0; i < n; i++) {
        const raw = take();
        if (raw === 0) { parents[i] = null; continue; }
        const zig = raw - 1;
        parents[i] = ids[i + ((zig % 2 === 0) ? (zig / 2) : -((zig + 1) / 2))];
      }

      const edges = [];
      for (let i = 0; i < n; i++) {
        let neighbour = i;
        for (let k = take(); k > 0; k--) {
          neighbour += take();
          edges.push([ids[i], ids[neighbour]]);
        }
      }

      frames.push({ ids, tokens, edges, parent_ids: parents });
    }
    return frames;
  },

  adopt(frame, carry = true) {
    this.frame = frame;
    this.metrics = new FrameMetrics(frame, this.settings);
    this.layout.setFrame(frame.ids, frame.edges, frame.parent_ids, carry);
    this.layout.reheat(carry ? 0.3 : 1);
  },

  resize() {
    if (!this.renderer) return;
    this.renderer.resize();
  },

  /** True once the layout's coordinates belong to the frame we are drawing. */
  get readyToDraw() {
    if (!this.frame || !this.layout) return false;
    if (!this.layout.positionsMatchFrame) return false;
    return this.layout.ids === this.frame.ids;
  },

  animate(time) {
    // Everything is wrapped so the next frame is always asked for. The loop
    // re-arms itself at the end of its own body, which means one thrown error
    // would otherwise stop the backdrop for good rather than for a frame.
    try {
      this.tick(time);
    } catch (err) {
      if (!this._complained) {
        this._complained = true;
        console.warn('home backdrop:', err.message);
      }
    }
    requestAnimationFrame(t => this.animate(t));
  },

  tick(time) {
    const dt = Math.min(0.1, (time - this.lastTime) / 1000) || 0;
    this.lastTime = time;

    if (!this.active || !this.loaded) return;

    // Move to the next frame first, so the check below sees it.
    //
    // Advancing further down — after the check and just before the draw — put
    // one ruined frame on screen every single second. Adopting a frame hands
    // its ids to the layout at once but its coordinates only come back from
    // the worker a moment later, so that draw painted the new frame's edges
    // against the previous frame's positions: every edge joining whichever
    // agents happened to hold those slots. Measured at 4,879 nodes drawn
    // against 4,777 nodes' worth of coordinates, once per iteration, which is
    // exactly the flicker.
    this.accumulator += dt;
    if (this.accumulator >= 1 / this.SPEED) {
      this.accumulator = 0;
      this.advance();
    }

    this.layout.tick();

    // The camera does not depend on which frame is showing, so it keeps
    // gliding through the wait rather than stalling for it.
    this.renderer.rotate(this.ROTATION * dt, 0);
    this.renderer.stepCamera();

    // Until the coordinates belong to the frame we are holding, the canvas
    // simply keeps what it already shows.
    if (!this.readyToDraw || !(this.renderer.cssWidth > 0)) return;

    // Hold the framing as the graph breathes, rather than chasing every
    // change: a camera that re-fits on each frame reads as flinching.
    if (this._needsFraming && this.layout.positions && this.layout.positions.length > 3) {
      this.renderer.fitToContent(this.layout, this.ZOOM, true);
      this._needsFraming = false;
      this._sinceFraming = 0;
    }
    this._sinceFraming = (this._sinceFraming || 0) + dt;
    if (this._sinceFraming > 2) {
      this.renderer.fitToContent(this.layout, this.ZOOM);
      this._sinceFraming = 0;
    }

    this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
  },

  /**
   * One frame onward, turning round at each end.
   *
   * Back and forth rather than looping, because a recording that jumps from
   * its last frame to its first is a cut, and there is nothing to cut to on a
   * page whose whole job is to look continuous.
   */
  advance() {
    if (this.frames.length < 2) return;
    let next = this.index + this.direction;
    if (next >= this.frames.length || next < 0) {
      this.direction *= -1;
      next = this.index + this.direction;
    }
    this.index = next;
    this.adopt(this.frames[this.index], true);
  }
};
