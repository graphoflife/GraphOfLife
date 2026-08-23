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

  /**
   * Draws a second, at most.
   *
   * This is scenery behind a title, not something anyone is reading frame by
   * frame, and at sixty it was asking a browser to redraw five thousand agents
   * and seven thousand edges as fast as it possibly could. Thirty halves that
   * for no visible difference to a graph turning this slowly, and leaves room
   * for browsers that put their own work between us and the canvas.
   */
  MAX_FPS: 30,

  /**
   * The floor it will drop to on a browser that cannot keep up.
   *
   * How fast a canvas is varies more between browsers than it has any right
   * to — the same page that is comfortable in one can stutter in another that
   * puts its own work between the drawing and the screen. Rather than pick a
   * rate that suits this machine and hope, the loop watches how long its own
   * draws take and slows down if they are expensive. A backdrop turning at a
   * sixteenth of a radian a second still reads perfectly at twelve.
   */
  MIN_FPS: 12,

  fps: 30,

  frames: [],
  buffer: null,
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
      // Quieter than the Viewer's. At several thousand agents the default edge
      // opacity makes a bright mesh, and white text on a bright mesh is not
      // text. The agents stay as they are, so the structure still reads.
      edgeAlpha: 0.52,
      nodeAlpha: 1,
      // The page's own colour. The veil above the canvas is mixed to match it
      // exactly, so it darkens the graph without ever tinting the background.
      bgColorA: '#0d1117',
      bgColorB: '#0d1117'
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
    if (!active) {
      this.unload();
      return;
    }

    // Replay the titles by taking the class off and putting it back, which is
    // the only way to restart a CSS animation.
    if (this.stage) {
      this.stage.classList.remove('play');
      void this.stage.offsetWidth;
      this.stage.classList.add('play');
    }

    this.build();
    this.resize();
    if (!this.loaded && !this._loading) this.load();
  },

  /** The renderer and the layout worker, made on demand. */
  build() {
    if (!this.renderer) {
      this.renderer = new GraphRenderer(this.canvas);
      this.renderer.setMode3D(this.settings.dimensions === 3);
      // One device pixel per CSS pixel is plenty for scenery. On a retina
      // screen the default would be four times the area to fill, every frame,
      // for a picture nobody is inspecting — and it is fill rate that runs out
      // first on a browser without much canvas headroom.
      this.renderer.maxDpr = 1;
    }
    if (!this.layout) {
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
      this._needsFraming = true;
    }
  },

  /**
   * Give back everything the backdrop was holding.
   *
   * A worker thread running a force layout over five thousand agents does not
   * stop being expensive because nobody is looking at it, so leaving one alive
   * behind the Viewer would be a tax on the rest of the page. The frames go
   * too — decoded they are the largest thing here by far.
   *
   * The packed buffer stays. It is a megabyte and a half, it is what the
   * frames are decoded from, and keeping it is what makes coming back
   * instant instead of another download.
   */
  unload() {
    if (this.layout) {
      this.layout.dispose();
      this.layout = null;
    }
    this.frames = [];
    this.frame = null;
    this.metrics = null;
    this.loaded = false;
    this.index = 0;
    this.direction = 1;
    this.accumulator = 0;

    if (this.renderer && this.renderer.ctx && this.renderer.cssWidth > 0) {
      this.renderer.ctx.clearRect(0, 0, this.renderer.cssWidth, this.renderer.cssHeight);
    }
  },

  async load() {
    this._loading = true;
    try {
      // Not force-cache. That will happily reuse a cached failure, and a 404
      // from before the recording existed then sticks for the life of the
      // browser cache — which is exactly what happened the first time.
      if (!this.buffer) {
        const response = await fetch(this.SOURCE);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.buffer = await response.arrayBuffer();
      }
      this.frames = this.decode(this.buffer);
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
    // Never negative. A timestamp that goes backwards — which happens when
    // a tab is restored, and whenever the loop is driven by hand — would
    // otherwise turn the camera the wrong way and rewind playback.
    const dt = Math.max(0, Math.min(0.1, (time - this.lastTime) / 1000)) || 0;
    this.lastTime = time;

    if (!this.active || !this.loaded || !this.layout || !this.renderer) return;

    // Move to the next frame first, so the check below sees it.
    //
    // Advancing further down — after the check and just before the draw — put
    // one ruined frame on screen every single second. Adopting a frame hands
    // its ids to the layout at once but its coordinates only come back from
    // the worker a moment later, so that draw painted the new frame's edges
    // against the previous frame's positions: every edge joining whichever
    // agents happened to hold those slots. Measured at 4,879 nodes drawn
    // against 4,777 nodes' worth of coordinates, once per iteration.
    this.accumulator += dt;
    if (this.accumulator >= 1 / this.SPEED) {
      this.accumulator = 0;
      this.advance();
    }

    // Scenery does not need sixty frames a second.
    this._sinceDraw = (this._sinceDraw || 0) + dt;
    if (this._sinceDraw < 1 / this.fps) return;
    const step = this._sinceDraw;
    this._sinceDraw = 0;

    this.layout.tick();

    // Measure again if we have never had a size. resize() can run before the
    // canvas has been laid out — during start-up, or while the view is still
    // hidden — and the renderer then holds a width of zero, which the check
    // below turns into "never draw anything". Nothing else would ever ask it
    // to measure a second time, so the backdrop stayed blank for good.
    if (!(this.renderer.cssWidth > 0)) this.resize();

    // The camera does not depend on which frame is showing, so it keeps
    // gliding through the wait rather than stalling for it.
    this.renderer.rotate(this.ROTATION * step, 0);

    // Until the coordinates belong to the frame we are holding, the canvas
    // simply keeps what it already shows.
    if (!this.readyToDraw || !(this.renderer.cssWidth > 0)) {
      this.renderer.stepCamera();
      return;
    }

    // Framed the way the Viewer's Fit View does it: the tight framing is
    // recomputed every frame and the camera is a spring chasing it, so the
    // view eases as the graph grows and shrinks. Refitting on a timer instead
    // meant the camera sat still and then lurched.
    if (this._needsFraming) {
      this.renderer.fitToContent(this.layout, this.ZOOM, true);
      this._needsFraming = false;
    } else {
      this.renderer.fitToContent(this.layout, this.ZOOM);
    }
    this.renderer.stepCamera();

    const began = performance.now();
    this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
    this.measure(performance.now() - began);
  },

  /**
   * Keep the frame rate to something this browser can actually deliver.
   *
   * A rolling average rather than the last draw, because one slow frame is
   * usually the machine doing something else, and reacting to it would make
   * the rate flap. The thresholds leave a gap between them for the same
   * reason: it drops at twenty milliseconds and only climbs back below eight,
   * so it settles instead of oscillating around one number.
   */
  measure(spent) {
    this._cost = this._cost === undefined ? spent : this._cost * 0.9 + spent * 0.1;
    if (this._cost > 20 && this.fps > this.MIN_FPS) {
      this.fps = this.MIN_FPS;
    } else if (this._cost < 8 && this.fps < this.MAX_FPS) {
      this.fps = this.MAX_FPS;
    }
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
