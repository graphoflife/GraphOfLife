/*
 * Two ways of changing what is shown without changing the run.
 *
 * Focus crops the graph to one node's neighbourhood. Fullscreen gives the
 * whole thing the screen. They belong together because both are about framing
 * rather than about the simulation, and both get switched on and off while
 * looking at something else.
 */

Object.assign(Viewer, {

  // ------------------------------------------------------------------
  // Focus: one node's neighbourhood, a few steps out
  // ------------------------------------------------------------------

  /**
   * Crop a frame to what lies within `focusRadius` steps of the focused node.
   *
   * Everything on screen then describes the neighbourhood rather than the
   * world: the layout spreads only these nodes, and the statistics, charts and
   * hover card are all computed from the cropped frame. That is the point — a
   * neighbourhood you can actually read — but it does mean a node at the edge
   * of the ball shows the degree it has *here*, not the degree it has in the
   * whole graph. What you see is what is measured.
   *
   * The focused node does not always survive the phase. It is followed to a
   * neighbour when it dies, and the whole graph comes back only when the
   * neighbourhood went with it.
   */
  focusFrame(full) {
    if (!full || this.focusId === null) return full;

    const adjacency = new Map();
    for (const id of full.ids) adjacency.set(id, []);
    for (const [a, b] of full.edges) {
      if (a === b) continue;
      const la = adjacency.get(a), lb = adjacency.get(b);
      if (la) la.push(b);
      if (lb) lb.push(a);
    }

    let anchor = this.focusId;
    if (!adjacency.has(anchor)) {
      // Gone this phase. Its neighbours from the last frame it was in are the
      // only handle left on where it was, so the densest survivor inherits the
      // focus — that keeps the view in the thick of the same neighbourhood
      // rather than on whichever id happened to be listed first.
      const gone = anchor;
      let best = null, bestDegree = -1;
      for (const id of this.focusNeighbours) {
        const list = adjacency.get(id);
        if (!list) continue;
        if (list.length > bestDegree) { best = id; bestDegree = list.length; }
      }
      if (best === null) {
        this.setFocus(null, `node ${gone} and everything around it is gone \u2014 showing the whole graph`);
        return full;
      }
      anchor = best;
      this.focusId = best;
      this.focusNote = `node ${gone} died \u2014 following neighbour ${best}`;
    }

    this.focusNeighbours = adjacency.get(anchor).slice();

    // The ball, breadth first.
    const inBall = new Set([anchor]);
    let layer = [anchor];
    for (let step = 0; step < this.settings.focusRadius && layer.length; step++) {
      const next = [];
      for (const u of layer) {
        for (const v of adjacency.get(u)) {
          if (inBall.has(v)) continue;
          inBall.add(v);
          next.push(v);
        }
      }
      layer = next;
    }

    return this.cropFrame(full, inBall, { focusAnchor: anchor });
  },

  /**
   * A frame reduced to one set of agents, and nothing else changed.
   *
   * Shared by every view that shows part of the graph, because the hard part
   * is not choosing the agents — it is carrying across everything that is
   * index-aligned with `ids` without quietly dropping a field. Written twice,
   * the second copy is where a field goes missing.
   */
  cropFrame(full, keepSet, extra = {}) {
    const keep = [];
    for (let i = 0; i < full.ids.length; i++) if (keepSet.has(full.ids[i])) keep.push(i);
    const take = arr => (arr ? keep.map(i => arr[i]) : undefined);

    const sub = {
      iteration: full.iteration,
      phase: full.phase,
      nodes_before: full.nodes_before,
      ids: keep.map(i => full.ids[i]),
      edges: full.edges.filter(([a, b]) => keepSet.has(a) && keepSet.has(b)),
      cleanup: full.cleanup,
      previous: full.previous,
      ...extra
    };
    // The per-node arrays, listed in one place rather than spelled out one
    // assignment at a time. They are index-aligned with `ids` and so all crop
    // the same way; the engine's _frame is what decides the list, and nothing
    // links the two, so a field added there has to be added here as well. Kept
    // together, that is one obvious line — spread through the object above, it
    // was a field that silently read as absent in focus mode and nowhere else.
    // Ones an older recording lacks are simply not copied.
    for (const name of ['tokens', 'brain_ids', 'parent_brain_ids', 'parent_ids',
                        'ages', 'delta']) {
      if (full[name]) sub[name] = take(full[name]);
    }

    // Decisions are filtered to the agents on screen, so the reproduction and
    // game statistics describe this part of the graph too rather than the world.
    const d = full.decisions;
    if (d) {
      sub.decisions = {};
      if (d.births) sub.decisions.births = d.births.filter(b => keepSet.has(b.agent));
      if (d.allocations) sub.decisions.allocations = d.allocations.filter(a => keepSet.has(a.agent));
      if (d.winners) sub.decisions.winners = d.winners.filter(w => keepSet.has(w.node));
      if (d.gifts) sub.decisions.gifts = d.gifts.filter(g => keepSet.has(g[0]));
      if (d.pruned_edges) sub.decisions.pruned_edges = d.pruned_edges;
    }

    // The frame-level counts are whole-graph numbers; dropping them makes
    // the stat fall back to counting the records that survived the crop.
    sub.summary = {
      nodes: sub.ids.length,
      edges: sub.edges.length,
      tokens: (sub.tokens || []).reduce((a, b) => a + b, 0)
    };
    return sub;
  },

  // ------------------------------------------------------------------
  // k-core: the part of the graph where everyone has k neighbours
  // ------------------------------------------------------------------

  /**
   * Crop a frame to its k-core.
   *
   * The k-core is what survives repeatedly deleting every agent with fewer
   * than k connections, until nobody has fewer. Peeling is iterative because
   * removing an agent lowers its neighbours' degrees, which can put them below
   * the line in turn — one pass would leave most of the fringe standing.
   *
   * At k = 1 this drops the isolated; at k = 2 the whole tree-shaped fringe
   * goes and what is left is the part of the graph where every agent sits on
   * some loop; higher k asks for progressively denser cores.
   *
   * Everything under the canvas is then measured on the core alone. Some of
   * those numbers stop being what they were — the core can be several
   * components where the graph was one, and the fringe an agent lost is real
   * structure it no longer appears to have. That is the same bargain focus
   * makes: what you see is what is measured.
   */
  coreFrame(full, k) {
    if (!full || !(k >= 1)) return full;

    const degree = new Map();
    const adjacency = new Map();
    for (const id of full.ids) { adjacency.set(id, []); degree.set(id, 0); }
    for (const [a, b] of full.edges) {
      if (a === b) continue;
      const la = adjacency.get(a), lb = adjacency.get(b);
      if (la && lb) {
        la.push(b); lb.push(a);
        degree.set(a, degree.get(a) + 1);
        degree.set(b, degree.get(b) + 1);
      }
    }

    // Peel, carrying a queue rather than sweeping repeatedly: an agent goes on
    // it the moment its degree drops under k, so each is considered once.
    const gone = new Set();
    const queue = full.ids.filter(id => degree.get(id) < k);
    for (const id of queue) gone.add(id);
    for (let head = 0; head < queue.length; head++) {
      for (const v of adjacency.get(queue[head])) {
        if (gone.has(v)) continue;
        degree.set(v, degree.get(v) - 1);
        if (degree.get(v) < k) { gone.add(v); queue.push(v); }
      }
    }

    const core = new Set(full.ids.filter(id => !gone.has(id)));
    return this.cropFrame(full, core, { coreK: k });
  },

  /**
   * What the Viewer should actually draw: the whole frame, one neighbourhood,
   * or one core.
   *
   * Focus wins when it is set. The two answer different questions — focus is
   * "what is around this agent", the core is "what does the graph look like
   * with its fringe taken off" — and cropping to a ball and then peeling it
   * would report a core of a neighbourhood, which is neither. The control is
   * disabled while a node is focused for the same reason.
   */
  viewFrame(full) {
    if (!full) return full;
    if (this.focusId !== null) return this.focusFrame(full);
    if (this.settings.kCoreOn) return this.coreFrame(full, this.settings.kCore);
    return full;
  },

  /** Point the view at a node, or at nothing. */
  setFocus(id, note = '') {
    this.focusId = id;
    this.focusNote = note;
    if (id === null) this.focusNeighbours = [];
    this.refocus(true);
  },

  /** Rebuild what is on screen from the frame that was read. */
  refocus(reframe = false) {
    if (!this.fullFrame) { this.updateFocusUi(); return; }

    this.frame = this.viewFrame(this.fullFrame);
    // Handed over together, so nothing is ever drawn against the other one's
    // node ordering.
    this.layout.setFrame(this.frame.ids, this.frame.edges, this.frame.parent_ids,
                         this.settings.layoutCarry);
    this.layout.reheat(reframe ? 1 : 0.5);

    this.rebuildMetrics();
    this.updateStats();
    this.updateCharts();
    this.updateFocusUi();
    if (reframe) this.setAutoFit(true);
  },

  updateFocusUi() {
    const note = document.getElementById('focusNote');
    const clear = document.getElementById('btnClearFocus');
    const focused = this.focusId !== null;

    if (clear) clear.disabled = !focused;

    // Focus and the core answer different questions and cannot be read
    // together, so while a node is focused the core control is unavailable and
    // says so by being disabled rather than by quietly doing nothing.
    const coreOn = document.getElementById('kCoreOn');
    const coreK = document.getElementById('kCoreK');
    const coreField = document.getElementById('kCoreField');
    if (coreOn && coreK) {
      coreOn.disabled = focused;
      coreK.disabled = focused || !coreOn.checked;
      if (coreField) coreField.classList.toggle('is-disabled', focused);
      if (focused && this.settings.kCoreOn) this.settings.kCoreOn = false;
      coreOn.checked = this.settings.kCoreOn;
    }

    if (!note) return;

    if (!focused) {
      if (this.settings.kCoreOn && this.frame && this.fullFrame) {
        const shown = this.frame.ids.length;
        const whole = this.fullFrame.ids.length;
        note.textContent =
          `${this.settings.kCore}-core: ${formatNumber(shown)} of `
          + `${formatNumber(whole)} nodes — every statistic below is measured `
          + `on the core alone`
          + (shown === 0 ? ', and nothing has that many connections' : '');
        return;
      }
      note.textContent = this.focusNote || '';
      return;
    }
    const shown = this.frame ? this.frame.ids.length : 0;
    const whole = this.fullFrame ? this.fullFrame.ids.length : 0;
    note.textContent =
      `Focus: node ${this.focusId}, ${this.settings.focusRadius} step`
      + `${this.settings.focusRadius === 1 ? '' : 's'} out \u2014 `
      + `${formatNumber(shown)} of ${formatNumber(whole)} nodes`
      + (this.focusNote ? ` \u00b7 ${this.focusNote}` : '');
  },

  // ------------------------------------------------------------------
  // Fullscreen
  // ------------------------------------------------------------------

  /** The element currently filling the screen, whatever the browser calls it. */
  get fullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  },

  /**
   * Fill the screen with the viewer, or give it back.
   *
   * The whole two-pane layout goes fullscreen rather than the canvas alone, so
   * the settings and the playbar come with it — a graph you cannot recolour or
   * step through is a screenshot, not a view.
   */
  toggleFullscreen() {
    const target = document.getElementById('viewerLayout');
    if (!target) return;

    if (this.fullscreenElement) {
      const exit = document.exitFullscreen || document.webkitExitFullscreen;
      if (exit) exit.call(document);
      return;
    }

    const request = target.requestFullscreen || target.webkitRequestFullscreen;
    if (!request) {
      this.emptyEl.textContent = 'This browser will not allow fullscreen here.';
      return;
    }
    // Refused when not driven by a real click, and on some embedded views at
    // any time; there is nothing to recover, so just leave the view as it was.
    Promise.resolve(request.call(target)).catch(() => {});
  },

  /** Follow the browser's idea of fullscreen, however it was changed. */
  syncFullscreen() {
    const target = document.getElementById('viewerLayout');
    const on = this.fullscreenElement === target;

    target.classList.toggle('is-fullscreen', on);
    const button = document.getElementById('btnFullscreen');
    if (button) {
      button.setAttribute('aria-pressed', String(on));
      button.textContent = on ? 'Exit fullscreen' : 'Fullscreen';
    }

    // The canvas has just changed size by a lot. The ResizeObserver catches
    // this on its own, but not before the next frame is drawn, and refitting
    // here keeps the graph from being briefly framed for the old box.
    this.resize();
    if (this.settings.autoFit) this.renderer.fitToContent(this.layout, undefined, true);
  }

});
