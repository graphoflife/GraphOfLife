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

    const keep = [];
    for (let i = 0; i < full.ids.length; i++) if (inBall.has(full.ids[i])) keep.push(i);
    const take = arr => (arr ? keep.map(i => arr[i]) : undefined);

    const sub = {
      iteration: full.iteration,
      phase: full.phase,
      nodes_before: full.nodes_before,
      ids: keep.map(i => full.ids[i]),
      tokens: take(full.tokens),
      brain_ids: take(full.brain_ids),
      parent_brain_ids: take(full.parent_brain_ids),
      parent_ids: take(full.parent_ids),
      edges: full.edges.filter(([a, b]) => inBall.has(a) && inBall.has(b)),
      cleanup: full.cleanup,
      previous: full.previous,
      focusAnchor: anchor
    };
    if (full.delta) sub.delta = take(full.delta);

    // Decisions are filtered to the agents on screen, so the reproduction and
    // game statistics describe this neighbourhood too rather than the world.
    const d = full.decisions;
    if (d) {
      sub.decisions = {};
      if (d.births) sub.decisions.births = d.births.filter(b => inBall.has(b.agent));
      if (d.rewires) sub.decisions.rewires = d.rewires.filter(r => inBall.has(r.agent));
      if (d.allocations) sub.decisions.allocations = d.allocations.filter(a => inBall.has(a.agent));
      if (d.winners) sub.decisions.winners = d.winners.filter(w => inBall.has(w.node));
      if (d.pruned_edges) sub.decisions.pruned_edges = d.pruned_edges;
    }

    // The frame-level rewire count is a whole-graph number; dropping it makes
    // the stat fall back to counting the records that survived the crop.
    sub.summary = {
      nodes: sub.ids.length,
      edges: sub.edges.length,
      tokens: sub.tokens.reduce((a, b) => a + b, 0)
    };
    return sub;
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

    this.frame = this.focusFrame(this.fullFrame);
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
    if (!note) return;

    if (!focused) { note.textContent = this.focusNote || ''; return; }
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
