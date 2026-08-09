/*
 * The force layout, running off the main thread.
 *
 * This is not a second implementation: it loads the same force.js and drives
 * the same ForceLayout. Only the loop around it lives here. Everything the
 * layout does is arithmetic over its own state — it never touches the canvas
 * or the document — which is what makes it movable at all.
 *
 * The point is to stop a slow tick from freezing the interface. On a large
 * graph a tick costs tens of milliseconds, and on the main thread nothing else
 * can happen while it runs: no repaint, no response to a drag. Here it runs on
 * its own core and the page redraws from whatever positions have most recently
 * arrived.
 *
 * Positions go back either through a SharedArrayBuffer, which both sides read
 * without copying, or as an ordinary copied message when the page is not
 * cross-origin isolated and shared memory is unavailable.
 */
importScripts('force.js');

const layout = new ForceLayout();

let running = false;
let scheduled = false;
let shared = null;        // Float32Array over a SharedArrayBuffer, when available
let ticksDone = 0;

// How long to spend ticking before handing positions back. A little under a
// frame, so the page never waits long for fresh coordinates, and never more
// than a few ticks in case each one is slow.
const BUDGET_MS = 12;
const MAX_TICKS_PER_BATCH = 8;

/**
 * Copy the current positions out, in the same order as `ids`.
 *
 * With shared memory this writes straight into the buffer the page is reading;
 * otherwise it returns a fresh array for the message to carry.
 */
function publish() {
  const n = layout.ids.length;

  if (shared && shared.length >= n * 3) {
    const ids = layout.ids;
    for (let i = 0; i < n; i++) {
      const p = layout.pos.get(ids[i]);
      const o = i * 3;
      if (p) { shared[o] = p.x; shared[o + 1] = p.y; shared[o + 2] = p.z; }
    }
    return null;
  }
  return layout.syncPositions().slice(0, n * 3);
}

function report(positions) {
  const message = { type: 'positions', alpha: layout.alpha, count: layout.ids.length, ticks: ticksDone };
  if (positions) message.positions = positions;
  postMessage(message);
}

function loop() {
  scheduled = false;
  if (!running) return;

  const start = performance.now();
  let ticks = 0;

  // Run as many ticks as fit in the budget. A settled layout still yields, so
  // the worker stays responsive to messages rather than spinning.
  do {
    if (!layout.tick()) break;
    ticks++;
    ticksDone++;
  } while (ticks < MAX_TICKS_PER_BATCH && performance.now() - start < BUDGET_MS);

  report(publish());

  // setTimeout rather than a tight loop: it returns to the event loop, so
  // messages queued by the page are handled between batches.
  scheduled = true;
  setTimeout(loop, 0);
}

function start() {
  if (running) return;
  running = true;
  if (!scheduled) { scheduled = true; setTimeout(loop, 0); }
}

self.onmessage = (e) => {
  const msg = e.data;

  switch (msg.type) {
    case 'init':
      if (msg.buffer) shared = new Float32Array(msg.buffer);
      postMessage({ type: 'ready', shared: Boolean(shared) });
      break;

    case 'buffer':
      // The page grew the shared buffer because the graph did.
      shared = msg.buffer ? new Float32Array(msg.buffer) : null;
      break;

    case 'frame':
      layout.setFrame(msg.ids, msg.edges, msg.parents, msg.carry);
      // Answer immediately: the page has a new frame to draw and should not
      // wait a whole batch for coordinates.
      report(publish());
      start();
      break;

    case 'params':
      for (const [key, value] of Object.entries(msg.params)) {
        if (typeof value === 'number') layout[key] = value;
      }
      break;

    case 'dimensions':
      layout.setDimensions(msg.dimensions);
      report(publish());
      start();
      break;

    case 'reheat':
      layout.reheat(msg.alpha);
      start();
      break;

    case 'scatter':
      layout.scatter();
      report(publish());
      start();
      break;

    case 'stop':
      running = false;
      break;

    default:
      break;
  }
};
