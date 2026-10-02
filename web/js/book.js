/*
 * The book: a research programme, written down as it is carried out.
 *
 * Its source is book/ — book.json for the order of the chapters, one Markdown
 * file per chapter, a plan per experiment, and the figures and results an
 * analysis writes — read here and on GitHub alike. This file only lays it out:
 * a list of chapters, the chapter, and three things a chapter can ask to have
 * drawn in its text with a fenced block:
 *
 *     ```thesis E02         the experiment's claim, as its plan states it
 *     ```experiment E02     what it runs, and on a machine with gol_server.py,
 *                           how far it has got, with ▶ and ⏸
 *     ```figure E02/nodes   a chart from book/figures/E02/nodes.json
 *
 * The claim is drawn from the plan rather than written into the chapter, so
 * it cannot quietly be reworded after the results are in.
 *
 * Experiments are run by the lab (gol_lab.py) through the server. On the
 * published site there is no server and no lab: the book is the same, minus
 * the buttons and the progress, and never asks.
 */
const Book = {
  // The build stamps this with its commit, and everything else the book
  // fetches carries the same stamp, so a published chapter is never read from
  // a stale cache next to a fresh one.
  INDEX: 'book/book.json',
  EMBEDS: ['thesis', 'experiment', 'figure'],
  POLL_MS: 5000,
  STORE: 'gol.book.chapter',

  index: null,
  plans: new Map(),
  figures: [],
  lab: null,
  // What the lab last refused, said in the panel rather than in a dialog.
  refusal: null,
  current: null,
  active: false,
  timer: null,

  init() {
    this.toc = document.getElementById('bookToc');
    this.page = document.getElementById('bookPage');
    this.toc.addEventListener('click', event => {
      const link = event.target.closest('[data-chapter]');
      if (!link) return;
      event.preventDefault();
      this.open(link.dataset.chapter);
    });
    this.page.addEventListener('click', event => this.onClick(event));
    this.page.addEventListener('change', event => this.onChange(event));
    window.addEventListener('resize', () => { if (this.active) this.drawFigures(); });
  },

  async setActive(active) {
    this.active = active;
    clearInterval(this.timer);
    this.timer = null;
    if (!active) return;
    await this.load();
    await this.refreshLab();
    // Only while the book is on screen, and only where there is a lab.
    if (this.lab) this.timer = setInterval(() => this.refreshLab(), this.POLL_MS);
  },

  resume() {
    if (this.active) this.refreshLab();
  },

  // ---- reading the book -------------------------------------------------

  url(path) {
    const at = this.INDEX.indexOf('?');
    return `book/${path}${at < 0 ? '' : this.INDEX.slice(at)}`;
  },

  async text(path) {
    const response = await fetch(this.url(path));
    if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
    return response.text();
  },

  async json(path) {
    return JSON.parse(await this.text(path));
  },

  async plan(name) {
    if (!this.plans.has(name)) this.plans.set(name, this.json(`experiments/${name}.json`));
    return this.plans.get(name);
  },

  chapters() {
    return this.index ? this.index.parts.flatMap(part => part.chapters) : [];
  },

  chapter(id) {
    return this.chapters().find(c => c.id === id);
  },

  async load() {
    if (this.index) return;
    try {
      this.index = await this.json('book.json');
    } catch (err) {
      this.page.innerHTML = `<p class="md-status">Could not read the book: ${
        Markdown.escape(String(err.message))}</p>`;
      return;
    }
    let remembered = null;
    try { remembered = localStorage.getItem(this.STORE); } catch (err) { /* private window */ }
    const first = this.chapters().find(c => c.file);
    await this.open(this.chapter(remembered)?.file ? remembered : first.id);
  },

  async open(id) {
    const chapter = this.chapter(id);
    if (!chapter) return;
    this.current = id;
    try { localStorage.setItem(this.STORE, id); } catch (err) { /* private window */ }
    this.renderToc();

    if (!chapter.file) {
      this.page.innerHTML = `${this.kicker(chapter)}<h2>${Markdown.inline(chapter.title)}</h2>`
        + `<p class="md-status">Not written yet. ${Markdown.inline(chapter.note || '')}</p>`;
      return;
    }
    let source;
    try {
      source = await this.text(chapter.file);
    } catch (err) {
      this.page.innerHTML = `<p class="md-status">Could not read this chapter: ${
        Markdown.escape(String(err.message))}</p>`;
      return;
    }
    if (this.current !== id) return;           // another chapter was opened meanwhile
    this.page.innerHTML = this.kicker(chapter) + Markdown.render(source, this.EMBEDS)
      + this.turns(chapter);
    window.scrollTo(0, 0);
    this.figures = [];
    await Promise.all([...this.page.querySelectorAll('.md-embed')].map(el => this.embed(el)));
    this.drawFigures();
  },

  /** Which chapter this is, above its title: a number, an appendix letter, or nothing. */
  kicker(chapter) {
    const what = chapter.experiment ? ` · Experiment ${Number(chapter.experiment.slice(1))}` : '';
    const number = /^\d+$/.test(chapter.id) ? `Chapter ${chapter.id}`
      : /^[A-Z]$/.test(chapter.id) ? `Appendix ${chapter.id}` : '';
    return number ? `<p class="book-kicker">${number}${what}</p>` : '';
  },

  /** Links to the chapters before and after, at the foot of one. */
  turns(chapter) {
    const written = this.chapters().filter(c => c.file);
    const at = written.indexOf(chapter);
    const link = (c, rel) => c ? `<a href="#" data-chapter="${c.id}" class="book-turn-${rel}">${
      rel === 'prev' ? '←' : ''} ${Markdown.inline(c.title)} ${rel === 'next' ? '→' : ''}</a>` : '<span></span>';
    return `<nav class="book-turns">${link(written[at - 1], 'prev')}${link(written[at + 1], 'next')}</nav>`;
  },

  // ---- the chapter list -------------------------------------------------

  renderToc() {
    if (!this.index) return;
    const parts = this.index.parts.map(part => {
      const items = part.chapters.map(c => {
        const state = this.describe(c);
        return `<a href="#" data-chapter="${c.id}" class="book-toc-item${
          c.id === this.current ? ' active' : ''}${c.file ? '' : ' unwritten'}">`
          + `<span class="book-toc-num">${/^(\d+|[A-Z])$/.test(c.id) ? c.id : ''}</span>`
          + `<span class="book-toc-title">${Markdown.inline(c.title)}</span>`
          + (state.label ? `<span class="book-chip book-chip-${state.state}">${state.label}</span>` : '')
          + '</a>';
      }).join('');
      return `<div class="book-part"><h3>${Markdown.inline(part.title)}</h3>${items}</div>`;
    }).join('');
    this.toc.innerHTML = `<div class="book-title">${Markdown.inline(this.index.title)}</div>${parts}`;
  },

  /**
   * What a chapter's chip says. An analysed chapter is analysed whatever the
   * lab says; otherwise an experiment is where the lab says it is, and a
   * chapter without one is just written or not.
   */
  describe(chapter, lab = this.lab) {
    if (chapter.status === 'analysed') return { state: 'analysed', label: 'analysed' };
    const live = chapter.experiment && lab && lab.experiments[chapter.experiment];
    if (live) {
      const share = live.total ? Math.floor(100 * live.done / live.total) : 0;
      const labels = {
        ready: 'ready to run', queued: 'queued', running: `running · ${share}%`,
        paused: `paused · ${share}%`, stopped: `not running · ${share}%`,
        finished: 'finished · to analyse', blocked: 'blocked', invalid: 'plan invalid'
      };
      return { state: live.state, label: labels[live.state] || live.state };
    }
    // Dimmed in the list already; a chip on every one of them would be noise.
    if (!chapter.file) return { state: 'planned', label: '' };
    if (chapter.status === 'ready') return { state: 'ready', label: 'ready to run' };
    if (chapter.status === 'draft') return { state: 'draft', label: 'draft' };
    return { state: 'written', label: '' };
  },

  // ---- the lab ----------------------------------------------------------

  async refreshLab() {
    await API.choose();
    if (API.runsInBrowser) {
      this.lab = null;
      return;
    }
    try {
      this.lab = await API.labStatus();
    } catch (err) {
      this.lab = null;
    }
    this.renderToc();
    for (const panel of this.page.querySelectorAll('.book-experiment')) this.updatePanel(panel);
  },

  async act(body) {
    try {
      this.lab = await API.labRequest(body);
      this.refusal = null;
    } catch (err) {
      this.refusal = err.message;
    }
    this.renderToc();
    for (const panel of this.page.querySelectorAll('.book-experiment')) this.updatePanel(panel);
    if (this.lab && !this.timer && this.active) {
      this.timer = setInterval(() => this.refreshLab(), this.POLL_MS);
    }
  },

  onClick(event) {
    const chapter = event.target.closest('[data-chapter]');
    if (chapter) {
      event.preventDefault();
      this.open(chapter.dataset.chapter);
      return;
    }
    const action = event.target.closest('[data-lab]');
    if (action) {
      const [what, name] = action.dataset.lab.split(':');
      this.act(what === 'run' ? { run: name } : { pause: true });
      return;
    }
    const run = event.target.closest('[data-run]');
    if (run && run.dataset.started === 'yes') {
      App.showView('viewer');
      Viewer.load(run.dataset.run);
    }
  },

  onChange(event) {
    if (event.target.matches('[data-workers]')) this.act({ workers: Number(event.target.value) });
  },

  // ---- what a chapter asks to have drawn ----------------------------------

  async embed(el) {
    const { kind, arg } = el.dataset;
    try {
      if (kind === 'thesis') await this.thesis(el, arg);
      else if (kind === 'experiment') await this.experiment(el, arg);
      else if (kind === 'figure') await this.figure(el, arg);
    } catch (err) {
      el.innerHTML = `<p class="md-status">Could not draw ${Markdown.escape(kind)} ${
        Markdown.escape(arg)}: ${Markdown.escape(String(err.message))}</p>`;
    }
  },

  async thesis(el, name) {
    const { thesis = {} } = await this.plan(name);
    const parts = [['claim', 'The claim'], ['why', 'Why it would be so'],
                   ['confirm', 'It holds if'], ['refute', 'It fails if']];
    el.className = 'book-thesis';
    el.innerHTML = parts.filter(([key]) => thesis[key])
      .map(([key, label]) => `<p><b>${label}.</b> ${Markdown.inline(thesis[key])}</p>`).join('')
      + '<p class="book-note">Written down before the runs, in the experiment\'s plan.</p>';
  },

  async experiment(el, name) {
    const plan = await this.plan(name);
    el.className = 'book-experiment';
    el.dataset.experiment = name;
    el.innerHTML = `<div class="book-live"></div>${this.planHtml(plan)}`;
    this.updatePanel(el);
  },

  /** What an experiment runs, from its plan: the same on every host. */
  planHtml(plan) {
    const runs = plan.runs;
    if (typeof runs === 'string') {
      const other = runs.replace('same as ', '');
      const chapter = this.chapters().find(c => c.experiment === other);
      return `<p>Uses the runs of ${chapter ? `<a href="#" data-chapter="${chapter.id}">Experiment ${
        Number(other.slice(1))}</a>` : other}: nothing new is run for it.</p>`;
    }
    const seeds = String(runs.seeds);
    const rows = runs.conditions.map(c => {
      const set = Object.entries(c.set || {}).map(([k, v]) => `<code>${k} = ${
        Markdown.escape(JSON.stringify(v))}</code>`).join(', ') || 'as the baseline';
      const how = [c.stops && c.stops.length ? `stopped at ${c.stops.join(', ')} and continued in a new process` : '',
                   c.fault_at !== undefined ? `cut off at ${c.fault_at} with no checkpoint, then resumed` : '',
                   c.threads === 'default' ? 'matrix library on all its threads' : '']
        .filter(Boolean).join('; ');
      return `<tr><td>${Markdown.escape(c.name)}</td><td>${set}</td><td>${how || '—'}</td></tr>`;
    }).join('');
    const record = runs.record || {};
    return `<div class="md-scroll"><table class="book-plan">`
      + `<tr><th>Baseline</th><td colspan="2"><code>${Markdown.escape(runs.baseline)}</code></td></tr>`
      + `<tr><th>World</th><td colspan="2">${formatNumber(runs.world.total_tokens)} tokens</td></tr>`
      + `<tr><th>Seeds</th><td colspan="2">${Markdown.escape(seeds)}</td></tr>`
      + `<tr><th>Iterations</th><td colspan="2">${formatNumber(runs.iterations)}</td></tr>`
      + `<tr><th>Recorded</th><td colspan="2">every frame with its decisions; every statistic `
      + `every iteration, the graph ones every ${record.heavy_every || 25}</td></tr>`
      + `<tr><th>Condition</th><th>Differs from the baseline</th><th>How it is run</th></tr>`
      + rows + '</table></div>';
  },

  updatePanel(el) {
    const name = el.dataset.experiment;
    const live = el.querySelector('.book-live');
    const status = this.lab && this.lab.experiments[name];
    if (!this.lab || !status) {
      live.innerHTML = this.lab ? '' : '<p class="book-note">Experiments run on a machine with '
        + '<code>gol_server.py</code>: open this page there to run one, see how far it has got, '
        + 'or pause it.</p>';
      return;
    }
    live.innerHTML = this.liveHtml(name, status, this.lab);
  },

  /** Progress, the time left, the buttons and every run's bar: only where there is a lab. */
  liveHtml(name, status, lab) {
    const chapter = this.chapters().find(c => c.experiment === name) || {};
    const state = this.describe(chapter, lab);
    const share = status.total ? status.done / status.total : 0;
    const button = ['ready', 'stopped', 'paused'].includes(status.state)
      ? `<button class="primary small" data-lab="run:${name}">▶ ${status.done ? 'Continue' : 'Run'}</button>`
      : ['running', 'queued'].includes(status.state)
        ? `<button class="ghost small" data-lab="pause">⏸ Pause the lab</button>` : '';
    const left = ['running', 'queued', 'paused', 'stopped'].includes(status.state)
      ? `<span class="book-left">${this.eta(status.secondsLeft)}</span>` : '';
    const workers = Array.from({ length: lab.lab.cores || 8 }, (_, i) => i + 1)
      .map(n => `<option value="${n}"${n === lab.lab.workers ? ' selected' : ''}>${n}</option>`).join('');
    const estimate = status.state === 'finished' ? '' : `<p class="book-note">Estimated: ${
      this.eta(status.estimate.seconds, '')} on ${lab.lab.workers} workers from the start, `
      + `${formatBytes(status.estimate.diskBytes)} of disk (${formatBytes(lab.disk.free)} free), `
      + `up to ${formatNumber(Math.round(status.estimate.peakMB))} MB of memory a run. `
      + `Fitted to ${lab.costs.runs || 'no'} recorded runs so far.</p>`;
    const reason = [this.refusal, lab.lab.reason && status.state === 'paused' ? lab.lab.reason : null]
      .filter(Boolean).map(text => `<p class="book-reason">${Markdown.escape(text)}</p>`).join('');
    const blocked = status.runs.filter(r => r.reason)
      .map(r => `<p class="book-reason">${Markdown.escape(r.id)}: ${Markdown.escape(r.reason)}</p>`).join('');
    const engines = Object.keys(status.engines || {}).length > 1
      ? `<p class="book-note">Its runs were made by ${Object.keys(status.engines).length} versions `
        + 'of the engine; the analysis checks that the newest remakes the others exactly.</p>' : '';
    const shared = [...new Set(status.runs.flatMap(r => r.sharedWith))].sort();
    const runs = status.runs.map(r => {
      const done = r.until ? Math.min(1, r.iteration / r.until) : 0;
      return `<span class="book-run book-run-${r.state}" data-run="${r.id}" data-started="${
        r.iteration > 0 ? 'yes' : 'no'}" title="${Markdown.escape(r.name)} — ${
        formatNumber(r.iteration)} of ${formatNumber(r.until)}${r.iteration > 0 ? ' · click to open' : ''}">`
        + `<span style="width:${(100 * done).toFixed(1)}%"></span></span>`;
    }).join('');
    return `<div class="book-exp-head"><span class="book-chip book-chip-${state.state}">${state.label}</span>`
      + `<span class="book-progress"><span style="width:${(100 * share).toFixed(1)}%"></span></span>`
      + `${left}${button}<label class="book-workers">Workers <select data-workers>${workers}</select></label></div>`
      + reason + blocked + estimate + engines
      + `<div class="book-runs">${runs}</div>`
      + (shared.length ? `<p class="book-note">Its runs are shared with ${shared.join(', ')}.</p>` : '');
  },

  /** A duration as a person would say it. */
  eta(seconds, suffix = ' left') {
    if (!Number.isFinite(seconds) || seconds <= 0) return suffix ? 'almost done' : 'no time';
    if (seconds < 90) return `under 2 min${suffix}`;
    if (seconds < 5400) return `about ${Math.round(seconds / 60)} min${suffix}`;
    if (seconds < 172800) return `about ${(seconds / 3600).toFixed(1)} h${suffix}`;
    return `about ${(seconds / 86400).toFixed(1)} days${suffix}`;
  },

  // ---- figures ----------------------------------------------------------

  async figure(el, arg) {
    const data = await this.json(`figures/${arg}.json`);
    el.className = 'book-figure';
    el.innerHTML = '<canvas></canvas>'
      + (data.caption ? `<p class="book-caption">${Markdown.inline(data.caption)}</p>` : '');
    this.figures.push({ canvas: el.querySelector('canvas'), data });
  },

  drawFigures() {
    for (const { canvas, data } of this.figures) this.draw(canvas, data);
  },

  /**
   * Lines over a shared axis, each with the band of the runs behind it: the
   * middle half of them, and a fainter band for nine in ten. Drawn with the
   * chart helpers every other chart on the page uses.
   */
  draw(canvas, fig) {
    const series = fig.series || [];
    const chrome = { title: fig.title, xLabel: fig.x.label, yLabel: fig.y.label, ticks: true,
                     legend: series.map((s, i) => ({ label: s.label, colour: Ink.line(i) })) };
    const pad = _chromePad(chrome);
    const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);
    const scale = axis => (axis.log ? v => (v > 0 ? Math.log10(v) : NaN) : v => v);
    const sx = scale(fig.x), sy = scale(fig.y);

    const xs = [], ys = [];
    for (const s of series) {
      xs.push(...s.x.map(sx));
      for (const key of ['y', 'lo', 'hi', 'outerLo', 'outerHi']) {
        if (s[key]) ys.push(...s[key].filter(v => v !== null).map(sy));
      }
    }
    const finite = values => values.filter(Number.isFinite);
    const span = (values, lo, hi) => {
      const f = finite(values);
      return f.length ? [lo ?? Math.min(...f), hi ?? Math.max(...f)] : [0, 1];
    };
    const [x0, x1] = span(xs);
    const [y0, y1] = span(ys, fig.y.min !== undefined ? sy(fig.y.min) : undefined);
    const back = axis => (axis.log ? v => _short(10 ** v) : v => _short(v));
    const x = { lo: x0, hi: x1, format: back(fig.x) };
    const y = { lo: y0, hi: y1 === y0 ? y0 + 1 : y1, format: back(fig.y) };
    const px = v => (sx(v) - x.lo) / (x.hi - x.lo || 1) * w;
    const py = v => h - (sy(v) - y.lo) / (y.hi - y.lo || 1) * h;

    _axes(ctx, w, h, { x, y, pad,
      guides: (fig.guides || []).map(g => ({ ...g, at: g.axis === 'y' ? sy(g.at) : sx(g.at) })) });

    const band = (s, lo, hi, alpha, colour) => {
      if (!s[lo] || !s[hi]) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = colour;
      let open = [];
      const flush = () => {
        if (open.length > 1) {
          ctx.beginPath();
          open.forEach(([t, a], k) => (k ? ctx.lineTo(px(t), py(a)) : ctx.moveTo(px(t), py(a))));
          [...open].reverse().forEach(([t, , b]) => ctx.lineTo(px(t), py(b)));
          ctx.closePath();
          ctx.fill();
        }
        open = [];
      };
      s.x.forEach((t, k) => (s[lo][k] === null || s[hi][k] === null
        ? flush() : open.push([t, s[lo][k], s[hi][k]])));
      flush();
      ctx.restore();
    };
    series.forEach((s, i) => {
      band(s, 'outerLo', 'outerHi', 0.10, Ink.line(i));
      band(s, 'lo', 'hi', 0.22, Ink.line(i));
    });
    series.forEach((s, i) => {
      if (s.points) {
        // A cloud of measurements rather than a path through them.
        ctx.fillStyle = Ink.line(i);
        ctx.globalAlpha = 0.55;
        s.x.forEach((t, k) => {
          const v = s.y[k];
          if (v !== null && Number.isFinite(sy(v))) ctx.fillRect(px(t) - 1.5, py(v) - 1.5, 3, 3);
        });
        ctx.globalAlpha = 1;
        return;
      }
      ctx.strokeStyle = Ink.line(i);
      ctx.lineWidth = s.width || 1.6;
      ctx.beginPath();
      let pen = false;
      s.x.forEach((t, k) => {
        const v = s.y[k];
        if (v === null || !Number.isFinite(sy(v))) { pen = false; return; }
        if (pen) ctx.lineTo(px(t), py(v)); else ctx.moveTo(px(t), py(v));
        pen = true;
      });
      ctx.stroke();
    });
    _chrome(ctx, pad, outer, chrome);
  }
};
