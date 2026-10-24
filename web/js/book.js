/*
 * The book: a research programme, written down as it is carried out.
 *
 * Its source is book/ — book.json for the order of the chapters and notes,
 * Markdown files, an SVG file per figure and diagram, a plan per experiment
 * and the results an analysis writes. The same files read as a vault in
 * Obsidian and as pages on GitHub; this file only lays them out here: a list
 * of chapters, the page, and on a computer running gol_server.py, the lab.
 *
 * A page is rendered by markdown.js. A link to another file of the book
 * (`[…](08-thirty-worlds.md#the-runs)`) opens it in place; an image's path
 * is taken from the file it is written in, as Obsidian takes it, and a
 * figure or diagram is set into the page, so it is drawn in the site's font. One thing
 * a page can ask to have drawn, with a comment no other reader shows:
 *
 *     <!-- experiment E08 -->   how far an experiment has got, with ▶ and ⏸
 *
 * Everything else in a chapter — its figures, the thesis of its experiment,
 * the settings of its runs — is written into the Markdown by book_fill.py,
 * so that the file is complete wherever it is read.
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
  EMBEDS: ['experiment'],
  POLL_MS: 5000,
  STORE: 'gol.book.chapter',
  LISTING: 'gol.book.listing',

  // How the contents can be listed: the orders, the kinds of page and the states
  // a page can be in, each with the words the reader chooses it by.
  SORTS: [['topic', 'By topic'], ['newest', 'Newest first'], ['oldest', 'Oldest first'],
          ['title', 'A to Z'], ['kind', 'By kind']],
  KINDS: [['overview', 'Overviews', 'Overview'], ['explanation', 'Explanations', 'Explanation'],
          ['experiment', 'Experiments', 'Experiment'], ['study', 'Studies', 'Study'], ['term', 'Terms', 'Term'],
          ['method', 'Methods and statistics', 'Method']],
  STATUSES: [['analysed', 'Analysed'], ['ready', 'Ready to run'], ['draft', 'Drafts'],
             ['written', 'Written'], ['planned', 'Not written yet']],
  /** How the reader asked for the contents: an order, three filters and words to look for. */
  view: { sort: 'topic', topic: '', kind: '', status: '', query: '' },

  index: null,
  lab: null,
  // What the lab last refused, said in the panel rather than in a dialog.
  refusal: null,
  current: null,
  file: null,
  active: false,
  timer: null,

  init() {
    this.toc = document.getElementById('bookToc');
    this.list = document.getElementById('bookTocList');
    this.page = document.getElementById('bookPage');
    this.choices = { query: document.getElementById('bookSearch'), sort: document.getElementById('bookSort'),
                     topic: document.getElementById('bookTopic'), kind: document.getElementById('bookKind'),
                     status: document.getElementById('bookStatus') };
    this.offer(this.choices.sort, this.SORTS);
    this.offer(this.choices.kind, [['', 'Every kind'], ...this.KINDS.map(([id, kinds]) => [id, kinds])]);
    this.offer(this.choices.status, [['', 'Any status'], ...this.STATUSES]);
    for (const [key, el] of Object.entries(this.choices)) {
      el.addEventListener(key === 'query' ? 'input' : 'change', () => {
        this.view = { ...this.view, [key]: el.value };
        try { localStorage.setItem(this.LISTING, JSON.stringify(this.view)); } catch (err) { /* private window */ }
        this.renderToc();
      });
    }
    this.toc.addEventListener('click', event => {
      const link = event.target.closest('[data-chapter]');
      if (!link) return;
      event.preventDefault();
      this.showToc(false);
      this.open(link.dataset.chapter);
    });
    // The drawer of a narrow screen: the button opens and closes it, a tap
    // beside it or Escape closes it.
    this.toggle = document.getElementById('bookTocToggle');
    this.scrim = document.getElementById('bookScrim');
    this.toggle.addEventListener('click', () => this.showToc(!this.toc.classList.contains('open')));
    this.scrim.addEventListener('click', () => this.showToc(false));
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape') this.showToc(false);
    });
    this.page.addEventListener('click', event => this.onClick(event));
    this.page.addEventListener('change', event => this.onChange(event));
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

  /** `href`, written in the file `from`, as a path under book/ — or null if it leaves the book. */
  resolve(from, href) {
    const parts = from.split('/').slice(0, -1);
    for (const step of decodeURI(href).split('/')) {
      if (step === '..') {
        if (!parts.length) return null;
        parts.pop();
      } else if (step && step !== '.') {
        parts.push(step);
      }
    }
    return parts.join('/');
  },

  /** Every page of the index, written or not, in reading order. */
  pages() {
    return this.index ? this.index.pages : [];
  },

  /** A topic of the index, by its id. A page's first topic is its home. */
  topic(id) {
    return ((this.index && this.index.topics) || []).find(t => t.id === id);
  },

  chapter(id) {
    return this.pages().find(c => c.id === id);
  },

  /**
   * The index, read once, and the page last open. Callers that arrive while it is
   * being read share the one reading, so a page opened from elsewhere (a setting's
   * explanation) is not overwritten by the page the reading then opens.
   */
  load() {
    if (!this.loading) {
      this.loading = this.readIndex().then(read => {
        if (!read) this.loading = null;           // read again next time
      });
    }
    return this.loading;
  },

  async readIndex() {
    if (this.index) return true;
    try {
      this.index = await this.json('book.json');
    } catch (err) {
      this.page.innerHTML = `<p class="md-status">Could not read the notebook: ${
        Markdown.escape(String(err.message))}</p>`;
      return false;
    }
    document.getElementById('bookTitle').textContent = this.index.title;
    this.offer(this.choices.topic, [['', 'Every topic'], ...this.index.topics.map(t => [t.id, t.title])]);
    try { this.view = this.chosen(JSON.parse(localStorage.getItem(this.LISTING))); } catch (err) { /* none kept */ }
    for (const [key, el] of Object.entries(this.choices)) el.value = this.view[key];
    let remembered = null;
    try { remembered = localStorage.getItem(this.STORE); } catch (err) { /* private window */ }
    const first = this.pages().find(c => c.file);
    await this.open(this.chapter(remembered)?.file ? remembered : first.id);
    return true;
  },

  /** The page that explains every setting. */
  SETTINGS: 'pages/settings.md',

  /** The notebook opened at one setting's explanation, as a run's settings link to it. */
  async openSetting(key) {
    App.showView('book');
    await this.load();
    await this.show(this.SETTINGS, key);
  },

  /** A chapter or note of book.json, by its id. */
  async open(id, part = '') {
    const chapter = this.chapter(id);
    if (!chapter) return;
    if (!chapter.file) {
      this.current = id;
      this.file = null;
      this.renderToc();
      this.page.innerHTML = `${this.kicker(chapter)}<h2>${Markdown.inline(chapter.title)}</h2>`
        + `<p class="md-status">Not written yet. ${Markdown.inline(chapter.summary || '')}</p>`;
      return;
    }
    await this.show(chapter.file, part);
  },

  /** Any Markdown file of the book, by its path under book/, scrolled to `part` if given. */
  async show(file, part = '') {
    const chapter = this.pages().find(c => c.file === file) || null;
    this.current = chapter ? chapter.id : null;
    this.file = file;
    if (chapter) {
      try { localStorage.setItem(this.STORE, chapter.id); } catch (err) { /* private window */ }
    }
    this.renderToc();
    let source;
    try {
      source = await this.text(file);
    } catch (err) {
      this.page.innerHTML = `<p class="md-status">Could not read ${Markdown.escape(file)}: ${
        Markdown.escape(String(err.message))}</p>`;
      return;
    }
    if (this.file !== file) return;            // another page was opened meanwhile
    this.page.innerHTML = (chapter ? this.kicker(chapter) : '') + Markdown.render(source, this.EMBEDS);
    // An image's path is relative to the file it is written in.
    const drawings = [];
    for (const img of this.page.querySelectorAll('img[src]')) {
      const src = img.getAttribute('src');
      if (/^[a-z]+:|^\//i.test(src)) continue;
      const path = this.resolve(file, src);
      if (path === null) continue;
      img.src = this.url(path);
      if (path.endsWith('.svg')) drawings.push(this.inline(img, path));
    }
    if (part) {
      await Promise.all(drawings);          // the page has its height before it is scrolled
      this.page.querySelector(`#${CSS.escape(Markdown.slug(decodeURIComponent(part)))}`)?.scrollIntoView();
    } else {
      window.scrollTo(0, 0);
    }
    await Promise.all([...this.page.querySelectorAll('.md-embed')].map(el => this.embed(el)));
    Markdown.typeset(this.page);
  },

  /**
   * A figure or diagram of the book, set into the page in place of its image,
   * so that its text is in the page's typeface: a picture shown as an image
   * sees only the fonts installed on the computer, never the site's web font.
   * The image stays if the drawing cannot be read or is not a plain one.
   */
  async inline(img, path) {
    try {
      const svg = this.safeSvg(await this.text(path));
      if (!img.isConnected) return;               // another page was opened meanwhile
      const box = document.createElement('div');
      box.className = 'md-svg';
      box.setAttribute('role', 'img');
      box.setAttribute('aria-label', img.alt);
      box.innerHTML = svg;
      img.replaceWith(box);
    } catch (err) { /* the image it was */ }
  },

  /** An SVG drawing that only draws: no script, no handlers, no embedded page. */
  safeSvg(text) {
    if (!/^\s*<svg[\s>]/.test(text) || /<script|<foreignObject|\son[a-z]+\s*=|javascript:/i.test(text)) {
      throw new Error('not a plain SVG drawing');
    }
    return text;
  },

  /**
   * Where a page belongs, above its title: its topic, which opens the topic's list
   * in the contents, and the experiments it reports, or else what kind of page it is.
   */
  kicker(chapter) {
    const topic = this.topic((chapter.topics || [])[0]);
    if (!topic) return '';
    const numbers = (chapter.experiments || []).map(name => Number(name.slice(1)));
    const kind = (this.KINDS.find(([id]) => id === chapter.kind) || [])[2];
    const what = numbers.length ? `Experiment${numbers.length > 1 ? 's' : ''} ${
      numbers.length > 1 ? `${numbers.slice(0, -1).join(', ')} and ${numbers.at(-1)}` : numbers[0]}` : kind;
    return `<p class="book-kicker"><a href="#" data-topic="${Markdown.escape(topic.id)}">${
      Markdown.escape(topic.title)}</a>${what ? ` · ${what}` : ''}</p>`;
  },

  // ---- the chapter list -------------------------------------------------

  /** Open or close the chapter list where it is a drawer (narrow screens). */
  showToc(open) {
    if (!this.toggle) return;
    this.toc.classList.toggle('open', open);
    this.scrim.hidden = !open;
    this.toggle.setAttribute('aria-expanded', String(open));
    if (open) {
      const here = this.toc.querySelector('.book-toc-item.active');
      if (here) here.scrollIntoView({ block: 'center' });
    }
  },

  /** The drawer's button names the chapter that is open. */
  labelToggle() {
    if (!this.toggle || !this.index) return;
    const chapter = this.pages().find(c => c.id === this.current);
    const text = this.toggle.querySelector('.book-toc-toggle-text');
    if (!chapter) { text.textContent = 'Contents'; return; }
    text.textContent = chapter.title.replace(/[*_`]/g, '');
  },

  /** A select's options, as [value, words] pairs. */
  offer(select, options) {
    select.innerHTML = options.map(([value, words]) =>
      `<option value="${Markdown.escape(value)}">${Markdown.escape(words)}</option>`).join('');
  },

  /** A kept view, with anything this index no longer offers put back to its default. */
  chosen(kept) {
    const view = { sort: 'topic', topic: '', kind: '', status: '', query: '' };
    if (!kept || typeof kept !== 'object') return view;
    const offered = { sort: this.SORTS, kind: this.KINDS, status: this.STATUSES,
                      topic: ((this.index && this.index.topics) || []).map(t => [t.id]) };
    for (const [key, options] of Object.entries(offered)) {
      if (options.some(([value]) => value === kept[key])) view[key] = kept[key];
    }
    if (typeof kept.query === 'string') view.query = kept.query;
    return view;
  },

  /**
   * The contents as the reader asked for them, as [[heading, pages], …]: by topic
   * (each page under its home topic, in reading order) or by kind, in groups; by
   * the day a page was added or by title, in one list without a heading. A page
   * shows if any of its topics is the one asked for, its kind and status are, and
   * every word looked for is in its title or summary.
   */
  listing(view = this.view) {
    const words = view.query.toLowerCase().split(/\s+/).filter(Boolean);
    const shown = this.pages().filter(p => (!view.topic || p.topics.includes(view.topic))
      && (!view.kind || p.kind === view.kind) && (!view.status || p.status === view.status)
      && words.every(w => `${p.title} ${p.summary || ''}`.toLowerCase().includes(w)));
    const grouped = (groups, inGroup) => groups.map(([key, heading]) => [heading, shown.filter(p => inGroup(p, key))])
      .filter(([, list]) => list.length);
    if (view.sort === 'topic') return grouped(this.index.topics.map(t => [t.id, t.title]), (p, id) => p.topics[0] === id);
    if (view.sort === 'kind') return grouped(this.KINDS.map(([id, kinds]) => [id, kinds]), (p, kind) => p.kind === kind);
    const order = new Map(this.pages().map((p, i) => [p.id, i]));
    const plain = p => p.title.replace(/^(the|a|an)\s+/i, '').replace(/[*_`]/g, '');
    const by = {
      newest: (a, b) => (b.added || '').localeCompare(a.added || '') || order.get(a.id) - order.get(b.id),
      oldest: (a, b) => (a.added || '').localeCompare(b.added || '') || order.get(a.id) - order.get(b.id),
      title: (a, b) => plain(a).localeCompare(plain(b), 'en', { sensitivity: 'base' }),
    }[view.sort];
    return shown.length ? [['', [...shown].sort(by)]] : [];
  },

  /** The contents, as listing() orders and filters them. */
  renderToc() {
    if (!this.index) return;
    this.labelToggle();
    const dated = ['newest', 'oldest'].includes(this.view.sort);
    const groups = this.listing().map(([heading, list]) => {
      const items = list.map(c => {
        const state = this.describe(c);
        return `<a href="#" data-chapter="${c.id}" class="book-toc-item${
          c.id === this.current ? ' active' : ''}${c.file ? '' : ' unwritten'}">`
          + `<span class="book-toc-title">${Markdown.inline(c.title)}</span>`
          + (dated && c.added ? `<span class="book-toc-date">${Markdown.escape(c.added)}</span>` : '')
          + (state.label ? `<span class="book-chip book-chip-${state.state}">${state.label}</span>` : '')
          + '</a>';
      }).join('');
      return `<div class="book-part">${heading ? `<h3>${Markdown.inline(heading)}</h3>` : ''}${items}</div>`;
    }).join('');
    this.list.innerHTML = groups || '<p class="book-toc-empty">No page matches.</p>';
  },

  /**
   * What a chapter's chip says. An analysed chapter is analysed whatever the
   * lab says; otherwise its experiments are where the lab says they are, and a
   * chapter without one is just written or not.
   */
  describe(chapter, lab = this.lab) {
    if (chapter.status === 'analysed') return { state: 'analysed', label: 'analysed' };
    const live = lab && this.together((chapter.experiments || []).map(name => lab.experiments[name]).filter(Boolean));
    if (live) {
      const share = live.total ? Math.floor(100 * live.done / live.total) : 0;
      const labels = {
        ready: 'ready to run', queued: 'queued', running: `running · ${share}%`,
        paused: `paused · ${share}%`, stopped: `not running · ${share}%`,
        finished: 'finished · to analyse', blocked: 'blocked', invalid: 'plan invalid',
        withdrawn: 'withdrawn'
      };
      return { state: live.state, label: labels[live.state] || live.state };
    }
    // Dimmed in the list already; a chip on every one of them would be noise.
    if (!chapter.file) return { state: 'planned', label: '' };
    if (chapter.status === 'ready') return { state: 'ready', label: 'ready to run' };
    if (chapter.status === 'draft') return { state: 'draft', label: 'draft' };
    return { state: 'written', label: '' };
  },

  /**
   * Several experiments of one page as one: their runs added up, and the state of
   * the one furthest from done or most in need of a look — a page is running while
   * any of its experiments runs, and finished only when all are.
   */
  together(states) {
    if (!states.length) return null;
    if (states.length === 1) return states[0];
    const order = ['invalid', 'blocked', 'running', 'queued', 'paused', 'stopped', 'ready', 'finished'];
    const rank = s => (order.includes(s.state) ? order.indexOf(s.state) : order.length);
    const first = states.reduce((a, b) => (rank(b) < rank(a) ? b : a));
    return { state: first.state, done: states.reduce((n, s) => n + (s.done || 0), 0),
             total: states.reduce((n, s) => n + (s.total || 0), 0) };
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
    const topic = event.target.closest('[data-topic]');
    if (topic) {
      event.preventDefault();
      const found = this.topic(topic.dataset.topic);
      if (found) this.show('README.md', Markdown.slug(found.title));
      return;
    }
    const chapter = event.target.closest('[data-chapter]');
    if (chapter) {
      event.preventDefault();
      this.open(chapter.dataset.chapter);
      return;
    }
    const link = event.target.closest('a[data-md]');
    if (link && this.file) {
      const [href, part = ''] = link.dataset.md.split('#');
      const path = href ? this.resolve(this.file, href) : this.file;
      if (path && path.endsWith('.md')) {
        event.preventDefault();
        if (path === this.file && part) {
          this.page.querySelector(`#${CSS.escape(Markdown.slug(decodeURIComponent(part)))}`)?.scrollIntoView();
        } else {
          this.show(path, part);
        }
        return;
      }
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

  // ---- what a page asks to have drawn ----------------------------------

  async embed(el) {
    const { kind, arg } = el.dataset;
    try {
      if (kind === 'experiment') this.experiment(el, arg);
    } catch (err) {
      el.innerHTML = `<p class="md-status">Could not draw ${Markdown.escape(kind)} ${
        Markdown.escape(arg)}: ${Markdown.escape(String(err.message))}</p>`;
    }
  },

  /** How far an experiment has got: drawn only where there is a lab, updated as it runs. */
  experiment(el, name) {
    el.className = 'book-experiment';
    el.dataset.experiment = name;
    el.innerHTML = '<div class="book-live"></div>';
    this.updatePanel(el);
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

  /**
   * How the time left was corrected: the experiment's own pace against the cost model,
   * which counts in proportion to the share of the work already done (gol_lab).
   */
  paceNote(pace) {
    if (!pace || pace.measured == null) return '';
    const times = pace.measured >= 1 ? `${pace.measured.toFixed(1)}× slower` : `${(1 / pace.measured).toFixed(1)}× faster`;
    return ` So far it has run ${times} than the model says; with ${Math.round(100 * pace.share)}% of `
      + `the work done, the time left for runs not yet started is ${pace.correction.toFixed(1)}× the model's.`;
  },

  /** Progress, the time left, the buttons and every run's bar: only where there is a lab. */
  liveHtml(name, status, lab) {
    const chapter = this.pages().find(c => (c.experiments || []).includes(name)) || {};
    const state = this.describe({ ...chapter, experiments: [name] }, lab);
    // A plan the lab cannot carry out, or one withdrawn, has no runs and no estimate:
    // only its reason.
    if (status.state === 'invalid' || status.state === 'withdrawn') {
      return `<div class="book-exp-head"><span class="book-chip book-chip-${state.state}">${state.label}</span></div>`
        + `<p class="book-reason">${Markdown.escape(status.reason || 'the lab cannot read this plan')}</p>`;
    }
    const share = status.total ? status.done / status.total : 0;
    const button = ['ready', 'stopped', 'paused'].includes(status.state)
      ? `<button class="primary small" data-lab="run:${name}">▶ ${status.done ? 'Continue' : 'Run'}</button>`
      : ['running', 'queued'].includes(status.state)
        ? `<button class="ghost small" data-lab="pause">⏸ Pause the lab</button>` : '';
    const left = ['running', 'queued', 'paused', 'stopped'].includes(status.state)
      ? `<span class="book-left">${this.eta(status.secondsLeft)}</span>` : '';
    const workers = Array.from({ length: lab.lab.cores || 8 }, (_, i) => i + 1)
      .map(n => `<option value="${n}"${n === lab.lab.workers ? ' selected' : ''}>${n}</option>`).join('');
    const own = status.workers ?? lab.lab.workers;
    const estimate = status.state === 'finished' || !status.estimate ? '' : `<p class="book-note">Estimated: ${
      this.eta(status.estimate.seconds, '')} on ${own} worker${own === 1 ? '' : 's'} from the start, `
      + `${formatBytes(status.estimate.diskBytes)} of disk (${formatBytes(lab.disk.free)} free), `
      + `up to ${formatNumber(Math.round(status.estimate.peakMB))} MB of memory a run. `
      + `Fitted to ${lab.costs.runs || 'no'} recorded runs so far.${this.paceNote(status.pace)}</p>`;
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
      const where = r.diedAt != null ? `died out at iteration ${formatNumber(r.diedAt)}`
        : r.slowAt != null ? `stopped as too slow at iteration ${formatNumber(r.slowAt)}`
        : `${formatNumber(r.iteration)} of ${formatNumber(r.until)}`;
      // A trial keeps no frames, so there is nothing to open (gol_trial.py).
      const opens = r.iteration > 0 && !r.trial;
      const look = r.diedAt != null ? 'extinct' : r.slowAt != null ? 'slow' : r.state;
      return `<span class="book-run book-run-${look}" data-run="${r.id}" data-started="${
        opens ? 'yes' : 'no'}" title="${Markdown.escape(r.name)} — ${where}${
        opens ? ' · click to open' : ''}">`
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
  }
};
