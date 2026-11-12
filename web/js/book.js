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
  SORTS: [['part', 'By part'], ['newest', 'Newest first'], ['oldest', 'Oldest first'],
          ['title', 'A to Z'], ['kind', 'By kind']],
  KINDS: [['overview', 'Overviews', 'Overview'], ['explanation', 'Explanations', 'Explanation'],
          ['experiment', 'Experiments', 'Experiment'], ['study', 'Studies', 'Study'],
          ['reference', 'Reference pages', 'Reference'], ['term', 'Terms', 'Term'],
          ['method', 'Methods and statistics', 'Method']],
  // The mark beside a page in the contents says what kind of page it is (the author,
  // 2026-11-09: the part's colour as a stripe, and a mark for the kind): a study, an
  // experiment, a plan (an experiment with no results yet), a reference page, a guide.
  MARKS: [['study', 'Study'], ['experiment', 'Experiment'], ['plan', 'Plan, not played yet'],
          ['reference', 'Reference'], ['guide', 'Guide']],
  /** The parts the reader left open, kept from one visit to the next; Start and round 3's at first. */
  OPENED: 'gol.book.open',
  /**
   * The research's three documents, read here as pages of the notebook (they
   * were the Research tab's second half): the findings, the reading list and
   * the notes on graphs. By the name docpage.js keeps each under.
   */
  READING: ['notes', 'literature', 'graphs'],
  STATUSES: [['analysed', 'Analysed'], ['ready', 'Ready to run'], ['draft', 'Drafts'],
             ['written', 'Written'], ['planned', 'Not written yet']],
  /** How the reader asked for the contents: an order, three filters and words to look for. */
  view: { sort: 'part', topic: '', kind: '', status: '', query: '' },
  /** The parts the reader shut while looking for something, open again once the search changes. */
  searchShut: new Set(),

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
      el.addEventListener(key === 'query' ? 'input' : 'change', () => this.choose(key, el.value));
    }
    this.toc.addEventListener('click', event => {
      const summary = event.target.closest('details.book-section > summary');
      if (summary) {
        // The details opens or shuts itself; this keeps the choice for the next drawing.
        this.keepOpen(summary.parentElement.dataset.part, !summary.parentElement.open);
        return;
      }
      const doc = event.target.closest('[data-doc]');
      if (doc) {
        event.preventDefault();
        this.showToc(false);
        this.showDoc(doc.dataset.doc);
        return;
      }
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
    // The notebook's title arrives out of the machine's digits, every visit.
    if (typeof Decoder !== 'undefined') Decoder.decode(document.getElementById('bookTitle'));
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

  /** The parts of the index, each a colour and a few topics, in the order the contents show them. */
  parts() {
    return (this.index && this.index.parts) || [];
  },

  /** The part a page belongs to: the part of its home topic, its first. */
  partOf(page) {
    const home = ((page && page.topics) || [])[0];
    return this.parts().find(part => part.topics.includes(home));
  },

  /** What the mark beside a page says it is. */
  mark(page) {
    if (page.kind === 'experiment') return ['draft', 'ready', 'planned'].includes(page.status) ? 'plan' : 'experiment';
    if (page.kind === 'study') return 'study';
    if (['reference', 'term', 'method'].includes(page.kind)) return 'reference';
    return 'guide';
  },

  /**
   * The parts open in the contents: the reader's choice, and nothing else's. The
   * lab's refresh draws the contents again every few seconds, and each drawing
   * used to open the part of the page on screen, so a part shut a moment before
   * opened by itself (the author, 2026-11-09).
   */
  opened() {
    if (!this._opened) {
      let kept = null;
      try { kept = JSON.parse(localStorage.getItem(this.OPENED)); } catch (err) { /* none kept, or a private window */ }
      this._opened = new Set(Array.isArray(kept) ? kept : ['start', 'now']);
    }
    return this._opened;
  },

  /** The reader opened or shut a part: kept for good, or while they search, until the search changes. */
  keepOpen(part, open) {
    if (this.filtered()) {
      if (open) this.searchShut.delete(part); else this.searchShut.add(part);
      return;
    }
    const kept = this.opened();
    if (open) kept.add(part); else kept.delete(part);
    try { localStorage.setItem(this.OPENED, JSON.stringify([...kept])); } catch (err) { /* private window */ }
  },

  /** The reader chose an order, a filter or words to look for: every part with a match opens. */
  choose(key, value) {
    this.view = { ...this.view, [key]: value };
    this.searchShut.clear();
    try { localStorage.setItem(this.LISTING, JSON.stringify(this.view)); } catch (err) { /* private window */ }
    this.renderToc();
  },

  /** Whether the reader is looking for something: then every part with a match is open. */
  filtered(view = this.view) {
    return Boolean(view.query || view.topic || view.kind || view.status);
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
    this.offer(this.choices.topic, this.scopes());
    try { this.view = this.chosen(JSON.parse(localStorage.getItem(this.LISTING))); } catch (err) { /* none kept */ }
    for (const [key, el] of Object.entries(this.choices)) el.value = this.view[key];
    let remembered = null;
    try { remembered = localStorage.getItem(this.STORE); } catch (err) { /* private window */ }
    if (remembered && remembered.startsWith('doc:') && this.documents()[remembered.slice(4)]) {
      await this.showDoc(remembered.slice(4));
      return true;
    }
    const first = this.pages().find(c => c.file);
    await this.open(this.chapter(remembered)?.file ? remembered : first.id);
    return true;
  },

  /** The documents docpage.js holds, by name; none where it is not loaded. */
  documents() {
    if (typeof DocPage === 'undefined') return {};
    return { notes: Notes, literature: Literature, graphs: Graphs };
  },

  /**
   * One of the research's documents, as a page of the notebook. It is not one
   * of book.json's pages, so it has no part and no kicker of its own: the
   * kicker says it is reading, and which.
   */
  async showDoc(id) {
    const doc = this.documents()[id];
    if (!doc) return;
    this.current = `doc:${id}`;
    this.file = null;
    try { localStorage.setItem(this.STORE, this.current); } catch (err) { /* private window */ }
    this.renderToc();
    this.page.style.setProperty('--part', 'transparent');
    this.page.innerHTML = '<p class="md-status">Reading…</p>';
    let source;
    try {
      source = await doc.read();
    } catch (err) {
      this.page.innerHTML = `<p class="md-status">Could not read ${Markdown.escape(doc.source)}: ${
        Markdown.escape(String(err.message))}</p>`;
      return;
    }
    if (this.current !== `doc:${id}`) return;      // another page was opened meanwhile
    this.page.innerHTML = `<p class="book-kicker">Reading · ${Markdown.escape(doc.title)}</p>`
      + Markdown.render(source);
    window.scrollTo(0, 0);
    Markdown.typeset(this.page);
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
    this.page.style.setProperty('--part', (chapter && this.partOf(chapter)?.colour) || 'transparent');
    this.sortable(this.page);
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
   * A long table sorts by any column whose head is clicked: rising, then falling, then back as it was
   * written. A cell that begins with a number sorts as that number ("1,234", "0.95", "−0.2"), and numbers
   * come before words.
   */
  sortable(root) {
    for (const table of root.querySelectorAll('.md-scroll table')) {
      const body = table.tBodies[0];
      if (!body || body.rows.length < 8 || !table.tHead) continue;
      const written = [...body.rows];
      const heads = [...table.tHead.rows[0].cells];
      table.classList.add('md-sortable');
      const key = (row, col) => {
        const text = (row.cells[col]?.textContent ?? '').trim().replace(/^−/, '-');
        const n = parseFloat(text.replace(/,/g, ''));
        return /^-?[\d.]/.test(text) && Number.isFinite(n) ? n : text;
      };
      const order = (a, b) => (typeof a === 'number') === (typeof b === 'number')
        ? (typeof a === 'number' ? a - b : String(a).localeCompare(String(b)))
        : (typeof a === 'number' ? -1 : 1);
      heads.forEach((th, col) => {
        th.title = 'Sort by this column';
        th.addEventListener('click', () => {
          const next = { '': 'up', up: 'down', down: '' }[th.dataset.sort || ''];
          for (const other of heads) delete other.dataset.sort;
          if (next) th.dataset.sort = next;
          const rows = next ? [...written].sort((a, b) => order(key(a, col), key(b, col))) : written;
          if (next === 'down') rows.reverse();
          body.append(...rows);
        });
      });
    }
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
    const part = this.partOf(chapter);
    return `<p class="book-kicker"${part ? ` style="--part: ${Markdown.escape(part.colour)}"` : ''}>${
      part ? `<span class="book-kicker-part">${Markdown.escape(part.title)}</span> · ` : ''
    }<a href="#" data-topic="${Markdown.escape(topic.id)}">${
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
    const doc = String(this.current || '').startsWith('doc:') && this.documents()[this.current.slice(4)];
    if (doc) { text.textContent = doc.title; return; }
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
    const view = { sort: 'part', topic: '', kind: '', status: '', query: '' };
    if (!kept || typeof kept !== 'object') return view;
    const offered = { sort: this.SORTS, kind: this.KINDS, status: this.STATUSES, topic: this.scopes() };
    for (const [key, options] of Object.entries(offered)) {
      if (options.some(([value]) => value === kept[key])) view[key] = kept[key];
    }
    if (typeof kept.query === 'string') view.query = kept.query;
    return view;
  },

  /**
   * What the contents can be narrowed to: everything, a part, or one topic of a part,
   * as [value, words] pairs; a part's value is `part:<id>`.
   */
  scopes() {
    const topics = (this.index && this.index.topics) || [];
    const out = [['', 'Every part']];
    for (const part of this.parts()) {
      out.push([`part:${part.id}`, part.title]);
      for (const id of part.topics) {
        const topic = topics.find(t => t.id === id);
        if (topic && part.topics.length > 1) out.push([id, `\u00a0\u00a0· ${topic.title}`]);
      }
    }
    // An index without parts (an old one) is narrowed by topic alone.
    if (!this.parts().length) out.push(...topics.map(t => [t.id, t.title]));
    return out;
  },

  /**
   * (each page under its home topic, the part's topics in order, each in reading
   * order; the part's id third) or by kind, in groups; by the day a page was added
   * or by title, in one list without a heading. A page shows if it belongs to the
   * part asked for, or any of its topics is the topic asked for, its kind and
   * status are, and every word looked for is in its title or summary.
   */
  listing(view = this.view) {
    const words = view.query.toLowerCase().split(/\s+/).filter(Boolean);
    const inScope = p => !view.topic || (view.topic.startsWith('part:')
      ? this.partOf(p)?.id === view.topic.slice(5) : p.topics.includes(view.topic));
    const shown = this.pages().filter(p => inScope(p)
      && (!view.kind || p.kind === view.kind) && (!view.status || p.status === view.status)
      && words.every(w => `${p.title} ${p.summary || ''}`.toLowerCase().includes(w)));
    const grouped = (groups, inGroup) => groups.map(([key, heading]) => [heading, shown.filter(p => inGroup(p, key))])
      .filter(([, list]) => list.length);
    const order = new Map(this.pages().map((p, i) => [p.id, i]));
    if (view.sort === 'part') {
      return this.parts().map(part => [part.title, shown.filter(p => this.partOf(p) === part).sort((a, b) =>
        part.topics.indexOf(a.topics[0]) - part.topics.indexOf(b.topics[0]) || order.get(a.id) - order.get(b.id)),
      part.id]).filter(([, list]) => list.length);
    }
    if (view.sort === 'kind') return grouped(this.KINDS.map(([id, kinds]) => [id, kinds]), (p, kind) => p.kind === kind);
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
    const marks = Object.fromEntries(this.MARKS);
    const item = c => {
      const state = this.describe(c);
      const part = this.partOf(c);
      const mark = this.mark(c);
      return `<a href="#" data-chapter="${c.id}" class="book-toc-item${
        c.id === this.current ? ' active' : ''}${c.file ? '' : ' unwritten'}"${
        part ? ` style="--part: ${Markdown.escape(part.colour)}"` : ''}>`
        + `<span class="book-toc-title"><span class="book-mark book-mark-${mark}" title="${marks[mark]}"></span>${
          Markdown.inline(c.title)}</span>`
        + (dated && c.added ? `<span class="book-toc-date">${Markdown.escape(c.added)}</span>` : '')
        + (state.label ? `<span class="book-chip book-chip-${state.state}">${state.label}</span>` : '')
        + '</a>';
    };
    const opened = this.opened();
    const groups = this.listing().map(([heading, list, partId]) => {
      const part = partId && this.parts().find(p => p.id === partId);
      if (!part) {
        return `<div class="book-part">${heading ? `<h3>${Markdown.inline(heading)}</h3>` : ''}${
          list.map(item).join('')}</div>`;
      }
      const open = this.filtered() ? !this.searchShut.has(part.id) : opened.has(part.id);
      let last = null;
      const items = list.map(c => {
        const home = c.topics[0];
        const head = home !== last && part.topics.length > 1
          ? `<h4 class="book-topic-head">${Markdown.escape(this.topic(home)?.title || '')}</h4>` : '';
        last = home;
        return head + item(c);
      }).join('');
      return `<details class="book-section" data-part="${part.id}" style="--part: ${Markdown.escape(part.colour)}"${
        open ? ' open' : ''}><summary title="${Markdown.escape(part.about || '')}"><span class="book-section-title">${
        Markdown.escape(part.title)}</span><span class="book-section-count">${list.length}</span></summary>${
        items}</details>`;
    }).join('');
    const legend = `<p class="book-legend">${this.MARKS.map(([id, words]) =>
      `<span><span class="book-mark book-mark-${id}"></span>${Markdown.escape(words.split(',')[0])}</span>`).join('')}</p>`;
    this.list.innerHTML = legend + (groups || '<p class="book-toc-empty">No page matches.</p>') + this.readingToc();
  },

  /**
   * The research's documents, after the parts: shown unless the reader has
   * narrowed the contents to a topic, a kind or a status they do not have,
   * and matched against the words looked for like any page.
   */
  readingToc() {
    const docs = this.documents();
    const view = this.view;
    if (view.topic || view.kind || view.status) return '';
    const words = view.query.toLowerCase().split(/\s+/).filter(Boolean);
    const items = this.READING.filter(id => docs[id]).map(id => [id, docs[id]])
      .filter(([, doc]) => words.every(w => `${doc.title} ${doc.about}`.toLowerCase().includes(w)))
      .map(([id, doc]) => `<a href="#" data-doc="${id}" class="book-toc-item book-doc${
        this.current === `doc:${id}` ? ' active' : ''}" title="${Markdown.escape(doc.about)}">`
        + `<span class="book-toc-title"><span class="book-mark book-mark-reference"></span>${
          Markdown.escape(doc.title)}</span></a>`).join('');
    return items ? `<div class="book-reading"><h3 class="book-reading-head">Reading</h3>${items}</div>` : '';
  },

  /**
   * What a chapter's chip says. An analysed chapter is analysed whatever the
   * lab says of the experiments it has analysed, but one it was given since,
   * which the lab has not finished, is where the lab says it is (Experiment 43,
   * on a page that had analysed round 3: 2026-11-09). Otherwise its experiments
   * are where the lab says they are, and a chapter without one is just written
   * or not.
   */
  describe(chapter, lab = this.lab) {
    const states = lab ? (chapter.experiments || []).map(name => lab.experiments[name]).filter(Boolean) : [];
    const analysed = chapter.status === 'analysed';
    const open = analysed ? states.filter(s => !['finished', 'withdrawn'].includes(s.state)) : states;
    if (analysed && !open.length) return { state: 'analysed', label: 'analysed' };
    const live = this.together(open);
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
    // A link written [name](run:<run id>) opens that world in the viewer, where the runs are (not on
    // the published site, which holds none).
    const world = event.target.closest('a[href^="run:"]');
    if (world) {
      event.preventDefault();
      if (!API.runsInBrowser) {
        App.showView('viewer');
        Viewer.load(world.getAttribute('href').slice(4));
      }
      return;
    }
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
