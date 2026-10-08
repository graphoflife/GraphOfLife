/*
 * Just enough Markdown to render this project's own notes, and its book.
 *
 * Not a general implementation and not trying to be. It covers exactly what
 * research/Research.md and the files in book/ use — headings, tables, rules,
 * lists, block quotes and Obsidian's callouts, fenced code, images, formulas,
 * and inline bold/italic/code/links — measured rather than guessed, and it
 * escapes everything first so a stray `<` in a note about a comparison does
 * not become markup.
 *
 * The book is written to be read in Obsidian and on GitHub as well as here,
 * so it uses their conventions rather than ones of its own:
 *
 *   > [!example]- Title     a callout; with - folded away, with + open
 *   $$ … $$  and  $ … $      a formula in TeX, set apart or in the line,
 *                            typeset by KaTeX once the page is on screen
 *   [text](other.md#part)    a link to another file of the book, which the
 *                            caller opens in place (it carries data-md)
 *   <!-- kind arg -->        a comment, which no reader shows — unless its
 *                            first word is one the caller names, in which
 *                            case it is an empty placeholder for the caller
 *                            to fill (a live experiment's progress, say)
 *
 * Headings get ids, so a page can point at a section of itself, and a link
 * to one (`#…`) stays on the page where a link out of the book opens a new tab.
 *
 * The point of rendering the file rather than keeping a second copy of the
 * words is that there is only ever one document. Research.md is the thing that
 * is edited; the page is a view of it. A hand-maintained HTML twin would be
 * wrong within a week, and wrong quietly.
 */
const Markdown = {
  KATEX: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/',
  katex: null,

  /** Nothing from the source is ever markup until this file decides it is. */
  escape(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  },

  /**
   * Inline spans, innermost first.
   *
   * Code and formulas before everything else: both are literal, so `**x**` or
   * $a_i * b_i$ inside one has to survive as written. Each is replaced by a
   * placeholder that cannot occur in the text and put back at the end. Images
   * go before links, which they would otherwise be taken for.
   */
  inline(text) {
    const kept = [];
    const keep = html => `\u0000${kept.push(html) - 1}\u0000`;
    let out = this.escape(text)
      .replace(/`([^`]+)`/g, (_, body) => keep(`<code>${body}</code>`))
      .replace(/(^|[^\\$])\$(?=\S)([^$\n]*?\S)\$(?!\d)/g,
        (_, before, tex) => before + keep(`<span class="md-math-inline">${tex}</span>`))
      .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g,
        (_, alt, src) => keep(`<img src="${src}" alt="${alt.replace(/"/g, '&quot;')}" loading="lazy">`));

    out = out
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => href.startsWith('#')
        ? `<a href="${href}">${label}</a>`
        : /^[a-z]+:/i.test(href)
          ? `<a href="${href}" target="_blank" rel="noopener">${label}</a>`
          : `<a href="${href}" data-md="${href}">${label}</a>`)
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');

    // A placeholder can sit inside another (a formula in a link's words), so
    // they are put back until none is left.
    while (/\u0000\d+\u0000/.test(out)) {
      out = out.replace(/\u0000(\d+)\u0000/g, (_, i) => kept[Number(i)]);
    }
    return out;
  },

  /** One table, from the rows that make it up. Header, divider, body. */
  table(rows) {
    // A pipe inside a code span or a formula is not a cell's edge, nor is an
    // escaped one, `\|`, which then stands for the pipe itself — as in GitHub
    // and Obsidian, so a cell can hold |x|.
    const cells = row => row.replace(/^\||\|$/g, '').split(/(?<!\\)\|(?=(?:[^`]*`[^`]*`)*[^`]*$)/)
      .map(c => c.trim().replace(/\\\|/g, '|'));
    const head = cells(rows[0]);
    const body = rows.slice(2).map(cells);      // rows[1] is the --- divider
    return '<div class="md-scroll"><table>'
      + `<thead><tr>${head.map(c => `<th>${this.inline(c)}</th>`).join('')}</tr></thead>`
      + `<tbody>${body.map(r =>
          `<tr>${r.map(c => `<td>${this.inline(c)}</td>`).join('')}</tr>`).join('')}</tbody>`
      + '</table></div>';
  },

  /** An id for a heading: its words, lower case, joined by dashes. */
  slug(text) {
    return String(text).toLowerCase().replace(/<[^>]*>/g, '').replace(/&[a-z]+;/g, '')
      .replace(/[^a-z0-9À-ɏ]+/g, '-').replace(/^-+|-+$/g, '') || 'section';
  },

  /**
   * A block quote, or an Obsidian callout when its first line says
   * `[!kind] title`: rendered as a page of its own, so a callout can hold
   * lists, tables and formulas. A folded one (`-`) is a <details>.
   */
  quote(lines, embeds, ids) {
    const head = lines[0].match(/^\[!([\w-]+)\]([-+]?)\s*(.*)$/);
    if (!head) return `<blockquote>${this.render(lines.join('\n'), embeds, ids)}</blockquote>`;
    const [, kind, fold, title] = head;
    const name = this.inline(title || kind[0].toUpperCase() + kind.slice(1));
    const body = this.render(lines.slice(1).join('\n'), embeds, ids);
    const cls = `md-callout md-callout-${this.escape(kind.toLowerCase())}`;
    return fold
      ? `<details class="${cls}"${fold === '+' ? ' open' : ''}><summary>${name}</summary>${body}</details>`
      : `<div class="${cls}"><p class="md-callout-title">${name}</p>${body}</div>`;
  },

  /**
   * The HTML for `source`. `embeds` names the first words of comments that
   * are not comments but stand for something the caller draws.
   */
  render(source, embeds = [], ids = new Map()) {
    const lines = String(source).replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let list = null;                            // 'ul' | 'ol' | null
    let paragraph = [];
    // The list item being built, still as source text. Held rather than
    // rendered on sight because it can wrap, and a bold or a link that opens
    // on one line and closes on the next only pairs up if the whole item is
    // handed to `inline` in one piece.
    let item = null;

    const closeItem = () => {
      if (item === null) return;
      out.push(`<li>${this.inline(item)}</li>`);
      item = null;
    };
    const closeList = () => {
      closeItem();
      if (list) { out.push(`</${list}>`); list = null; }
    };
    const closeParagraph = () => {
      if (!paragraph.length) return;
      out.push(`<p>${this.inline(paragraph.join(' '))}</p>`);
      paragraph = [];
    };
    const close = () => { closeParagraph(); closeList(); };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Fenced code, taken verbatim to its closing fence.
      if (/^\s*```/.test(line)) {
        close();
        const body = [];
        while (++i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i]);
        out.push(`<pre><code>${this.escape(body.join('\n'))}</code></pre>`);
        continue;
      }

      // A formula set apart: $$ on a line of its own to the next, or on one line.
      const display = line.match(/^\s*\$\$(.*?)(\$\$)?\s*$/);
      if (display) {
        close();
        let tex = display[1];
        if (!display[2]) {
          const body = [tex];
          while (++i < lines.length && !/\$\$\s*$/.test(lines[i])) body.push(lines[i]);
          if (i < lines.length) body.push(lines[i].replace(/\$\$\s*$/, ''));
          tex = body.join('\n');
        }
        out.push(`<div class="md-math">${this.escape(tex.trim())}</div>`);
        continue;
      }

      // A comment line: nothing, or a place for what the caller draws.
      const comment = line.match(/^\s*<!--\s*(\S+)\s*(.*?)\s*-->\s*$/);
      if (comment) {
        if (embeds.includes(comment[1])) {
          close();
          out.push(`<div class="md-embed" data-kind="${this.escape(comment[1])}"`
                   + ` data-arg="${this.escape(comment[2]).replace(/"/g, '&quot;')}"></div>`);
        }
        continue;
      }

      // A table is a run of lines starting with a pipe, and is only a table if
      // the second one is the divider — otherwise it is prose about pipes.
      if (line.startsWith('|') && /^\|[\s:|-]+\|$/.test(lines[i + 1] || '')) {
        close();
        const rows = [];
        while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++]);
        i--;
        out.push(this.table(rows));
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        close();
        const level = Math.min(6, heading[1].length + 1);   // h1 in the file is the page's h2
        const html = this.inline(heading[2]);
        const base = this.slug(html);
        const seen = ids.get(base) || 0;
        ids.set(base, seen + 1);
        const id = seen ? `${base}-${seen + 1}` : base;
        out.push(`<h${level} id="${id}">${html}</h${level}>`);
        continue;
      }

      if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { close(); out.push('<hr>'); continue; }

      // A block quote is every quoted line in a row, rendered as a page of its own.
      if (/^\s*>/.test(line)) {
        close();
        const body = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ''));
        i--;
        out.push(this.quote(body, embeds, ids));
        continue;
      }

      const bullet = line.match(/^\s*[-*]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
      if (bullet || numbered) {
        closeParagraph();
        closeItem();
        const want = bullet ? 'ul' : 'ol';
        if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
        item = (bullet || numbered)[1];
        continue;
      }

      if (!line.trim()) { close(); continue; }

      // An indented line continues the item above it. Research.md wraps its
      // longer points that way, and a sentence broken across two lines has to
      // be rejoined before anything is rendered.
      if (item !== null && /^\s{2,}\S/.test(line)) {
        item += ` ${line.trim()}`;
        continue;
      }
      closeList();
      paragraph.push(line.trim());
    }
    close();
    return out.join('\n');
  },

  /**
   * Typeset the formulas under `root` with KaTeX, fetched the first time a
   * page has one. Without it — offline, say — a formula stays as its TeX,
   * which is how Obsidian's source view shows it too.
   */
  async typeset(root) {
    const all = root.querySelectorAll('.md-math, .md-math-inline');
    if (!all.length) return;
    try {
      this.katex = this.katex || new Promise((resolve, reject) => {
        const css = document.createElement('link');
        css.rel = 'stylesheet';
        css.href = `${this.KATEX}katex.min.css`;
        document.head.appendChild(css);
        const script = document.createElement('script');
        script.src = `${this.KATEX}katex.min.js`;
        script.onload = () => resolve(window.katex);
        script.onerror = () => reject(new Error('KaTeX did not load'));
        document.head.appendChild(script);
      });
      const katex = await this.katex;
      for (const el of all) {
        if (el.dataset.typeset) continue;
        katex.render(el.textContent, el, { displayMode: el.classList.contains('md-math'),
                                           throwOnError: false });
        el.dataset.typeset = 'yes';
      }
    } catch (err) {
      this.katex = null;                 // try again on the next page
    }
  }
};
