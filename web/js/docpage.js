/*
 * A page that is a markdown file in the repository.
 *
 * Both documents in the Research tab work this way. They are files first —
 * written and edited as text, reviewed as text, diffable as text — and the
 * page is a view of one. There is no second copy of the words to keep in step,
 * which a hand-maintained HTML twin would be within a week, and quietly.
 *
 * The Literature page used to be five hundred lines of prose held in a
 * JavaScript array. Same words, but they could not be read outside a browser,
 * could not be diffed usefully, and had to be edited around markup.
 */
const DocPage = {
  /**
   * A page module for one document.
   *
   * Returns something with `render()`, which is all Research.MODES asks of a
   * page — so a new document is one line there and one file here.
   */
  of(hostId, source) {
    return {
      source,
      painted: false,

      /**
       * Fetch and render, once.
       *
       * Tens of kilobytes of prose that most visits never open, so it waits
       * until asked; and it never changes, so it is not asked twice.
       */
      async render() {
        const host = document.getElementById(hostId);
        if (!host || this.painted) return;
        this.painted = true;

        host.innerHTML = '<div class="md"><p class="md-status">Reading…</p></div>';
        try {
          const response = await fetch(this.source);
          if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
          host.innerHTML = `<div class="md">${Markdown.render(await response.text())}</div>`;
        } catch (err) {
          // Only stays painted if it worked, so a failure is retried by coming
          // back to the tab rather than by reloading the page.
          this.painted = false;
          host.innerHTML = '<div class="md"><p class="md-status">Could not read '
            + `<code>${this.source}</code>: ${err.message}.</p></div>`;
        }
      }
    };
  }
};

const Notes = DocPage.of('research-notes', 'data/Research.md');
const Literature = DocPage.of('research-literature', 'data/Literature.md');
