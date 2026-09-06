/*
 * research/Research.md, rendered.
 *
 * The state of knowledge on the one claim this project is testing — what has
 * been measured, what has been retracted, and what is still open. It is a file
 * in the repository first: it is written and edited as text, it goes through
 * review as text, and its diffs are readable. This tab is a view of that file
 * and not a copy of it, so there is no second document to keep in step.
 *
 * Fetched rather than built in, for the same reason the Explanation fetches
 * the script it walks through: the words are the artefact, and shipping a
 * transformed copy of them is how the two come to disagree.
 */
const Notes = {
  SOURCE: 'data/Research.md',

  /**
   * Fetch and render, once.
   *
   * Fifty kilobytes of prose that most visitors never open, so it waits until
   * somebody asks for it — and once asked, it never changes, so it is not
   * asked for twice.
   */
  async render() {
    const host = document.getElementById('research-notes');
    if (!host || this.painted) return;
    this.painted = true;

    host.innerHTML = '<div class="md"><p class="md-status">Reading the notes…</p></div>';
    try {
      const response = await fetch(this.SOURCE);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const source = await response.text();
      host.innerHTML = `<div class="md">${Markdown.render(source)}</div>`;
    } catch (err) {
      // Painted stays true only if it worked, so a failure can be retried by
      // coming back to the tab rather than reloading the page.
      this.painted = false;
      host.innerHTML = '<div class="md"><p class="md-status">Could not read '
        + `<code>${this.SOURCE}</code>: ${err.message}. It is in the repository `
        + 'under <code>research/Research.md</code>.</p></div>';
    }
  }
};
