/*
 * A page that is a markdown file in the repository.
 *
 * The three documents of the research — the findings, the reading list and
 * the notes on graphs — work this way. They are files first, written and
 * edited as text, reviewed as text, diffable as text, and the page is a view
 * of one. There is no second copy of the words to keep in step, which a
 * hand-maintained HTML twin would be within a week, and quietly.
 *
 * They used to be the Research tab's second half; they are read in the
 * Research Notebook now (book.js lists them under "Reading" and draws them as
 * its pages). The paths stay here, where build_site.sh stamps them, so a
 * returning visitor is never handed last week's copy.
 */
const DocPage = {
  /**
   * A document: where it is, and what it is called in the contents.
   * `read()` fetches it once — tens of kilobytes of prose that most visits
   * never open, so it waits until asked, and it never changes, so it is not
   * asked twice. A failure is not kept, so coming back tries again.
   */
  of(source, title, about = '') {
    return {
      source,
      title,
      about,
      text: null,

      async read() {
        if (this.text !== null) return this.text;
        const response = await fetch(this.source);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        this.text = await response.text();
        return this.text;
      }
    };
  }
};

const Notes = DocPage.of('data/Research.md', 'Findings',
  'The road to open-ended evolution: what the research is building, and how it will be measured');
const Literature = DocPage.of('data/Literature.md', 'Literature',
  'What others have found, read and set against this project');
const Graphs = DocPage.of('data/Graphs.md', 'Graphs',
  'The graph measures the site uses, what each one means, and how it is computed');
