/*
 * The Explanation, as three things at once.
 *
 * On the left a real run, stepped through a stage at a time. In the middle what
 * is happening. On the right the actual script, scrolled to the lines doing it,
 * with everything else dimmed. The whole script is on the page, so it can be
 * copied and run.
 *
 * The run is real and was chosen, not staged: seed 5 from iteration 129, picked
 * by searching for a window where every mechanic fires at least once and the
 * world stays small enough to follow one agent through it. Nothing in the
 * pictures is drawn for effect — each stage is a snapshot the engine handed
 * over while it was running.
 *
 * The code regions are found by searching the script for anchor text rather
 * than by line number, so editing explain_minimal.py cannot silently point a
 * step at the wrong lines. If an anchor stops matching, that step says so
 * instead of highlighting something arbitrary.
 */
const Explain = {
  RUN: 'data/explain-run.json',
  SCRIPT: 'py/explain_minimal.py',

  at: 0,
  cycle: 0,          // which recorded iteration the cycle steps are showing
  stages: null,
  script: null,
  lines: [],
  view: null,
  started: false,
  active: false,

  /**
   * The walk-through.
   *
   * `stage` names the recorded snapshot to draw and `effect` the animation laid
   * over it. `code` is a pair of strings found in the script, marking the first
   * and last line to light up — text and not line numbers, so editing
   * explain_minimal.py cannot silently point a step at the wrong place.
   */
  /**
   * The emblem behind each step's words.
   *
   * One light grey on nothing — no second colour, no fill behind the strokes —
   * so it reads as a watermark on the card rather than as an illustration
   * competing with the graph. Drawn here rather than kept as files because
   * they are six shapes and belong with the step that uses them.
   */
  MARK: '#c8cfd9',
  EMBLEMS: {
    eye: `<path d="M2.2 12C5.6 6.8 8.7 4.6 12 4.6s6.4 2.2 9.8 7.4c-3.4 5.2-6.5 7.4-9.8 7.4S5.6 17.2 2.2 12z"/>
          <circle cx="12" cy="12" r="3.5" FILL/>`,
    heart: `<path d="M12 21.2S2.6 14.9 2.6 8.9A5.4 5.4 0 0 1 12 5.6a5.4 5.4 0 0 1 9.4 3.3c0 6-9.4 12.3-9.4 12.3z" FILL/>`,
    skull: `<path fill-rule="evenodd" d="M12 1.8c-5 0-8.7 3.6-8.7 8.2 0 2.6 1.2 4.5 2.7 5.6v2.2c0 .9.7 1.6 1.6 1.6h.7v2.8h2.1v-2.8h3.2v2.8h2.1v-2.8h.7c.9 0 1.6-.7 1.6-1.6v-2.2c1.5-1.1 2.7-3 2.7-5.6 0-4.6-3.7-8.2-8.7-8.2zM8.6 8.1a2.1 2.1 0 1 0 0 4.2 2.1 2.1 0 0 0 0-4.2zm6.8 0a2.1 2.1 0 1 0 0 4.2 2.1 2.1 0 0 0 0-4.2zM12 13.4l1.3 2.6h-2.6z" FILL/>`,
    swords: `<path d="M4 3.5 18.5 19.5M20 3.5 5.5 19.5M16.9 14.5 13.7 17.5M7.1 14.5 10.3 17.5M15.3 16 18.5 19.5M8.7 16 5.5 19.5"/>`,
    flag: `<path d="M5.5 2.6V21.4"/><path d="M5.5 3.9c4.1-2.1 7.2 2 11.3 0v8.2c-4.1 2.1-7.2-2-11.3 0z" FILL/>`,
    dna: `<path d="M12.00 2.60C13.30 3.12 14.60 3.64 15.52 4.17C16.45 4.69 16.99 5.21 17.00 5.73C17.01 6.26 16.49 6.78 15.58 7.30C14.67 7.82 13.38 8.34 12.08 8.87C10.78 9.39 9.48 9.91 8.54 10.43C7.60 10.96 7.03 11.48 7.00 12.00C6.97 12.52 7.47 13.04 8.36 13.57C9.26 14.09 10.53 14.61 11.83 15.13C13.14 15.66 14.45 16.18 15.40 16.70C16.36 17.22 16.94 17.74 17.00 18.27C17.05 18.79 16.57 19.31 15.69 19.83C14.82 20.36 13.55 20.88 12.25 21.40"/><path d="M12.00 2.60C10.70 3.12 9.40 3.64 8.48 4.17C7.55 4.69 7.01 5.21 7.00 5.73C6.99 6.26 7.51 6.78 8.42 7.30C9.33 7.82 10.62 8.34 11.92 8.87C13.22 9.39 14.52 9.91 15.46 10.43C16.40 10.96 16.97 11.48 17.00 12.00C17.03 12.52 16.53 13.04 15.64 13.57C14.74 14.09 13.47 14.61 12.17 15.13C10.86 15.66 9.55 16.18 8.60 16.70C7.64 17.22 7.06 17.74 7.00 18.27C6.95 18.79 7.43 19.31 8.31 19.83C9.18 20.36 10.45 20.88 11.75 21.40"/><path d="M8.90 3.94H15.10M7.47 6.63H16.53M7.00 12.00H17.00M7.58 17.37H16.42M8.70 20.06H15.30"/>`
  },

  /** One emblem, as a background-image the card can wear. */
  emblem(name) {
    const body = this.EMBLEMS[name];
    if (!body) return 'none';
    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" ` +
      `fill="none" stroke="${this.MARK}" stroke-width="1.6" ` +
      `stroke-linecap="round" stroke-linejoin="round" opacity="0.17">` +
      body.replace(/FILL/g, `fill="${this.MARK}" stroke="none"`) +
      `</svg>`;
    return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
  },

  STEPS: [
    {
      title: 'An Individual Node with a Brain', intro: true,
      stage: 'repro.observe', solo: true, effect: 'brains bare',
      code: ['class Brain:', 'w += mask * np.random.randn'],
      text: `A node is one agent. It holds tokens and a small neural network.
        <p>The network is never trained. It is made at random, copied when the
        agent reproduces or conquers, and jittered slightly on every copy. That
        is the only way behaviour changes.</p>
        <p>Inputs go in one column per thing observed; outputs come back the
        same way.</p>`
    },
    {
      title: 'A Network of Individuals', intro: true,
      stage: 'repro.observe', effect: 'brains arrive',
      code: ['# ---- the starting graph ----', 'self.next_id = AGENTS'],
      text: `Agents are joined in a ring to their nearest few, then a fifth of
        the links are redrawn at random. Short paths, still mostly local.
        <p>A fixed pile of tokens is then shared out. That total never changes
        again: every rule after this only moves tokens between agents.</p>`
    },
    {
      title: 'Start of Simulation Loop', intro: true,
      stage: 'repro.observe', effect: 'brains',
      code: ['    def step(self) -> None:', '        self.game()'],
      text: `One iteration is two phases: reproduction, then the game. Each ends
        by removing the dead.
        <p>Everything below repeats. The number beside each step is the
        iteration of the recorded run being shown.</p>`
    },

    {
      title: 'Observation — Reproduction Phase', emblem: 'eye',
      stage: 'repro.observe', effect: 'eyes',
      code: ['    def observe(self, u: int', 'return self.brains[u].forward(x)'],
      text: `Each agent reads its whole neighbourhood in one pass: its own
        tokens and degree, each neighbour's, and last phase's messages. Both
        counts are logged, so what carries is the order of magnitude. It reads
        itself too — that is how it knows what it holds.
        <p>The same pass decides what to say: a short vector to every
        neighbour and one to itself, delivered at the end of the phase so
        everyone reads the same generation.</p>`
    },
    {
      title: 'Reproduction and Handover', emblem: 'heart',
      stage: 'repro.born', effect: 'brains inherit',
      code: ['# ---- how much of me goes into a child ----', 'self.unlink(u, v)'],
      text: `An agent spends a share of its tokens on a child. The child starts
        with exactly that; no tokens are created.
        <p>The child inherits a mutated copy of the brain and is linked to
        neighbours the parent chooses. The parent may instead <b>hand over</b> a
        link: it drops that connection and the child takes its place.</p>
        <p>Newborns and the links they arrive on are green. What the child was
        given crosses the link it came down.</p>`
    },
    {
      title: 'Elimination', emblem: 'skull',
      stage: 'repro.cleanup', effect: 'brains',
      code: ['    def cleanup(self) -> None:', 'self.tokens[random.choice(survivors)] += 1'],
      text: `Two removals, in order, after every phase.
        <p>First, agents holding no tokens. Then, of what remains, everything
        outside the largest connected piece — a group linked through a single
        agent comes adrift when that agent starves.</p>
        <p>Tokens held by the dead are redistributed at random over the
        survivors.</p>`
    },
    {
      title: 'Observation — Game Phase', emblem: 'eye',
      stage: 'game.observe', effect: 'eyes',
      code: ['    def game(self) -> None:', 'y = self.observe(u, targets)'],
      text: `The second phase begins with the same single pass, messages and
        all.
        <p>The graph has changed since the first: children have been born,
        links have moved, the starved are gone. What an agent writes now is
        what its neighbours read in the next reproduction phase.</p>`
    },
    {
      title: 'Colonel Blotto Game', emblem: 'swords', emblemAfter: 'flag',
      stage: 'game.stake', effect: 'brains stakes conquer',
      code: [['# ---- everyone stakes at once ----', 'self.unlink(a, b)'],
             ['def resolve(staked: dict', '    return hegemon']],
      text: `Every agent stakes its whole pile across itself and its
        neighbours, spread by score or all on one node. A node's new balance is
        everything staked on it; nothing is destroyed.
        <p>Its largest staker is the <b>hegemon</b>. Against it, every staker
        that flagged part of its stake as revolt, sorted weakest first: at the
        first rung where the class below outweighs everyone above it plus the
        hegemon, the node goes to that rung's strongest staker instead. Ties
        there are drawn at random; nothing else is.</p>
        <p>The winner's brain is copied into the node — the only selection step
        in the algorithm. Links that carried nothing are cut.</p>`
    },
    {
      title: 'Elimination', emblem: 'skull',
      stage: 'game.cleanup', effect: 'brains',
      code: ['    def cleanup(self) -> None:', 'self.tokens[random.choice(survivors)] += 1'],
      text: `The same two removals, because cleanup runs after both phases.
        <p>Cutting the unused links is what usually strands an agent: one whose
        every connection went unused is left attached to nothing.</p>`
    },
    {
      title: 'Mutation', emblem: 'dna',
      stage: 'game.mutate', effect: 'brains mutate',
      code: ['# ---- everyone mutates ----', 'brain.mutate()'],
      text: `Every surviving brain is jittered — not only newborns, not only
        winners. Every agent, every iteration.
        <p>It happens after the cleanup, so brains about to be removed are not
        jittered first. This is the only source of variation.</p>
        <p>The loop then returns to the reproduction observation.</p>`
    }
  ]
};

Object.assign(Explain, {
  init() {
    this.canvas = document.getElementById('explainCanvas');
    if (!this.canvas) return;
    this.textEl = document.getElementById('explainText');
    this.noteEl = document.querySelector('.ex-note');
    this.codeEl = document.getElementById('explainCode');
    this.countEl = document.getElementById('explainCount');

    document.getElementById('explainPrev').addEventListener('click', () => this.go(-1));
    document.getElementById('explainNext').addEventListener('click', () => this.go(1));
    document.addEventListener('keydown', e => {
      if (App.view !== 'explain') return;
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') { this.go(-1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { this.go(1); e.preventDefault(); }
    });

    if (!this.started) {
      this.started = true;
      requestAnimationFrame(t => this.frame(t));
    }
  },

  setActive(active) {
    this.active = active;
    if (active && !this.stages && !this._loading) this.load();
  },

  async load() {
    this._loading = true;
    try {
      const [run, script] = await Promise.all([
        fetch(this.RUN).then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))),
        fetch(this.SCRIPT).then(r => r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`)))
      ]);

      this.meta = run;
      this.script = script;
      this.lines = script.split('\n');
      this.stages = run.stages;

      // Stages in the order they happened, grouped by iteration, so a step can
      // ask for "game.stake in the third recorded iteration".
      this.byIteration = new Map();
      for (const stage of this.stages) {
        if (!this.byIteration.has(stage.iteration)) this.byIteration.set(stage.iteration, {});
        this.byIteration.get(stage.iteration)[stage.step] = stage;
      }
      this.iterations = [...this.byIteration.keys()].sort((a, b) => a - b);

      // The staking and the conquest are one step to a reader — the tokens
      // move, and where they land is who won — but the engine notes them
      // separately, and it is the staking's snapshot that still has the piles
      // as they stood. Carry who took what back onto it.
      for (const stages of this.byIteration.values()) {
        const stake = stages['game.stake'], conquer = stages['game.conquer'];
        if (stake && conquer) stake.marks.taken = conquer.marks.taken || [];
      }

      this.renderCode();
      this.view = StepView.create(this.canvas);

      // One scale for the green dots across the whole walk-through. The
      // richest holding in each stage, at the 95th percentile of those: a
      // single freak pile in one stage should not shrink every other step's
      // ring, and a step where nobody is rich should not inflate it.
      const peaks = this.stages
        .map(stage => Math.max(0, ...stage.tokens))
        .sort((a, b) => a - b);
      this.view.tokenRef = peaks.length
        ? peaks[Math.min(peaks.length - 1, Math.floor(peaks.length * 0.95))]
        : 1;

      this.showStep(0, { carry: false });
    } catch (err) {
      this.textEl.innerHTML =
        `<h3 class="ex-title">The walk-through could not be loaded</h3>` +
        `<p>${err.message}</p>`;
      console.warn('explanation:', err.message);
    } finally {
      this._loading = false;
    }
  },

  /**
   * The whole script, one line per row, so a region can be lit up by adding a
   * class rather than by re-rendering the text.
   */
  renderCode() {
    const coloured = this.colour(this.lines);
    this.codeEl.innerHTML = coloured.map((html, i) =>
      `<div class="explain-line" data-line="${i}">` +
      `<span class="explain-no">${i + 1}</span>` +
      `<span class="explain-src">${html || '&nbsp;'}</span></div>`).join('');
    this.lineEls = [...this.codeEl.querySelectorAll('.explain-line')];
  },

  escape(text) {
    return text.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  },

  /**
   * Colour the script, one line at a time, carrying state between them.
   *
   * Written here rather than pulled in: a highlighter is a few dozen lines for
   * the subset of Python this one file uses, and a dependency for the rest.
   *
   * It has to work line by line, because each line is its own row so that a
   * region can be lit. That makes one thing essential — remembering which
   * lines are inside a docstring. Most sections of the script open with one,
   * and a highlighter that loses track starts colouring the prose as code and
   * the code as prose and never recovers.
   */
  colour(lines) {
    const KEYWORDS = new Set(['and', 'as', 'assert', 'async', 'await', 'break',
      'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
      'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
      'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
      'True', 'False', 'None']);
    const BUILTINS = new Set(['abs', 'all', 'any', 'bool', 'dict', 'enumerate',
      'float', 'int', 'len', 'list', 'max', 'min', 'object', 'print', 'range',
      'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip', 'isinstance',
      'super', 'type', 'open']);

    const tag = (cls, text) => `<span class="${cls}">${this.escape(text)}</span>`;
    const TRIPLES = ['"""', "'''"];
    const out = [];
    let triple = null;              // the delimiter we are inside, if any

    for (const line of lines) {
      let html = '';
      let i = 0;

      if (triple) {
        const end = line.indexOf(triple);
        if (end < 0) { out.push(tag('tok-str', line)); continue; }
        html += tag('tok-str', line.slice(0, end + 3));
        i = end + 3;
        triple = null;
      }

      while (i < line.length) {
        const ch = line[i];
        const rest = line.slice(i);

        if (ch === '#') { html += tag('tok-com', rest); break; }

        const three = line.slice(i, i + 3);
        if (TRIPLES.includes(three)) {
          const end = line.indexOf(three, i + 3);
          if (end < 0) { html += tag('tok-str', rest); triple = three; break; }
          html += tag('tok-str', line.slice(i, end + 3));
          i = end + 3;
          continue;
        }

        if (ch === '"' || ch === "'") {
          let j = i + 1;
          while (j < line.length && line[j] !== ch) {
            if (line[j] === '\\') j++;
            j++;
          }
          html += tag('tok-str', line.slice(i, Math.min(j + 1, line.length)));
          i = j + 1;
          continue;
        }

        if (ch === '@' && /[A-Za-z_]/.test(line[i + 1] || '')) {
          const word = /^@[\w.]+/.exec(rest)[0];
          html += tag('tok-dec', word);
          i += word.length;
          continue;
        }

        if (/[0-9]/.test(ch) && !/\w/.test(line[i - 1] || '')) {
          const num = /^[0-9][\w.]*/.exec(rest)[0];
          html += tag('tok-num', num);
          i += num.length;
          continue;
        }

        if (/[A-Za-z_]/.test(ch)) {
          const word = /^[A-Za-z_]\w*/.exec(rest)[0];
          let cls = null;
          if (KEYWORDS.has(word)) cls = 'tok-kw';
          else if (word === 'self' || word === 'cls') cls = 'tok-self';
          else if (/\b(def|class)\s+$/.test(line.slice(0, i))) cls = 'tok-def';
          else if (BUILTINS.has(word)) cls = 'tok-bi';
          html += cls ? tag(cls, word) : this.escape(word);
          i += word.length;
          continue;
        }

        html += this.escape(ch);
        i++;
      }
      out.push(html);
    }
    return out;
  },

  /**
   * Which lines a step is about, found by searching for its anchors.
   *
   * By text and not by line number, so editing the script cannot quietly point
   * a step at the wrong place. A missing anchor returns nothing and the step
   * says so, which is a great deal better than lighting up whatever happens to
   * live at those numbers now.
   */
  /**
   * The lines a step is about.
   *
   * A step usually points at one run of the script, but not always: the game
   * stakes in one place and works out who won in another, with the whole
   * cleanup sitting between them. Naming both is better than lighting the
   * hundred and forty lines from the first to the last.
   */
  regions(step) {
    const pairs = Array.isArray(step.code[0]) ? step.code : [step.code];
    const found = [];
    for (const [head, tail] of pairs) {
      const from = this.lines.findIndex(l => l.includes(head));
      if (from < 0) continue;
      const rest = this.lines.slice(from);
      const offset = rest.findIndex((l, i) => i > 0 && l.includes(tail));
      if (offset < 0) continue;
      found.push({ from, to: from + offset });
    }
    return found;
  },

  go(by) {
    let next = this.at + by;
    // Past the last step, round again on the next recorded iteration.
    const firstCycle = this.STEPS.findIndex(s => !s.intro);
    if (next >= this.STEPS.length) {
      next = firstCycle;
      this.cycle = (this.cycle + 1) % Math.max(1, this.iterations.length);
    } else if (next < 0) {
      next = 0;
    }
    this.showStep(next);
  },

  /**
   * One agent on its own, for the opening step.
   *
   * Cut from the real recording rather than invented: the busiest agent in the
   * first stage, with its links dropped. What is being introduced is a node, so
   * the picture is a node.
   */
  soloStage(stage) {
    let best = 0;
    stage.tokens.forEach((t, i) => { if (t > stage.tokens[best]) best = i; });
    return {
      step: stage.step,
      iteration: stage.iteration,
      ids: [stage.ids[best]],
      tokens: [stage.tokens[best]],
      edges: [],
      marks: {}
    };
  },

  showStep(index, { carry = true } = {}) {
    if (!this.stages) return;
    this.at = Math.max(0, Math.min(this.STEPS.length - 1, index));
    const step = this.STEPS[this.at];

    // The title and where you are sit at the top of the explanation itself,
    // not in a bar above all three panes: they belong to the words.
    const first = this.STEPS.findIndex(s => !s.intro);
    const where = step.intro
      ? 'Before the loop'
      : `Step ${this.at - first + 1} / ${this.STEPS.length - first}`;
    this.textEl.innerHTML =
      `<h3 class="ex-title">${step.title}</h3>` +
      `<p class="ex-step">${where}</p>` +
      `<p>${step.text}</p>`;
    this.noteEl.style.backgroundImage = this.emblem(step.emblem);
    this._shownEmblem = step.emblem;
    // Only the emblem needs the room it takes; a step without one gets its
    // full width back.
    this.noteEl.style.paddingRight = step.emblem ? '' : '17px';
    this.noteEl.scrollTop = 0;
    // The bar keeps only how far through the whole walk you are; the step's own
    // number lives with its words.
    this.countEl.textContent = `${this.at + 1} of ${this.STEPS.length}`;

    // ---- the code ----
    const regions = this.regions(step);
    this.lineEls.forEach((el, i) => {
      el.classList.toggle('lit', regions.some(r => i >= r.from && i <= r.to));
    });
    if (regions.length) {
      // Measured against the scroller itself. offsetTop is counted from the
      // nearest positioned ancestor, which is not this panel, so the lit lines
      // were being scrolled to a position belonging to some other box — they
      // landed wherever that box happened to put them.
      const box = this.codeEl.getBoundingClientRect();
      const first = this.lineEls[regions[0].from].getBoundingClientRect();
      const last = this.lineEls[regions[0].to].getBoundingClientRect();
      const top = first.top - box.top + this.codeEl.scrollTop;
      const tall = last.bottom - first.top;
      // Centred when it fits, and started from the top when it does not:
      // centring a region taller than the panel hides the beginning of it.
      const want = tall < box.height
        ? top - (box.height - tall) / 2
        : top - 24;
      this.codeEl.scrollTo({
        top: Math.max(0, want),
        behavior: this._firstScrollDone ? 'smooth' : 'auto'
      });
      this._firstScrollDone = true;
    }

    // ---- the picture ----
    const iteration = this.iterations[this.cycle];
    let stage = (this.byIteration.get(iteration) || {})[step.stage];
    if (stage && step.solo) stage = this.soloStage(stage);
    if (stage) StepView.show(this.view, stage, { effect: step.effect || null });
  },

  frame(now) {
    const dt = Math.max(0, Math.min(0.05, (now - (this._last || now)) / 1000));
    this._last = now;
    if (this.active && this.view && this.stages) {
      StepView.tick(this.view, dt);
      StepView.gaze(this.view, now / 1000);
      StepView.mutating(this.view);
      StepView.draw(this.view, now / 1000, dt);

      // A step with two halves wears the emblem of the half being shown: the
      // game is swords while the tokens are crossing and a flag once the
      // brains start moving.
      const step = this.STEPS[this.at];
      const want = (step.emblemAfter && this.view._conquest)
        ? step.emblemAfter : step.emblem;
      if (want !== this._shownEmblem) {
        this._shownEmblem = want;
        this.noteEl.style.backgroundImage = this.emblem(want);
      }
    }
    requestAnimationFrame(t => this.frame(t));
  }
});
