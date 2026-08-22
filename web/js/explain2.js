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
const Explain2 = {
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
   * `stage` names which recorded snapshot to draw; the three intro steps have
   * none of their own and borrow the first. `code` is a pair of strings found
   * in the script, marking the first and last line to light up.
   */
  STEPS: [
    {
      title: 'A brain',
      stage: 'repro.observe', intro: true,
      code: ['class Brain:', 'w += mask * np.random.randn'],
      text: `Every agent is a node holding some tokens and a small neural
        network. The network is never trained — there is no gradient anywhere
        in the script. It is made once at random, copied when an agent
        reproduces or conquers, and jittered a little each time it is copied.
        That is the whole of how behaviour ever changes.
        <p>One column of inputs goes in per thing being looked at, and one
        column of outputs comes back, so an agent reads its entire
        neighbourhood in a single pass.</p>`
    },
    {
      title: 'A world of them',
      stage: 'repro.observe', intro: true,
      code: ['# ---- the starting graph ----', 'self.next_id = AGENTS'],
      text: `A ring where everyone is joined to their nearest few, then a fifth
        of the links redrawn to somewhere random — short paths everywhere but
        still mostly local, which is the shape most real networks have.
        <p>Then the tokens: a fixed pile, split evenly. That total never
        changes again. Every rule after this only moves tokens from one agent
        to another, so the sum is the same on the last iteration as the
        first.</p>`
    },
    {
      title: 'And round it goes',
      stage: 'repro.observe', intro: true,
      code: ['    def step(self) -> None:', '        self.game()'],
      text: `One iteration is two phases. Reproduction, then the game. Each
        ends by clearing up the dead, and then it happens again.
        <p>Everything from here on is inside that loop, and the number beside
        each step says which iteration of the recorded run you are looking
        at.</p>`
    },

    {
      title: 'Observation — reproduction phase',
      stage: 'repro.observe',
      code: ['    def observe(self, u: int', 'return self.brains[u].forward(x)'],
      text: `Each agent opens its eye and reads its whole neighbourhood at
        once: its own tokens and degree, each neighbour's, and whatever was
        written to it last phase. Tokens and degrees go in logged, because
        what matters is the order of magnitude — the difference between 1 token
        and 10 is everything, between 500 and 510 nothing.
        <p>It observes itself too. That column is how it knows what it has.</p>`
    },
    {
      title: 'Everyone writes',
      stage: 'repro.observe',
      code: ['    def write_messages(self, u: int', "outbox.setdefault(v, {})[u] ="],
      text: `The same look also decides what to say. An agent writes a short
        note to every neighbour, and one to itself — and that note to itself is
        the only memory it has. Nothing else survives from one phase to the
        next.
        <p>Both ends of a link write, so every link carries a note each way.
        Nothing forces a message to mean anything; whatever they come to
        signal is whatever survives.</p>
        <p>The notes go into an outbox and are delivered when the phase ends,
        so every agent in a phase reads the same generation of messages. Doing
        it live meant an agent's inputs depended on where its id fell in the
        loop.</p>`
    },
    {
      title: 'Reproduction, and handover',
      stage: 'repro.born',
      code: ['# ---- how much of me goes into a child ----', 'self.unlink(u, v)'],
      text: `An agent decides what share of its tokens to spend on a child and
        pays the full price out of its own pocket. The child starts with
        exactly what was spent — no tokens are created.
        <p>The child inherits a mutated copy of the brain, and is wired to
        whichever of the parent's neighbours the parent picks. The parent can
        also <b>hand over</b> a connection instead of copying it: it drops that
        link and the child takes its place, so a lineage can pass on position
        and not just tokens. Handed links are drawn orange, newborns green.</p>`
    },
    {
      title: 'Elimination',
      stage: 'repro.cleanup',
      code: ['    def cleanup(self) -> None:', 'self.tokens[random.choice(survivors)] += 1'],
      text: `Two removals, in this order, and it runs after both phases.
        <p>Anyone holding nothing is gone. Then, of what is left, only the
        largest connected piece survives — and the second rule bites because of
        the first: a group hanging off the rest through a single agent comes
        adrift when that agent starves, however healthy the group itself is.
        Both happen in this very iteration.</p>
        <p>Everything the dead held is scattered over the survivors, so the
        total still comes to 500.</p>`
    },
    {
      title: 'Observation — game phase',
      stage: 'game.observe',
      code: ['    def game(self) -> None:', 'self.write_messages(u, targets, y, outbox)'],
      text: `The second phase begins the same way, and with one look, exactly
        like the first. What comes back decides both what the agent says and
        where it puts its tokens.
        <p>It reads a different graph from the one phase one read: children
        have been born, links have moved, and the starved are gone.</p>`
    },
    {
      title: 'The Colonel Blotto game',
      stage: 'game.stake',
      code: ['scores = np.asarray(y[10, :]', 'flow.get(frozenset((u, v)), 0)'],
      text: `Every agent stakes its <em>entire</em> pile across itself and its
        neighbours. It can spread the pile by score or put all of it on one
        node — which of those it does is the brain's own choice, not a rule.
        <p>Nothing is destroyed. A node's new balance is simply everything
        staked on it, which is why the total never moves.</p>`
    },
    {
      title: 'Who takes a node',
      stage: 'game.winner',
      code: ['def resolve(staked: dict', '    return hegemon'],
      text: `Usually the biggest stake wins. Sometimes the small ones combine
        and take it instead.
        <p>The largest single staker is the <b>hegemon</b>. Against it stands
        the <b>mob</b>: every other agent that flagged part of its stake as a
        revolt. The mob is sorted weakest first and walked upward, gathering a
        lower class. At each rung the question is whether that lower class now
        outweighs everyone still above it <em>plus</em> the hegemon.</p>
        <p>At the first rung where it does, the revolution carries — and the
        node goes to the <b>strongest staker in that rung</b>, not to a random
        member of the crowd. Ties at that exact amount are split by drawing;
        nothing else is. So a crowd can take a node from someone who outspent
        every one of them individually, and the best-placed of them collects
        it.</p>`
    },
    {
      title: 'The winner moves in',
      stage: 'game.conquer',
      code: ['# ---- the winner moves in ----', 'self.brains[winner].copy()'],
      text: `The node stays; whatever was thinking in it does not. The winner's
        brain is copied over the top.
        <p>This is the only place in the whole algorithm where a brain is
        selected. Phase one spreads brains around by reproduction; phase two
        decides which of them carry on.</p>`
    },
    {
      title: 'Quiet links are cut',
      stage: 'game.prune',
      code: ['# ---- links nobody used are cut ----', 'self.unlink(a, b)'],
      text: `A link that carried no tokens in either direction this phase is
        removed. The graph keeps only the connections somebody actually used,
        so the shape of the world is decided by where the tokens went rather
        than by anything structural.`
    },
    {
      title: 'Elimination, again',
      stage: 'game.cleanup',
      code: ['    def cleanup(self) -> None:', 'self.tokens[random.choice(survivors)] += 1'],
      text: `The same clearing-up as before, because it happens after both
        phases and not once an iteration. Cutting the quiet links is what
        usually strands somebody: an agent whose every connection went unused
        is left attached to nothing.`
    },
    {
      title: 'Everyone mutates',
      stage: 'game.mutate',
      code: ['# ---- everyone mutates ----', 'brain.mutate()'],
      text: `Every brain still standing is jittered — not only the newborns and
        not only the winners. Every agent, every iteration, whether it
        reproduced or fought or did nothing at all.
        <p>It happens after the clearing-up, so brains belonging to agents
        about to be removed are not jittered on their way out. This is the
        whole engine of variation; nothing else in the script ever changes a
        brain.</p>
        <p><b>And then it starts again</b>, back at the reproduction
        observation, on whatever world is left.</p>`
    }
  ]
};

Object.assign(Explain2, {
  init() {
    this.canvas = document.getElementById('explain2Canvas');
    if (!this.canvas) return;
    this.titleEl = document.getElementById('explain2Title');
    this.textEl = document.getElementById('explain2Text');
    this.codeEl = document.getElementById('explain2Code');
    this.countEl = document.getElementById('explain2Count');
    this.railEl = document.getElementById('explain2Rail');

    document.getElementById('explain2Prev').addEventListener('click', () => this.go(-1));
    document.getElementById('explain2Next').addEventListener('click', () => this.go(1));
    document.addEventListener('keydown', e => {
      if (App.view !== 'explain2') return;
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') { this.go(-1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { this.go(1); e.preventDefault(); }
    });

    this.railEl.innerHTML = this.STEPS.map((s, i) =>
      `<button class="explain2-dot" data-step="${i}" title="${s.title}"></button>`).join('');
    this.railEl.addEventListener('click', e => {
      const dot = e.target.closest('[data-step]');
      if (dot) this.showStep(Number(dot.dataset.step));
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

      this.renderCode();
      this.view = StepView.create(this.canvas);
      this.showStep(0, { carry: false });
    } catch (err) {
      this.titleEl.textContent = 'The walk-through could not be loaded';
      this.textEl.innerHTML = `<p>${err.message}</p>`;
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
      `<div class="explain2-line" data-line="${i}">` +
      `<span class="explain2-no">${i + 1}</span>` +
      `<span class="explain2-src">${html || '&nbsp;'}</span></div>`).join('');
    this.lineEls = [...this.codeEl.querySelectorAll('.explain2-line')];
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
  region(step) {
    const from = this.lines.findIndex(l => l.includes(step.code[0]));
    if (from < 0) return null;
    const rest = this.lines.slice(from);
    const offset = rest.findIndex((l, i) => i > 0 && l.includes(step.code[1]));
    if (offset < 0) return null;
    return { from, to: from + offset };
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

  showStep(index, { carry = true } = {}) {
    if (!this.stages) return;
    this.at = Math.max(0, Math.min(this.STEPS.length - 1, index));
    const step = this.STEPS[this.at];

    this.titleEl.textContent = step.title;
    this.textEl.innerHTML = `<p>${step.text}</p>`;
    this.countEl.textContent =
      `${this.at + 1} / ${this.STEPS.length}` +
      (step.intro ? '' : `  ·  iteration ${this.iterations[this.cycle]}`);

    for (const dot of this.railEl.children) {
      dot.classList.toggle('active', Number(dot.dataset.step) === this.at);
    }

    // ---- the code ----
    const region = this.region(step);
    this.lineEls.forEach((el, i) => {
      el.classList.toggle('lit', region ? (i >= region.from && i <= region.to) : false);
    });
    if (region) {
      const target = this.lineEls[region.from];
      const box = this.codeEl.getBoundingClientRect();
      this.codeEl.scrollTo({
        top: target.offsetTop - box.height * 0.28,
        behavior: this._firstScrollDone ? 'smooth' : 'auto'
      });
      this._firstScrollDone = true;
    }

    // ---- the picture ----
    const iteration = this.iterations[this.cycle];
    const stage = (this.byIteration.get(iteration) || {})[step.stage];
    if (stage) StepView.show(this.view, stage, { carry });
  },

  frame(now) {
    if (this.active && this.view && this.stages) {
      StepView.tick(this.view);
      StepView.draw(this.view, now / 1000);
    }
    requestAnimationFrame(t => this.frame(t));
  }
});
