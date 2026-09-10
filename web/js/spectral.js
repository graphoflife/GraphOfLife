/*
 * The spectral gap: one number for how hard the graph is to cut in half.
 *
 * λ₂ of the normalised Laplacian, on the largest connected component. Near zero
 * and there is a cheap cut somewhere; away from zero and every split is
 * expensive, which is what an expander is. It is the axis the Flow modules view
 * is read against — that method finds communities by compressing a random walk
 * and always returns *something*, so without λ₂ there is no telling a real
 * division from a line drawn through an indivisible graph.
 *
 * Lanczos on `L = I - D^-1/2 A D^-1/2` with the trivial eigenvector projected
 * out at every step, then the bottom of the little tridiagonal matrix by
 * bisection on Sturm's count. Power iteration was tried first and could not do
 * it: on a graph nearly in two pieces its estimate halved every time the
 * iteration count doubled, so the number it gave was a reading of the iteration
 * count.
 *
 * Mirrors gol_spectral.py step for step — the same start vector, the same
 * reorthogonalisation, the same bisection, the same constants. That is the
 * whole reason bisection is used rather than a library eigensolver, and
 * tests/test_stats_parity.py compares the two.
 */
const Spectral = {

  /** Matching the constants in gol_spectral.py, and there for the same reasons. */
  STEPS: 48,
  EXHAUSTED: 1e-12,
  SPECTRUM: [0, 2],
  BISECTIONS: 50,

  /**
   * λ₂ of the normalised Laplacian of the largest component, or null.
   *
   * Null when there is nothing to measure: fewer than three agents in the
   * biggest island, an isolated agent inside it, or a start vector that lands
   * on the trivial eigenvector. Mirrors spectral_gap() in gol_spectral.py.
   */
  gap(ids, adj) {
    // The biggest island, in a stable order so the arithmetic repeats.
    const seen = new Set();
    let best = [];
    for (const start of ids) {
      if (seen.has(start)) continue;
      const stack = [start];
      seen.add(start);
      const island = [];
      while (stack.length) {
        const node = stack.pop();
        island.push(node);
        for (const other of adj.get(node) || []) {
          if (!seen.has(other)) { seen.add(other); stack.push(other); }
        }
      }
      if (island.length > best.length) best = island;
    }
    best.sort((a, b) => a - b);

    const n = best.length;
    if (n < 3) return null;

    const at = new Map(best.map((node, i) => [node, i]));
    const neighbours = Array.from({ length: n }, () => []);
    for (const node of best) {
      const i = at.get(node);
      for (const other of adj.get(node) || []) {
        const j = at.get(other);
        if (j !== undefined && j !== i) neighbours[i].push(j);
      }
    }
    // Sorted, because floating-point addition is not associative and the two
    // implementations must add these up in the same order.
    for (const row of neighbours) row.sort((a, b) => a - b);

    const degree = neighbours.map(row => row.length);
    if (degree.some(d => d === 0)) return null;

    const inverseRoot = degree.map(d => 1 / Math.sqrt(d));
    let total = 0;
    for (const d of degree) total += d;
    const root = Math.sqrt(total);
    const trivial = degree.map(d => Math.sqrt(d) / root);

    const dot = (a, b) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; };

    /** L x, with the neighbours summed before the node's own weight applies. */
    const laplacian = (vector) => {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (const j of neighbours[i]) sum += vector[j] * inverseRoot[j];
        out[i] = vector[i] - inverseRoot[i] * sum;
      }
      return out;
    };

    // Integers over a power of two, so every entry is exact and identical to
    // the Python. See gol_spectral.py for why it is not trigonometric.
    let vector = new Float64Array(n);
    for (let i = 0; i < n; i++) vector[i] = (((i * 2654435761) % 4096) / 4096) - 0.5;
    let overlap = dot(vector, trivial);
    for (let i = 0; i < n; i++) vector[i] -= overlap * trivial[i];
    const length = Math.sqrt(dot(vector, vector));
    if (!(length > 0)) return null;
    for (let i = 0; i < n; i++) vector[i] /= length;

    const basis = [vector];
    const alpha = [], beta = [];
    const steps = Math.min(this.STEPS, n - 1);

    for (let step = 0; step < steps; step++) {
      const w = laplacian(basis[basis.length - 1]);
      const a = dot(basis[basis.length - 1], w);
      alpha.push(a);
      const last = basis[basis.length - 1];
      const before = basis[basis.length - 2];
      for (let i = 0; i < n; i++) {
        w[i] -= a * last[i];
        if (step) w[i] -= beta[beta.length - 1] * before[i];
      }

      // Reorthogonalised in full, and against the trivial direction first.
      // Rounding pulls the walk back toward the eigenvector it is meant to
      // have left, and letting it return would put zero back at the bottom of
      // the spectrum and report a gap of nothing on every graph.
      let against = dot(w, trivial);
      for (let i = 0; i < n; i++) w[i] -= against * trivial[i];
      for (const earlier of basis) {
        against = dot(w, earlier);
        for (let i = 0; i < n; i++) w[i] -= against * earlier[i];
      }

      const b = Math.sqrt(dot(w, w));
      if (b < this.EXHAUSTED) break;
      beta.push(b);
      const next = new Float64Array(n);
      for (let i = 0; i < n; i++) next[i] = w[i] / b;
      basis.push(next);
    }

    const gap = this.smallestEigenvalue(alpha, beta);
    return gap === null
      ? null
      : Math.min(this.SPECTRUM[1], Math.max(this.SPECTRUM[0], gap));
  },

  /**
   * The smallest eigenvalue of a symmetric tridiagonal, by bisection.
   *
   * Sturm's theorem gives, for any x, how many eigenvalues lie below it from a
   * single sweep of the diagonal. Halving the interval on that count finds the
   * bottom without ever forming the matrix — and does it identically in both
   * languages, which a library eigensolver would not.
   *
   * Mirrors smallest_eigenvalue() in gol_spectral.py.
   */
  smallestEigenvalue(alpha, beta) {
    const n = alpha.length;
    if (!n) return null;

    const below = (x) => {
      let count = 0;
      let d = alpha[0] - x;
      if (d < 0) count++;
      for (let i = 1; i < n; i++) {
        // A zero pivot is stepped over rather than divided by; the standard
        // dodge, and the count either side of it is unaffected.
        if (d === 0) d = 1e-300;
        d = (alpha[i] - x) - (beta[i - 1] * beta[i - 1]) / d;
        if (d < 0) count++;
      }
      return count;
    };

    let [lo, hi] = this.SPECTRUM;
    for (let i = 0; i < this.BISECTIONS; i++) {
      const mid = 0.5 * (lo + hi);
      if (below(mid) >= 1) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
  }
};
