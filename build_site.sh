#!/usr/bin/env bash
#
# build_site.sh -- assemble the static site.
#
# The published site is the contents of web/ with the Python engine dropped in
# beside it, because in the browser the engine is just another file to fetch.
# The same script builds what GitHub Pages serves and what you can serve
# locally to check it, so there is no way for the deployed thing to differ
# from the thing that was tested.
#
#   ./build_site.sh              -> builds _site/
#   ./build_site.sh --serve      -> builds it and serves it on :8000
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${here}/_site"

rm -rf "${out}"
mkdir -p "${out}/py"

cp -r "${here}/web/." "${out}/"

# The engine, verbatim. Not a copy kept in step by hand: these are the files
# the desktop version runs, and the browser imports exactly them.
for f in GraphOfLifeSimple.py gol_config.py gol_series.py explain_minimal.py; do
  cp "${here}/${f}" "${out}/py/${f}"
done

# web/py already holds the two modules that only make sense in a browser —
# gol_browser.py and the gol_store stand-in — and cp -r brought them along.

# Stamp every script and stylesheet with the commit they came from.
#
# Without this a browser holding an older copy of js/app.js keeps using it: the
# page is new, the tabs are new, and the code behind them is last week's. That
# is not hypothetical — a tab shipped, worked when fetched fresh, and showed
# nothing at all to anyone who had visited before, because their app.js had no
# idea the tab existed. A changed URL is the only thing a cache cannot ignore.
stamp="$(git -C "${here}" rev-parse --short HEAD 2>/dev/null || date +%s)"
sed -i -E \
  -e "s#(<script src=\"js/[^\"?]+)\"#\1?v=${stamp}\"#g" \
  -e "s#(<link rel=\"stylesheet\" href=\"css/[^\"?]+)\"#\1?v=${stamp}\"#g" \
  "${out}/index.html"

# Everything else the page fetches by name from inside a script, where the tags
# in index.html cannot reach it: the two workers, what those workers import,
# the teaching script the Explanation walks through, and the recordings the
# pictures are drawn from. Each of these is a file a returning visitor can hold
# an old copy of while running new code around it — the Explanation reading
# last week's explain_minimal.py against this week's line anchors would light
# the wrong lines, and there would be nothing on screen to say why.
declare -A stamp_in=(
  ["${out}/js/layoutclient.js"]="js/layout-worker.js"
  ["${out}/js/layout-worker.js"]="force.js"
  ["${out}/js/api.js"]="js/sim-worker.js"
  ["${out}/js/sim-worker.js"]="runstore.js"
  ["${out}/js/explain.js"]="py/explain_minimal.py|data/explain-run.json"
  ["${out}/js/home.js"]="data/home-run.bin"
)
for file in "${!stamp_in[@]}"; do
  IFS='|' read -ra names <<< "${stamp_in[$file]}"
  for name in "${names[@]}"; do
    escaped="${name//./\\.}"
    grep -q "${name}" "${file}" || { echo "nothing to stamp: ${name} in ${file}"; exit 1; }
    sed -i -E "s#(['\"])(${escaped})\1#\1\2?v=${stamp}\1#g" "${file}"
  done
done

echo "  stamped assets with ${stamp}"

echo "built ${out}"
echo "  $(find "${out}" -type f | wc -l) files, $(du -sh "${out}" | cut -f1)"

if [ "${1:-}" = "--serve" ]; then
  echo "serving http://127.0.0.1:8000/ (no Python backend, exactly as a static host)"
  cd "${out}" && exec python3 -m http.server 8000 --bind 127.0.0.1
fi
