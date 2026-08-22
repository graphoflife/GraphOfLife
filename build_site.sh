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

echo "built ${out}"
echo "  $(find "${out}" -type f | wc -l) files, $(du -sh "${out}" | cut -f1)"

if [ "${1:-}" = "--serve" ]; then
  echo "serving http://127.0.0.1:8000/ (no Python backend, exactly as a static host)"
  cd "${out}" && exec python3 -m http.server 8000 --bind 127.0.0.1
fi
