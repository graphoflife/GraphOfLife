#!/bin/bash
# Stop every pilot and queue started from this scratchpad, by PID, and nothing else.
for p in $(pgrep -f "^python3 (run|jobq)\.py"); do
  if [ "$(readlink /proc/$p/cwd)" = "/home/stefan/Documents/CodingProjects/GraphOfLife Claude/research/oee/pilots" ]; then kill $p; fi
done
